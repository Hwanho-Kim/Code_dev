"""
1D Aqueous-Phase Chemistry Module.

Provides per-cell reaction rate computation for the 1D PDE solver.
Reuses Ver3's reaction system (109 reactions from Liu 2015) with
algebraic acid-base speciation.

Key difference from Ver3 chemistry.py:
- No ODE solver here; rates are computed per spatial cell
- The PDE solver (pde_solver.py) handles time integration
- Numba-JIT-accelerated compute_rates for 10-50x speedup
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import numba

from config_1d import (
    WATER, ACID_BASE_PAIRS, AQUEOUS_SPECIES,
    DEFAULTS, ODE_CONFIG,
    SALINE_SPECIES, SALINE_ACID_BASE_PAIRS,
)


# =============================================================================
# Numba-JIT Kernel (module-level, nopython)
# =============================================================================

# Encoding for reaction species references:
#   idx >= 0        → direct index into y_cell
#   idx = -(k+1)    → speciated species index k (0..N_spec-1)
#                      look up from speciated_conc array
# Encoding for target (where rate accumulates in dydt):
#   target >= 0     → direct index into dydt
#   target = -1     → skip (unknown species)

@numba.njit(cache=True)
def _compute_rates_kernel(
    y_cell,           # (N_s,) concentrations
    # Reaction arrays (all shape (N_rxn, max_r_or_p)):
    rxn_type,         # (N_rxn,) 0=irr, 1=rev
    rxn_k,            # (N_rxn,) rate constant for irr
    rxn_kf,           # (N_rxn,) forward rate for rev
    rxn_kb,           # (N_rxn,) backward rate for rev
    r_idx,            # (N_rxn, MAX_R) reactant species index (encoded)
    r_coeff,          # (N_rxn, MAX_R) reactant stoichiometric coeff
    r_target,         # (N_rxn, MAX_R) target index for reactant in dydt
    n_reactants,      # (N_rxn,) number of reactants
    p_idx,            # (N_rxn, MAX_P) product species index (encoded)
    p_coeff,          # (N_rxn, MAX_P) product stoichiometric coeff
    p_target,         # (N_rxn, MAX_P) target index for product in dydt
    n_products,       # (N_rxn,) number of products
    # Speciation arrays:
    n_pairs,          # int: number of acid-base pairs
    pair_total_idx,   # (n_pairs,) index of total variable in y_cell
    pair_Ka,          # (n_pairs,) Ka value
    H_idx,            # int: index of H+ in y_cell
    OH_idx,           # int: index of OH- in y_cell
    Kw,               # float: water autoionization constant
    # Constants:
    N_s,              # int: number of species
    trace,            # float: trace concentration
    max_rate,         # float: max rate clamp
    max_conc,         # float: max concentration
    # QSSA parameters (saline mode, 4-species):
    qssa_enabled,     # int64: 0=off, 1=on
    qssa_k,           # (41,) float64: rate constants
    qssa_idx,         # (13,) int64: species indices
    qssa_total_idx,   # (4,) int64: total variable indices
    qssa_Ka,          # (4,) float64: Ka values
    # Net effective rate:
    rxn_is_qssa,      # (N_rxn,) int8: 0=normal, 1=qssa-involved
):
    """Numba-JIT kernel for compute_rates. Returns dydt (N_s,)."""
    dydt = np.zeros(N_s, np.float64)

    # --- Sanitize y_cell ---
    yc = np.empty(N_s, np.float64)
    for i in range(N_s):
        v = y_cell[i]
        if v < trace:
            v = trace
        elif v > max_conc:
            v = max_conc
        yc[i] = v
    H = yc[H_idx]
    if H < 1e-14:
        H = 1e-14
        yc[H_idx] = H

    # --- Speciation ---
    n_spec = 2 * n_pairs + 1
    speciated_conc = np.empty(n_spec, np.float64)

    for p in range(n_pairs):
        C_total = yc[pair_total_idx[p]]
        if C_total < 0.0:
            C_total = 0.0
        Ka = pair_Ka[p]
        denom = H + Ka
        if denom < 1e-30:
            denom = 1e-30
        acid = C_total * H / denom
        base = C_total * Ka / denom
        if acid < trace:
            acid = trace
        if base < trace:
            base = trace
        speciated_conc[2 * p] = acid
        speciated_conc[2 * p + 1] = base

    oh_conc = Kw / H
    if oh_conc < trace:
        oh_conc = trace
    speciated_conc[2 * n_pairs] = oh_conc

    # --- 4-species QSSA: HOCl⁻, Cl₂⁻, Cl, HOClH (saline mode) ---
    if qssa_enabled > 0:
        i_HOClm = qssa_idx[0]; i_Cl2m  = qssa_idx[1]
        i_OH    = qssa_idx[2]; i_Cl    = qssa_idx[3]
        i_Clm   = qssa_idx[4]; i_HOClH = qssa_idx[6]
        i_Cl2   = qssa_idx[7]; i_Cl3m  = qssa_idx[8]
        i_H     = qssa_idx[9]; i_O3    = qssa_idx[10]
        i_ClO2  = qssa_idx[11]; i_NO3  = qssa_idx[12]

        OH_v = yc[i_OH]; Clm_v = yc[i_Clm]
        Cl2_v = yc[i_Cl2]; Cl3m_v = yc[i_Cl3m]
        H_v = yc[i_H]; O3_v = yc[i_O3]
        ClO2_v = yc[i_ClO2] if i_ClO2 >= 0 else trace
        NO3_v = yc[i_NO3] if i_NO3 >= 0 else trace
        # Old values for linearization of self-reactions
        Cl_old = yc[i_Cl]; Cl2m_old = yc[i_Cl2m]
        OHm_v = oh_conc  # already computed in speciation

        # Speciated co-reactants
        HO2_v = trace; O2m_v = trace; NO2m_v = trace
        H2O2_v = trace; HClO_v = trace; ClOm_v = trace
        it = qssa_total_idx[0]  # HO2_total
        if it >= 0:
            ct = yc[it]
            if ct < 0.0: ct = 0.0
            Ka_h = qssa_Ka[0]; den = H + Ka_h
            if den > 1e-30:
                HO2_v = ct * H / den
                O2m_v = ct * Ka_h / den
                if HO2_v < trace: HO2_v = trace
                if O2m_v < trace: O2m_v = trace
        it = qssa_total_idx[1]  # HONO_total
        if it >= 0:
            ct = yc[it]
            if ct < 0.0: ct = 0.0
            Ka_h = qssa_Ka[1]; den = H + Ka_h
            if den > 1e-30:
                NO2m_v = ct * Ka_h / den
                if NO2m_v < trace: NO2m_v = trace
        it = qssa_total_idx[2]  # H2O2_total
        if it >= 0:
            ct = yc[it]
            if ct < 0.0: ct = 0.0
            Ka_h = qssa_Ka[2]; den = H + Ka_h
            if den > 1e-30:
                H2O2_v = ct * H / den
                if H2O2_v < trace: H2O2_v = trace
        it = qssa_total_idx[3]  # HClO_total
        if it >= 0:
            ct = yc[it]
            if ct < 0.0: ct = 0.0
            Ka_h = qssa_Ka[3]; den = H + Ka_h
            if den > 1e-30:
                HClO_v = ct * H / den
                ClOm_v = ct * Ka_h / den
                if HClO_v < trace: HClO_v = trace
                if ClOm_v < trace: ClOm_v = trace

        # Rate constants from qssa_k array (indices 0..30 same as before)
        k3f=qssa_k[0]; k3b=qssa_k[1]; k4f=qssa_k[2]; k4b=qssa_k[3]
        k5f=qssa_k[4]; k5b=qssa_k[5]; k6f=qssa_k[6]; k6b=qssa_k[7]
        k7f=qssa_k[8]; k7b=qssa_k[9]; k8f=qssa_k[10]; k8b=qssa_k[11]
        k9f=qssa_k[12]; k9b=qssa_k[13]; k24=qssa_k[14]
        k28=qssa_k[15]; k29=qssa_k[16]
        k32=qssa_k[17]; k33=qssa_k[18]; k34=qssa_k[19]
        k35=qssa_k[20]; k36=qssa_k[21]; k37=qssa_k[22]
        k38=qssa_k[23]; k39=qssa_k[24]; k40=qssa_k[25]
        k41=qssa_k[26]; k42=qssa_k[27]
        k52=qssa_k[28]; k54=qssa_k[29]; k69=qssa_k[30]
        # New for 4-species (indices 31..40)
        k23=qssa_k[31]; k25=qssa_k[32]; k26=qssa_k[33]
        k27=qssa_k[34]; k43=qssa_k[35]; k48=qssa_k[36]
        k50=qssa_k[37]; k53=qssa_k[38]; k55=qssa_k[39]
        k61=qssa_k[40]

        # === Picard iteration for nonlinear QSSA self-terms ===
        for _qssa_iter in range(20):
            # Save old values for convergence check
            prev_x1 = yc[i_HOClm]; prev_x2 = yc[i_Cl2m]
            prev_x3 = yc[i_Cl];    prev_x4 = yc[i_HOClH]
            # Update linearization points from current estimates
            Cl_old = yc[i_Cl]
            Cl2m_old = yc[i_Cl2m]

            # === Cl (x3) equation: P3 + a31*x1 + a32*x2 - alpha3*x3 = 0 ===
            P3 = k48*HClO_v*HO2_v + k55*NO3_v*Clm_v + k61*ClO2_v
            a31 = k5b * H   # HOCl- -> Cl (S5 reverse)
            a32 = k6b + k41*O3_v  # Cl2- -> Cl (S6 reverse + S41)
            alpha3 = (k5f + k6f*Clm_v + 2.0*k23*Cl_old
                      + k24*OHm_v + k25*HO2_v + k26*H2O2_v
                      + k27*H_v + k33*Cl2m_old + k43*ClOm_v
                      + k50*HClO_v + k53*NO2m_v)
            if alpha3 < 1e-30: alpha3 = 1e-30

            # === HOClH (x4) equation ===
            P4 = 0.0
            a41 = k7b * H   # HOCl- -> HOClH (S7 reverse)
            a42 = k8f        # Cl2- -> HOClH (S8 forward)
            alpha4 = k7f + k8b*Clm_v
            if alpha4 < 1e-30: alpha4 = 1e-30

            # === HOCl- (x1) equation coefficients ===
            P1_ext = k4f * OH_v * Clm_v
            a12 = (k9f + k3b) * OHm_v
            a13 = k5f + k24 * OHm_v
            a14 = k7f
            alpha1 = k4b + k5b*H + k7b*H + (k3f + k9b)*Clm_v

            # Substitute x3, x4 into x1 equation
            P1_mod = P1_ext + a13*P3/alpha3 + a14*P4/alpha4
            alpha1_mod = alpha1 - a13*a31/alpha3 - a14*a41/alpha4
            a12_mod = a12 + a13*a32/alpha3 + a14*a42/alpha4

            # === Cl2- (x2) equation coefficients ===
            P2_ext = k28*Cl2_v*HO2_v + k29*Cl2_v*O2m_v + k52*Cl3m_v*H_v
            a21 = (k3f + k9b) * Clm_v
            a23 = k6f * Clm_v
            a24 = k8b * Clm_v
            alpha2 = (k6b + k8f + (k9f + k3b)*OHm_v
                      + 2.0*(k32+k34)*Cl2m_old
                      + k33*Cl_old + (k35+k36)*H2O2_v
                      + k37*HO2_v + k38*O2m_v + k39*OH_v
                      + k40*OHm_v + k41*O3_v + k42*H_v
                      + k54*NO2m_v + k69*ClO2_v)

            # Substitute x3, x4 into x2 equation
            P2_mod = P2_ext + a23*P3/alpha3 + a24*P4/alpha4
            a21_mod = a21 + a23*a31/alpha3 + a24*a41/alpha4
            alpha2_mod = alpha2 - a23*a32/alpha3 - a24*a42/alpha4

            # === Solve modified 2x2 system ===
            det = alpha1_mod*alpha2_mod - a12_mod*a21_mod
            if det > 1e-30 or det < -1e-30:
                x1 = (P1_mod*alpha2_mod + a12_mod*P2_mod) / det
                x2 = (alpha1_mod*P2_mod + a21_mod*P1_mod) / det
                x3 = (P3 + a31*x1 + a32*x2) / alpha3
                x4 = (P4 + a41*x1 + a42*x2) / alpha4
                if x1 < trace: x1 = trace
                if x2 < trace: x2 = trace
                if x3 < trace: x3 = trace
                if x4 < trace: x4 = trace
                yc[i_HOClm] = x1
                yc[i_Cl2m] = x2
                yc[i_Cl] = x3
                yc[i_HOClH] = x4

                # Convergence check: max relative change < 1e-12
                max_rel = 0.0
                if prev_x1 > trace:
                    r = abs(x1 - prev_x1) / prev_x1
                    if r > max_rel: max_rel = r
                if prev_x2 > trace:
                    r = abs(x2 - prev_x2) / prev_x2
                    if r > max_rel: max_rel = r
                if prev_x3 > trace:
                    r = abs(x3 - prev_x3) / prev_x3
                    if r > max_rel: max_rel = r
                if prev_x4 > trace:
                    r = abs(x4 - prev_x4) / prev_x4
                    if r > max_rel: max_rel = r
                if max_rel < 1e-12:
                    break

    # --- QSSA net effective rate: analytical S3-S9 + irreversible loop ---
    if qssa_enabled > 0:
        i_HOClm = qssa_idx[0]; i_Cl2m  = qssa_idx[1]
        i_Cl    = qssa_idx[3]; i_HOClH = qssa_idx[6]
        i_Clm   = qssa_idx[4]; i_OH_q  = qssa_idx[2]
        i_H_q   = qssa_idx[9]

        x1 = yc[i_HOClm]; x2 = yc[i_Cl2m]
        x3 = yc[i_Cl];    x4 = yc[i_HOClH]
        OH_v  = yc[i_OH_q]; Clm_v = yc[i_Clm]
        OHm_v = oh_conc

        k3f=qssa_k[0]; k3b=qssa_k[1]; k4f=qssa_k[2]; k4b=qssa_k[3]
        k5f=qssa_k[4]; k5b=qssa_k[5]; k6f=qssa_k[6]; k6b=qssa_k[7]
        k7f=qssa_k[8]; k7b=qssa_k[9]; k8f=qssa_k[10]; k8b=qssa_k[11]
        k9f=qssa_k[12]; k9b=qssa_k[13]

        # S3-S9 analytical net contributions to non-QSSA species
        # dydt[Cl⁻] from S3,S4,S6,S8,S9:
        dydt[i_Clm] += (
            - k4f * OH_v * Clm_v
            + (k4b - (k3f + k9b) * Clm_v) * x1
            + ((k3b + k9f) * OHm_v + k6b + k8f) * x2
            - k6f * Clm_v * x3
            - k8b * Clm_v * x4
        )
        # dydt[OH] from S4:
        dydt[i_OH_q] += -k4f * OH_v * Clm_v + k4b * x1
        # dydt[H⁺] from S5,S7 (H is the H⁺ concentration from line 89):
        dydt[H_idx] += -(k5b + k7b) * H * x1 + k5f * x3 + k7f * x4

        # S23-S69 irreversible: loop over tagged reactions (non-QSSA targets only)
        for ri in range(rxn_type.shape[0]):
            if rxn_is_qssa[ri] == 0:
                continue
            if rxn_type[ri] == 1:
                continue  # S3-S9 already handled analytically
            nr = n_reactants[ri]
            rate = rxn_k[ri]
            overflow = False
            for rr in range(nr):
                idx = r_idx[ri, rr]
                if idx >= 0:
                    conc = yc[idx]
                elif idx >= -(2 * n_pairs + 1):
                    conc = speciated_conc[-(idx + 1)]
                else:
                    conc = 0.0
                rate *= conc ** min(r_coeff[ri, rr], 3.0)
                if rate > 1e15:
                    overflow = True
                    break
            if overflow or rate < 1e-30:
                continue
            # Apply to non-QSSA targets only
            for rr in range(nr):
                t_idx = r_target[ri, rr]
                if (t_idx >= 0 and t_idx != i_HOClm and t_idx != i_Cl2m
                        and t_idx != i_Cl and t_idx != i_HOClH):
                    dydt[t_idx] -= r_coeff[ri, rr] * rate
            np_ = n_products[ri]
            for pp in range(np_):
                t_idx = p_target[ri, pp]
                if (t_idx >= 0 and t_idx != i_HOClm and t_idx != i_Cl2m
                        and t_idx != i_Cl and t_idx != i_HOClH):
                    dydt[t_idx] += p_coeff[ri, pp] * rate

    # --- Process reactions (skip QSSA-tagged when enabled) ---
    N_rxn = rxn_type.shape[0]

    for ri in range(N_rxn):
        if qssa_enabled > 0 and rxn_is_qssa[ri] > 0:
            continue
        nr = n_reactants[ri]
        np_ = n_products[ri]

        if rxn_type[ri] == 0:
            # Irreversible
            rate = rxn_k[ri]
            overflow = False
            for rr in range(nr):
                idx = r_idx[ri, rr]
                coeff = r_coeff[ri, rr]
                if idx >= 0:
                    conc = yc[idx]
                elif idx >= -(2 * n_pairs + 1):
                    conc = speciated_conc[-(idx + 1)]
                else:
                    conc = 0.0
                c_min = min(coeff, 3.0)
                rate *= conc ** c_min
                if rate > 1e15:
                    overflow = True
                    break
            if overflow:
                continue

        else:
            # Reversible
            rate_f = rxn_kf[ri]
            rate_b = rxn_kb[ri]

            for rr in range(nr):
                idx = r_idx[ri, rr]
                coeff = r_coeff[ri, rr]
                if idx >= 0:
                    conc = yc[idx]
                elif idx >= -(2 * n_pairs + 1):
                    conc = speciated_conc[-(idx + 1)]
                else:
                    conc = 0.0
                rate_f *= conc ** min(coeff, 3.0)

            for pp in range(np_):
                idx = p_idx[ri, pp]
                coeff = p_coeff[ri, pp]
                if idx >= 0:
                    conc = yc[idx]
                elif idx >= -(2 * n_pairs + 1):
                    conc = speciated_conc[-(idx + 1)]
                else:
                    conc = 0.0
                rate_b *= conc ** min(coeff, 3.0)

            rate = rate_f - rate_b

            if abs(rate) > 1e15:
                continue

        if abs(rate) < 1e-30:
            continue

        # Apply rate to dydt
        for rr in range(nr):
            t_idx = r_target[ri, rr]
            if t_idx >= 0:
                dydt[t_idx] -= r_coeff[ri, rr] * rate

        for pp in range(np_):
            t_idx = p_target[ri, pp]
            if t_idx >= 0:
                dydt[t_idx] += p_coeff[ri, pp] * rate

    # OH- algebraic
    if OH_idx >= 0:
        dydt[OH_idx] = 0.0

    # QSSA species: dydt is naturally zero because QSSA reactions are
    # skipped in the main loop and handled via net effective rate.
    # No explicit zeroing needed.

    # Sanitize output
    for i in range(N_s):
        v = dydt[i]
        if v != v:  # NaN check
            v = 0.0
        elif v > max_rate:
            v = max_rate
        elif v < -max_rate:
            v = -max_rate
        dydt[i] = v

    return dydt


# =============================================================================
# Reaction Loader (identical to Ver3)
# =============================================================================

class ReactionLoader:
    """Load and parse reaction definitions from YAML file."""

    @staticmethod
    def load_reactions(yaml_path: Path) -> List[Dict]:
        """Load reactions from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        reactions = []
        categories = [
            'no_producing_reactions',
            'no2_producing_reactions',
            'o3_reactions',
            'hydrolysis_reactions',
            'oh_reactions',
            'h_atom_reactions',
            'ho2_reactions',
            'peroxynitrite_reactions',
            'additional_reactions',
            'reversible_reactions',
            'irreversible_reactions',
        ]

        for category in categories:
            if category in data:
                for rxn in data[category]:
                    converted = ReactionLoader._convert_reaction(rxn)
                    reactions.append(converted)

        return reactions

    @staticmethod
    def _convert_reaction(rxn: Dict) -> Dict:
        """Convert YAML reaction format to internal format."""
        if rxn['type'] == 'reversible':
            return {
                'type': 'rev',
                'reactants': rxn['reactants'],
                'products': rxn.get('products', {}),
                'k_f': rxn['k_f'],
                'k_b': rxn['k_b'],
                'label': rxn['label']
            }
        else:
            return {
                'type': 'irr',
                'reactants': rxn['reactants'],
                'products': rxn.get('products', {}),
                'k': rxn['k'],
                'label': rxn['label']
            }


# =============================================================================
# Species-to-total mapping
# =============================================================================

def get_species_to_total_map(pairs: Dict = None) -> Dict[str, str]:
    if pairs is None:
        pairs = ACID_BASE_PAIRS
    species_to_total: Dict[str, str] = {}
    for total_name, (acid, base, _) in pairs.items():
        species_to_total[acid] = total_name
        species_to_total[base] = total_name
    return species_to_total


# =============================================================================
# Chemistry System
# =============================================================================

class AqueousChemistry1D:
    """
    Aqueous chemistry system for 1D model.

    Computes reaction rates per spatial cell. Does NOT do time integration —
    that is handled by PDESolver1D.
    """

    def __init__(self, reactions_file: Optional[Path] = None,
                 saline_mode: bool = False):
        self.Kw = WATER.KW
        self.saline_mode = saline_mode

        if saline_mode:
            self.aqueous_species = list(AQUEOUS_SPECIES) + list(SALINE_SPECIES)
            self.pKa_map = {**ACID_BASE_PAIRS, **SALINE_ACID_BASE_PAIRS}
        else:
            self.aqueous_species = list(AQUEOUS_SPECIES)
            self.pKa_map = dict(ACID_BASE_PAIRS)

        self.species_to_total = get_species_to_total_map(self.pKa_map)

        self.species_idx = {sp: i for i, sp in enumerate(self.aqueous_species)}
        self.n_species = len(self.aqueous_species)

        self.trace = DEFAULTS.trace_concentration
        self.max_rate = ODE_CONFIG.max_rate

        self._load_reactions(reactions_file)
        self._precompute_reaction_data()
        self._precompute_numba_arrays()

        if saline_mode:
            self._precompute_qssa_params()

    def _load_reactions(self, reactions_file: Optional[Path] = None):
        if reactions_file is None:
            reactions_file = Path(__file__).parent / 'reactions_full.yaml'
        self.reactions = ReactionLoader.load_reactions(reactions_file)

        if self.saline_mode:
            saline_file = Path(__file__).parent / 'reactions_saline.yaml'
            if saline_file.exists():
                saline_rxns = ReactionLoader.load_reactions(saline_file)
                # K_CAP disabled: QSSA handles stiffness from fast Cl reactions.
                # Original rate constants preserved for physical accuracy.
                # K_CAP = 1e6
                # for rxn in saline_rxns:
                #     if rxn['type'] == 'rev':
                #         kf, kb = float(rxn['k_f']), float(rxn['k_b'])
                #         if kf > K_CAP or kb > K_CAP:
                #             scale = max(kf, kb) / K_CAP
                #             rxn['k_f'] = kf / scale
                #             rxn['k_b'] = kb / scale
                #     elif 'k' in rxn and float(rxn['k']) > K_CAP:
                #         rxn['k'] = K_CAP
                self.reactions.extend(saline_rxns)

    def _precompute_reaction_data(self):
        """Pre-compute index-based reaction data for vectorized evaluation."""
        self._rxn_data = []

        for rxn in self.reactions:
            # Map reactant/product names to indices (or mark as speciated)
            reactant_info = []
            for sp, coeff in rxn['reactants'].items():
                idx = self._resolve_species_index(sp)
                reactant_info.append((sp, coeff, idx))

            product_info = []
            for sp, coeff in rxn.get('products', {}).items():
                idx = self._resolve_species_index(sp)
                product_info.append((sp, coeff, idx))

            if rxn['type'] == 'irr':
                self._rxn_data.append({
                    'type': 'irr',
                    'k': float(rxn['k']),
                    'reactants': reactant_info,
                    'products': product_info,
                })
            else:
                self._rxn_data.append({
                    'type': 'rev',
                    'k_f': float(rxn['k_f']),
                    'k_b': float(rxn['k_b']),
                    'reactants': reactant_info,
                    'products': product_info,
                })

    def _resolve_species_index(self, species: str) -> int:
        """
        Resolve species name to index in y vector.

        For speciated species (e.g., HONO, NO2-), returns -1
        to indicate they must be looked up from the speciated dict.
        """
        if species in self.species_idx:
            return self.species_idx[species]
        # Check if it's an acid/base that needs speciation
        if species in self.species_to_total:
            return -1  # Will use speciated dict
        return -2  # Unknown species (skip)

    def speciate(self, y_cell: np.ndarray) -> Dict[str, float]:
        """
        Compute individual acid/base concentrations from totals.

        Parameters
        ----------
        y_cell : ndarray, shape (n_species,)
            Concentration vector for one grid cell.

        Returns
        -------
        dict
            {species_name: concentration} for all speciated species.
        """
        speciated: Dict[str, float] = {}
        H = max(float(y_cell[self.species_idx['H+']]), 1e-14)

        for total_name, (HA_name, A_name, pKa) in self.pKa_map.items():
            if total_name not in self.species_idx:
                continue
            C_total = max(float(y_cell[self.species_idx[total_name]]), 0.0)
            Ka = 10.0 ** (-pKa)
            denom = H + Ka
            if denom < 1e-30:
                denom = 1e-30
            speciated[HA_name] = max(C_total * H / denom, self.trace)
            speciated[A_name] = max(C_total * Ka / denom, self.trace)

        speciated['OH-'] = max(self.Kw / H, self.trace)
        return speciated

    def compute_rates(self, y_cell: np.ndarray) -> np.ndarray:
        """
        Compute reaction rates for one grid cell.

        Parameters
        ----------
        y_cell : ndarray, shape (n_species,)
            Concentration vector for one grid cell.

        Returns
        -------
        ndarray, shape (n_species,)
            Rate of change from reactions only (no diffusion).
        """
        dydt = np.zeros(self.n_species, dtype=np.float64)

        # Sanitize
        y_cell = np.clip(y_cell, self.trace, ODE_CONFIG.max_concentration)
        H_idx = self.species_idx['H+']
        y_cell[H_idx] = max(y_cell[H_idx], 1e-14)

        # QSSA for saline fast intermediates
        if self.saline_mode and hasattr(self, '_qssa'):
            self.apply_qssa(y_cell)

        # Speciate
        speciated = self.speciate(y_cell)

        # QSSA net effective rate (analytical S3-S9 + irreversible loop)
        _qssa_on = self.saline_mode and hasattr(self, '_qssa')
        if _qssa_on:
            q = self._qssa
            x1 = y_cell[q['idx_HOCl-']]; x2 = y_cell[q['idx_Cl2-']]
            x3 = y_cell[q['idx_Cl']];    x4 = y_cell[q['idx_HOClH']]
            OH_v = y_cell[q['idx_OH']]; Clm_v = y_cell[q['idx_Cl-']]
            H_v = y_cell[q['idx_H+']]; OHm_v = speciated.get('OH-', self.trace)
            k3f = q['S3_kf']; k3b = q['S3_kb']; k4f = q['S4_kf']; k4b = q['S4_kb']
            k5f = q['S5_kf']; k5b = q['S5_kb']; k6f = q['S6_kf']; k6b = q['S6_kb']
            k7f = q['S7_kf']; k7b = q['S7_kb']; k8f = q['S8_kf']; k8b = q['S8_kb']
            k9f = q['S9_kf']; k9b = q['S9_kb']
            # S3-S9 analytical net contributions
            dydt[q['idx_Cl-']] += (
                - k4f * OH_v * Clm_v
                + (k4b - (k3f + k9b) * Clm_v) * x1
                + ((k3b + k9f) * OHm_v + k6b + k8f) * x2
                - k6f * Clm_v * x3
                - k8b * Clm_v * x4
            )
            dydt[q['idx_OH']] += -k4f * OH_v * Clm_v + k4b * x1
            dydt[q['idx_H+']] += -(k5b + k7b) * H_v * x1 + k5f * x3 + k7f * x4
            # QSSA species set for target filtering
            _qssa_idx_set = {q['idx_HOCl-'], q['idx_Cl2-'], q['idx_Cl'], q['idx_HOClH']}

        # Process reactions (skip QSSA-tagged when enabled)
        for ri, rxn_d in enumerate(self._rxn_data):
            if _qssa_on and self._nb_rxn_is_qssa[ri] > 0:
                # Irreversible QSSA reactions: apply to non-QSSA targets only
                if rxn_d['type'] == 'irr':
                    rate = self._compute_single_rate(rxn_d, y_cell, speciated)
                    if abs(rate) < 1e-30:
                        continue
                    self._apply_rate_filtered(rxn_d, rate, dydt, _qssa_idx_set)
                continue  # skip reversible (already analytical)
            rate = self._compute_single_rate(rxn_d, y_cell, speciated)
            if abs(rate) < 1e-30:
                continue
            self._apply_rate(rxn_d, rate, dydt)

        # OH- is algebraic, not evolved by ODE
        if 'OH-' in self.species_idx:
            dydt[self.species_idx['OH-']] = 0.0

        # Sanitize output
        dydt = np.nan_to_num(dydt, nan=0.0, posinf=0.0, neginf=0.0)
        dydt = np.clip(dydt, -self.max_rate, self.max_rate)

        return dydt

    def _compute_single_rate(
        self, rxn_d: Dict, y_cell: np.ndarray, speciated: Dict[str, float]
    ) -> float:
        """Compute rate for a single reaction."""
        if rxn_d['type'] == 'irr':
            rate = rxn_d['k']
            for sp_name, coeff, idx in rxn_d['reactants']:
                conc = self._get_conc(sp_name, idx, y_cell, speciated)
                rate *= conc ** min(coeff, 3)
                if rate > 1e15 or not np.isfinite(rate):
                    return 0.0

        else:  # reversible
            rate_f = rxn_d['k_f']
            rate_b = rxn_d['k_b']

            for sp_name, coeff, idx in rxn_d['reactants']:
                conc = self._get_conc(sp_name, idx, y_cell, speciated)
                rate_f *= conc ** min(coeff, 3)

            for sp_name, coeff, idx in rxn_d['products']:
                conc = self._get_conc(sp_name, idx, y_cell, speciated)
                rate_b *= conc ** min(coeff, 3)

            rate = rate_f - rate_b

            if not np.isfinite(rate) or abs(rate) > 1e15:
                return 0.0

        return rate

    def _get_conc(
        self, sp_name: str, idx: int, y_cell: np.ndarray,
        speciated: Dict[str, float]
    ) -> float:
        """Get concentration from y_cell or speciated dict."""
        if idx >= 0:
            return float(y_cell[idx])
        elif idx == -1:
            return speciated.get(sp_name, self.trace)
        return 0.0

    def _apply_rate(self, rxn_d: Dict, rate: float, dydt: np.ndarray):
        """Apply reaction rate to dydt vector."""
        # Reactants consumed
        for sp_name, coeff, idx in rxn_d['reactants']:
            target_idx = self._get_target_idx(sp_name, idx)
            if target_idx >= 0:
                dydt[target_idx] -= coeff * rate

        # Products formed
        for sp_name, coeff, idx in rxn_d['products']:
            target_idx = self._get_target_idx(sp_name, idx)
            if target_idx >= 0:
                dydt[target_idx] += coeff * rate

    def _get_target_idx(self, sp_name: str, idx: int) -> int:
        """Get the target index in dydt for a species."""
        if idx >= 0:
            return idx
        # Speciated species → map to total
        if sp_name in self.species_to_total:
            total_name = self.species_to_total[sp_name]
            return self.species_idx.get(total_name, -1)
        return -1

    def _apply_rate_filtered(self, rxn_d: Dict, rate: float,
                             dydt: np.ndarray, skip_set: set):
        """Apply reaction rate, skipping QSSA target indices."""
        for sp_name, coeff, idx in rxn_d['reactants']:
            target_idx = self._get_target_idx(sp_name, idx)
            if target_idx >= 0 and target_idx not in skip_set:
                dydt[target_idx] -= coeff * rate
        for sp_name, coeff, idx in rxn_d['products']:
            target_idx = self._get_target_idx(sp_name, idx)
            if target_idx >= 0 and target_idx not in skip_set:
                dydt[target_idx] += coeff * rate

    # =================================================================
    # QSSA for fast Cl intermediates (saline mode)
    # =================================================================

    def _precompute_qssa_params(self):
        """Pre-compute QSSA parameters for 4-species system.

        QSSA species: HOCl⁻, Cl₂⁻, Cl (radical), HOClH.
        Extracts K_CAP-scaled rate constants from self._rxn_data for
        all reactions involving these species, plus species indices
        needed at runtime to evaluate the 4→2→2 linear QSSA system.

        The 4×4 system reduces to 2×2 because Cl and HOClH can be
        expressed as linear functions of HOCl⁻ and Cl₂⁻:
          x₃(Cl)    = (P₃ + a₃₁·x₁ + a₃₂·x₂) / α₃
          x₄(HOClH) = (P₄ + a₄₁·x₁ + a₄₂·x₂) / α₄
        Then substitute into the HOCl⁻/Cl₂⁻ equations for modified 2×2.
        """
        # Build reaction-ID → rxn_data map from labels
        rxn_map = {}
        for i, rxn in enumerate(self.reactions):
            label = rxn.get('label', '')
            rxn_id = label.split(':')[0].strip()
            if rxn_id:
                rxn_map[rxn_id] = self._rxn_data[i]

        def _kf_kb(sid):
            rd = rxn_map.get(sid)
            if rd is None:
                return 0.0, 0.0
            if rd['type'] == 'rev':
                return rd['k_f'], rd['k_b']
            return 0.0, 0.0

        def _k_irr(sid):
            rd = rxn_map.get(sid)
            if rd is None:
                return 0.0
            return rd['k'] if rd['type'] == 'irr' else 0.0

        q = {}

        # Reversible rate constants (K_CAP-scaled)
        for sid in ('S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'):
            kf, kb = _kf_kb(sid)
            q[f'{sid}_kf'] = kf
            q[f'{sid}_kb'] = kb

        # Irreversible rate constants (K_CAP-capped)
        # Original 2-species set:
        for sid in ('S24', 'S28', 'S29',
                    'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38',
                    'S39', 'S40', 'S41', 'S42',
                    'S52', 'S54', 'S69'):
            q[f'{sid}_k'] = _k_irr(sid)
        # New for 4-species (Cl destruction/production):
        for sid in ('S23', 'S25', 'S26', 'S27', 'S43', 'S48', 'S50',
                    'S53', 'S55', 'S61'):
            q[f'{sid}_k'] = _k_irr(sid)

        # Species indices (all present in saline mode)
        for name in ('HOCl-', 'Cl2-', 'OH', 'Cl', 'Cl-', 'H+',
                     'HOClH', 'Cl2', 'Cl3-', 'H', 'O3', 'ClO2',
                     'NO3'):
            q[f'idx_{name}'] = self.species_idx.get(name, -1)

        # Total-variable indices for speciation at runtime
        q['idx_HO2_total'] = self.species_idx.get('HO2_total', -1)
        q['idx_HONO_total'] = self.species_idx.get('HONO_total', -1)
        q['idx_H2O2_total'] = self.species_idx.get('H2O2_total', -1)
        q['idx_HClO_total'] = self.species_idx.get('HClO_total', -1)

        # pKa for inline speciation
        q['Ka_HO2'] = 10.0 ** (-4.8)
        q['Ka_HONO'] = 10.0 ** (-3.4)
        q['Ka_H2O2'] = 10.0 ** (-11.65)
        q['Ka_HClO'] = 10.0 ** (-7.5)

        self._qssa = q

        # --- Numba arrays for kernel-integrated QSSA ---
        # qssa_k layout (41 values):
        #   [0..1]  k3f,k3b    [2..3]  k4f,k4b    [4..5]  k5f,k5b
        #   [6..7]  k6f,k6b    [8..9]  k7f,k7b    [10..11] k8f,k8b
        #   [12..13] k9f,k9b   [14] k24  [15] k28  [16] k29
        #   [17..27] k32..k42  [28] k52  [29] k54  [30] k69
        #   -- new for 4-species Cl/HOClH --
        #   [31] k23  [32] k25  [33] k26  [34] k27  [35] k43
        #   [36] k48  [37] k50  [38] k53  [39] k55  [40] k61
        self._nb_qssa_enabled = np.int64(1)
        self._nb_qssa_k = np.array([
            q['S3_kf'], q['S3_kb'], q['S4_kf'], q['S4_kb'],
            q['S5_kf'], q['S5_kb'], q['S6_kf'], q['S6_kb'],
            q['S7_kf'], q['S7_kb'], q['S8_kf'], q['S8_kb'],
            q['S9_kf'], q['S9_kb'], q['S24_k'],
            q['S28_k'], q['S29_k'],
            q['S32_k'], q['S33_k'], q['S34_k'],
            q['S35_k'], q['S36_k'], q['S37_k'],
            q['S38_k'], q['S39_k'], q['S40_k'],
            q['S41_k'], q['S42_k'],
            q['S52_k'], q['S54_k'], q['S69_k'],
            # New for 4-species:
            q['S23_k'], q['S25_k'], q['S26_k'], q['S27_k'], q['S43_k'],
            q['S48_k'], q['S50_k'], q['S53_k'], q['S55_k'], q['S61_k'],
        ], dtype=np.float64)
        # qssa_idx (14 values):
        #   [0] HOCl⁻  [1] Cl₂⁻  [2] OH    [3] Cl    [4] Cl⁻
        #   [5] H+     [6] HOClH  [7] Cl₂   [8] Cl₃⁻  [9] H
        #   [10] O₃    [11] ClO₂  [12] NO₃  [13] (reserved)
        self._nb_qssa_idx = np.array([
            q['idx_HOCl-'], q['idx_Cl2-'], q['idx_OH'], q['idx_Cl'],
            q['idx_Cl-'], q['idx_H+'], q['idx_HOClH'], q['idx_Cl2'],
            q['idx_Cl3-'], q['idx_H'], q['idx_O3'], q['idx_ClO2'],
            q['idx_NO3'],
        ], dtype=np.int64)
        # qssa_total_idx (4 values): HO2_total, HONO_total, H2O2_total, HClO_total
        self._nb_qssa_total_idx = np.array([
            q['idx_HO2_total'], q['idx_HONO_total'],
            q['idx_H2O2_total'], q['idx_HClO_total'],
        ], dtype=np.int64)
        # qssa_Ka (4 values): Ka_HO2, Ka_HONO, Ka_H2O2, Ka_HClO
        self._nb_qssa_Ka = np.array([
            q['Ka_HO2'], q['Ka_HONO'], q['Ka_H2O2'], q['Ka_HClO'],
        ], dtype=np.float64)

        # --- Tag reactions involving QSSA species (for net effective rate) ---
        qssa_set = {q['idx_HOCl-'], q['idx_Cl2-'], q['idx_Cl'], q['idx_HOClH']}
        qssa_set.discard(-1)
        N_rxn = self._nb_rxn_type.shape[0]
        rxn_is_qssa = np.zeros(N_rxn, dtype=np.int8)
        for ri in range(N_rxn):
            for rr in range(self._nb_n_reactants[ri]):
                if int(self._nb_r_idx[ri, rr]) in qssa_set or int(self._nb_r_target[ri, rr]) in qssa_set:
                    rxn_is_qssa[ri] = 1
            for pp in range(self._nb_n_products[ri]):
                if int(self._nb_p_idx[ri, pp]) in qssa_set or int(self._nb_p_target[ri, pp]) in qssa_set:
                    rxn_is_qssa[ri] = 1
        self._nb_rxn_is_qssa = rxn_is_qssa
        n_tagged = int(rxn_is_qssa.sum())
        n_rev = int((self._nb_rxn_type[rxn_is_qssa > 0] == 1).sum())
        n_irr = int((self._nb_rxn_type[rxn_is_qssa > 0] == 0).sum())
        print(f"  QSSA net-rate: {n_tagged} reactions tagged ({n_rev} rev, {n_irr} irr)")

        # Re-warm Numba with updated rxn_is_qssa
        y_test = np.full(self.n_species, 1e-10)
        y_test[self._nb_H_idx] = 1e-3
        _compute_rates_kernel(
            y_test,
            self._nb_rxn_type, self._nb_rxn_k, self._nb_rxn_kf, self._nb_rxn_kb,
            self._nb_r_idx, self._nb_r_coeff, self._nb_r_target, self._nb_n_reactants,
            self._nb_p_idx, self._nb_p_coeff, self._nb_p_target, self._nb_n_products,
            self._nb_n_pairs, self._nb_pair_total_idx, self._nb_pair_Ka,
            self._nb_H_idx, self._nb_OH_idx, self._nb_Kw,
            self._nb_N_s, self._nb_trace, self._nb_max_rate, self._nb_max_conc,
            self._nb_qssa_enabled, self._nb_qssa_k, self._nb_qssa_idx,
            self._nb_qssa_total_idx, self._nb_qssa_Ka,
            self._nb_rxn_is_qssa,
        )

    def set_qssa_enabled(self, enabled: bool):
        """Enable/disable QSSA for monolithic vs split solver."""
        self._nb_qssa_enabled = np.int64(1 if enabled else 0)

    def apply_qssa(self, y_cell: np.ndarray) -> None:
        """Compute QSSA concentrations for 4 species, overwrite in-place.

        QSSA species: x₁=HOCl⁻, x₂=Cl₂⁻, x₃=Cl, x₄=HOClH.

        Strategy (4→2→2 reduction):
          1. Express x₃ = f(x₁,x₂) and x₄ = g(x₁,x₂)
          2. Substitute into HOCl⁻/Cl₂⁻ equations → modified 2×2 system
          3. Solve 2×2 for x₁,x₂, then back-substitute for x₃,x₄
        """
        if not hasattr(self, '_qssa'):
            return

        q = self._qssa
        tr = self.trace

        # --- Non-QSSA species (read from y_cell) ---
        OH    = max(y_cell[q['idx_OH']],    tr)
        Cl_m  = max(y_cell[q['idx_Cl-']],   tr)
        Hp    = max(y_cell[q['idx_H+']],    1e-14)
        Cl2   = max(y_cell[q['idx_Cl2']],   tr)
        Cl3_m = max(y_cell[q['idx_Cl3-']],  tr)
        H_at  = max(y_cell[q['idx_H']],     tr)
        O3    = max(y_cell[q['idx_O3']],     tr)
        ClO2  = max(y_cell[q['idx_ClO2']],  tr) if q['idx_ClO2'] >= 0 else tr
        NO3   = max(y_cell[q['idx_NO3']],   tr) if q['idx_NO3'] >= 0 else tr
        # Old values for linearization of self-reactions
        Cl_old    = max(y_cell[q['idx_Cl']],    tr)
        Cl2_m_old = max(y_cell[q['idx_Cl2-']],  tr)

        # --- Algebraic / speciated species ---
        OH_m = max(self.Kw / Hp, tr)

        idx = q['idx_HO2_total']
        HO2_tot = max(y_cell[idx], 0.0) if idx >= 0 else 0.0
        Ka = q['Ka_HO2']
        denom_ho2 = Hp + Ka
        HO2  = max(HO2_tot * Hp / denom_ho2, tr)
        O2_m = max(HO2_tot * Ka / denom_ho2, tr)

        idx = q['idx_HONO_total']
        HONO_tot = max(y_cell[idx], 0.0) if idx >= 0 else 0.0
        Ka = q['Ka_HONO']
        NO2_m = max(HONO_tot * Ka / (Hp + Ka), tr)

        idx = q['idx_H2O2_total']
        H2O2_tot = max(y_cell[idx], 0.0) if idx >= 0 else 0.0
        Ka = q['Ka_H2O2']
        H2O2 = max(H2O2_tot * Hp / (Hp + Ka), tr)

        idx = q['idx_HClO_total']
        HClO_tot = max(y_cell[idx], 0.0) if idx >= 0 else 0.0
        Ka_hclo = q['Ka_HClO']
        HClO  = max(HClO_tot * Hp / (Hp + Ka_hclo), tr)
        ClO_m = max(HClO_tot * Ka_hclo / (Hp + Ka_hclo), tr)

        # --- Rate constants (K_CAP-scaled) ---
        k3f  = q['S3_kf'];  k3b  = q['S3_kb']
        k4f  = q['S4_kf'];  k4b  = q['S4_kb']
        k5f  = q['S5_kf'];  k5b  = q['S5_kb']
        k6f  = q['S6_kf'];  k6b  = q['S6_kb']
        k7f  = q['S7_kf'];  k7b  = q['S7_kb']
        k8f  = q['S8_kf'];  k8b  = q['S8_kb']
        k9f  = q['S9_kf'];  k9b  = q['S9_kb']
        k24  = q['S24_k']
        k28  = q['S28_k'];  k29  = q['S29_k']
        k32  = q['S32_k'];  k33  = q['S33_k'];  k34  = q['S34_k']
        k35  = q['S35_k'];  k36  = q['S36_k'];  k37  = q['S37_k']
        k38  = q['S38_k'];  k39  = q['S39_k'];  k40  = q['S40_k']
        k41  = q['S41_k'];  k42  = q['S42_k']
        k52  = q['S52_k'];  k54  = q['S54_k'];  k69  = q['S69_k']
        k23  = q['S23_k'];  k25  = q['S25_k'];  k26  = q['S26_k']
        k27  = q['S27_k'];  k43  = q['S43_k'];  k48  = q['S48_k']
        k50  = q['S50_k'];  k53  = q['S53_k'];  k55  = q['S55_k']
        k61  = q['S61_k']

        # =====================================================================
        # Picard iteration for nonlinear QSSA self-terms
        # =====================================================================
        for _qssa_iter in range(20):
            # Save old values for convergence check
            prev_x1 = y_cell[q['idx_HOCl-']]
            prev_x2 = y_cell[q['idx_Cl2-']]
            prev_x3 = y_cell[q['idx_Cl']]
            prev_x4 = y_cell[q['idx_HOClH']]
            # Update linearization points from current estimates
            Cl_old    = max(y_cell[q['idx_Cl']],   tr)
            Cl2_m_old = max(y_cell[q['idx_Cl2-']], tr)

            # Cl (x₃) equation
            P3 = k48 * HClO * HO2 + k55 * NO3 * Cl_m + k61 * ClO2
            a31 = k5b * Hp
            a32 = k6b + k41 * O3
            alpha3 = (k5f + k6f * Cl_m + 2.0 * k23 * Cl_old
                      + k24 * OH_m + k25 * HO2 + k26 * H2O2
                      + k27 * H_at + k33 * Cl2_m_old + k43 * ClO_m
                      + k50 * HClO + k53 * NO2_m)
            alpha3 = max(alpha3, 1e-30)

            # HOClH (x₄) equation
            P4 = 0.0
            a41 = k7b * Hp
            a42 = k8f
            alpha4 = k7f + k8b * Cl_m
            alpha4 = max(alpha4, 1e-30)

            # HOCl⁻ (x₁) equation
            P1_ext = k4f * OH * Cl_m
            a12 = (k9f + k3b) * OH_m
            a13 = k5f + k24 * OH_m
            a14 = k7f
            alpha1 = k4b + k5b * Hp + k7b * Hp + (k3f + k9b) * Cl_m

            P1_mod = P1_ext + a13 * P3 / alpha3 + a14 * P4 / alpha4
            alpha1_mod = alpha1 - a13 * a31 / alpha3 - a14 * a41 / alpha4
            a12_mod = a12 + a13 * a32 / alpha3 + a14 * a42 / alpha4

            # Cl₂⁻ (x₂) equation
            P2_ext = (k28 * Cl2 * HO2 + k29 * Cl2 * O2_m
                      + k52 * Cl3_m * H_at)
            a21 = (k3f + k9b) * Cl_m
            a23 = k6f * Cl_m
            a24 = k8b * Cl_m
            alpha2 = (k6b + k8f + (k9f + k3b) * OH_m
                      + 2.0 * (k32 + k34) * Cl2_m_old
                      + k33 * Cl_old + (k35 + k36) * H2O2
                      + k37 * HO2 + k38 * O2_m + k39 * OH
                      + k40 * OH_m + k41 * O3 + k42 * H_at
                      + k54 * NO2_m + k69 * ClO2)

            P2_mod = P2_ext + a23 * P3 / alpha3 + a24 * P4 / alpha4
            a21_mod = a21 + a23 * a31 / alpha3 + a24 * a41 / alpha4
            alpha2_mod = alpha2 - a23 * a32 / alpha3 - a24 * a42 / alpha4

            # Solve modified 2×2 system
            det = alpha1_mod * alpha2_mod - a12_mod * a21_mod
            if abs(det) < 1e-30:
                return  # degenerate

            x1 = (P1_mod * alpha2_mod + a12_mod * P2_mod) / det
            x2 = (alpha1_mod * P2_mod + a21_mod * P1_mod) / det
            x3 = (P3 + a31 * x1 + a32 * x2) / alpha3
            x4 = (P4 + a41 * x1 + a42 * x2) / alpha4

            y_cell[q['idx_HOCl-']] = max(x1, tr)
            y_cell[q['idx_Cl2-']]  = max(x2, tr)
            y_cell[q['idx_Cl']]    = max(x3, tr)
            y_cell[q['idx_HOClH']] = max(x4, tr)

            # Convergence check: max relative change < 1e-12
            max_rel = 0.0
            for v_new, v_old in [(x1, prev_x1), (x2, prev_x2),
                                 (x3, prev_x3), (x4, prev_x4)]:
                if v_old > tr:
                    r = abs(v_new - v_old) / v_old
                    if r > max_rel:
                        max_rel = r
            if max_rel < 1e-12:
                break

    # =================================================================
    # Numba-accelerated methods
    # =================================================================

    def _precompute_numba_arrays(self):
        """
        Convert reaction data from list-of-dicts to flat numpy arrays
        for Numba JIT kernel.

        Speciated species encoding:
            Each acid-base pair p has acid (index 2*p) and base (2*p+1)
            in a speciated_conc array. OH- is at index 2*n_pairs.
            In reaction index arrays, speciated species k is encoded as
            idx = -(k+1).
        """
        pair_list = list(self.pKa_map.items())
        n_pairs = len(pair_list)
        speciated_name_to_idx = {}  # name → index in speciated_conc

        pair_total_idx = np.zeros(n_pairs, dtype=np.int64)
        pair_Ka = np.zeros(n_pairs, dtype=np.float64)

        for p, (total_name, (acid_name, base_name, pKa)) in enumerate(pair_list):
            Ka = 10.0 ** (-pKa)
            pair_Ka[p] = Ka
            pair_total_idx[p] = self.species_idx[total_name]
            speciated_name_to_idx[acid_name] = 2 * p       # acid
            speciated_name_to_idx[base_name] = 2 * p + 1   # base

        speciated_name_to_idx['OH-'] = 2 * n_pairs

        # Determine max reactants and products
        MAX_R = 3
        MAX_P = 3
        N_rxn = len(self._rxn_data)

        rxn_type = np.zeros(N_rxn, dtype=np.int64)
        rxn_k = np.zeros(N_rxn, dtype=np.float64)
        rxn_kf = np.zeros(N_rxn, dtype=np.float64)
        rxn_kb = np.zeros(N_rxn, dtype=np.float64)

        r_idx = np.full((N_rxn, MAX_R), -99, dtype=np.int64)
        r_coeff = np.zeros((N_rxn, MAX_R), dtype=np.float64)
        r_target = np.full((N_rxn, MAX_R), -1, dtype=np.int64)
        n_reactants = np.zeros(N_rxn, dtype=np.int64)

        p_idx = np.full((N_rxn, MAX_P), -99, dtype=np.int64)
        p_coeff = np.zeros((N_rxn, MAX_P), dtype=np.float64)
        p_target = np.full((N_rxn, MAX_P), -1, dtype=np.int64)
        n_products = np.zeros(N_rxn, dtype=np.int64)

        for ri, rxn in enumerate(self._rxn_data):
            if rxn['type'] == 'irr':
                rxn_type[ri] = 0
                rxn_k[ri] = rxn['k']
            else:
                rxn_type[ri] = 1
                rxn_kf[ri] = rxn['k_f']
                rxn_kb[ri] = rxn['k_b']

            for rr, (sp_name, coeff, idx) in enumerate(rxn['reactants']):
                r_coeff[ri, rr] = coeff
                if idx >= 0:
                    r_idx[ri, rr] = idx
                    r_target[ri, rr] = idx
                elif idx == -1:
                    # Speciated → encode as -(spec_idx + 1)
                    spec_idx = speciated_name_to_idx[sp_name]
                    r_idx[ri, rr] = -(spec_idx + 1)
                    # Target in dydt → total variable
                    total_name = self.species_to_total[sp_name]
                    r_target[ri, rr] = self.species_idx.get(total_name, -1)
                else:
                    r_idx[ri, rr] = -99  # skip
                    r_target[ri, rr] = -1
            n_reactants[ri] = len(rxn['reactants'])

            for pp, (sp_name, coeff, idx) in enumerate(rxn['products']):
                p_coeff[ri, pp] = coeff
                if idx >= 0:
                    p_idx[ri, pp] = idx
                    p_target[ri, pp] = idx
                elif idx == -1:
                    spec_idx = speciated_name_to_idx[sp_name]
                    p_idx[ri, pp] = -(spec_idx + 1)
                    total_name = self.species_to_total[sp_name]
                    p_target[ri, pp] = self.species_idx.get(total_name, -1)
                else:
                    p_idx[ri, pp] = -99
                    p_target[ri, pp] = -1
            n_products[ri] = len(rxn.get('products', []))

        # Store as instance attributes
        self._nb_rxn_type = rxn_type
        self._nb_rxn_k = rxn_k
        self._nb_rxn_kf = rxn_kf
        self._nb_rxn_kb = rxn_kb
        self._nb_r_idx = r_idx
        self._nb_r_coeff = r_coeff
        self._nb_r_target = r_target
        self._nb_n_reactants = n_reactants
        self._nb_p_idx = p_idx
        self._nb_p_coeff = p_coeff
        self._nb_p_target = p_target
        self._nb_n_products = n_products
        self._nb_n_pairs = n_pairs
        self._nb_pair_total_idx = pair_total_idx
        self._nb_pair_Ka = pair_Ka
        self._nb_H_idx = self.species_idx['H+']
        self._nb_OH_idx = self.species_idx.get('OH-', -1)
        self._nb_Kw = self.Kw
        self._nb_N_s = self.n_species
        self._nb_trace = self.trace
        self._nb_max_rate = self.max_rate
        self._nb_max_conc = ODE_CONFIG.max_concentration

        # QSSA defaults (off for non-saline; overridden by _precompute_qssa_params)
        if not hasattr(self, '_nb_qssa_enabled'):
            self._nb_qssa_enabled = np.int64(0)
            self._nb_qssa_k = np.zeros(41, dtype=np.float64)
            self._nb_qssa_idx = np.full(13, -1, dtype=np.int64)
            self._nb_qssa_total_idx = np.full(4, -1, dtype=np.int64)
            self._nb_qssa_Ka = np.zeros(4, dtype=np.float64)

        # rxn_is_qssa: default all zeros, updated by _precompute_qssa_params
        self._nb_rxn_is_qssa = np.zeros(N_rxn, dtype=np.int8)

        # Warm up Numba JIT (first call compiles)
        y_test = np.full(self.n_species, 1e-10)
        y_test[self._nb_H_idx] = 1e-3
        _compute_rates_kernel(
            y_test,
            self._nb_rxn_type, self._nb_rxn_k, self._nb_rxn_kf, self._nb_rxn_kb,
            self._nb_r_idx, self._nb_r_coeff, self._nb_r_target, self._nb_n_reactants,
            self._nb_p_idx, self._nb_p_coeff, self._nb_p_target, self._nb_n_products,
            self._nb_n_pairs, self._nb_pair_total_idx, self._nb_pair_Ka,
            self._nb_H_idx, self._nb_OH_idx, self._nb_Kw,
            self._nb_N_s, self._nb_trace, self._nb_max_rate, self._nb_max_conc,
            self._nb_qssa_enabled, self._nb_qssa_k, self._nb_qssa_idx,
            self._nb_qssa_total_idx, self._nb_qssa_Ka,
            self._nb_rxn_is_qssa,
        )

    def compute_rates_numba(self, y_cell: np.ndarray) -> np.ndarray:
        """Numba-accelerated compute_rates. Drop-in replacement."""
        return _compute_rates_kernel(
            y_cell,
            self._nb_rxn_type, self._nb_rxn_k, self._nb_rxn_kf, self._nb_rxn_kb,
            self._nb_r_idx, self._nb_r_coeff, self._nb_r_target, self._nb_n_reactants,
            self._nb_p_idx, self._nb_p_coeff, self._nb_p_target, self._nb_n_products,
            self._nb_n_pairs, self._nb_pair_total_idx, self._nb_pair_Ka,
            self._nb_H_idx, self._nb_OH_idx, self._nb_Kw,
            self._nb_N_s, self._nb_trace, self._nb_max_rate, self._nb_max_conc,
            self._nb_qssa_enabled, self._nb_qssa_k, self._nb_qssa_idx,
            self._nb_qssa_total_idx, self._nb_qssa_Ka,
            self._nb_rxn_is_qssa,
        )

    def compute_rates_batch(self, y_2d: np.ndarray) -> np.ndarray:
        """
        Compute rates for all N_z cells at once.

        Parameters
        ----------
        y_2d : ndarray, shape (N_z, N_s)
            Concentration array for all cells.

        Returns
        -------
        ndarray, shape (N_z, N_s)
            Rate of change for all cells.
        """
        N_z = y_2d.shape[0]
        result = np.zeros_like(y_2d)
        for j in range(N_z):
            result[j] = self.compute_rates_numba(y_2d[j])
        return result
