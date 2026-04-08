"""Parser for BOLSIG+ output files (LXCat format, ver. 03/2016).

Reads transport coefficients (A-series) and rate coefficients (C-series)
from BOLSIG+ swarm parameter output files.  Returns structured data
suitable for building ε̄-indexed look-up tables.

Standalone module — no imports from other plasma0d_v2 packages.

Usage::

    from plasma0d_v2.bolsig_parser import parse_bolsig_file, load_bolsig_files

    data = parse_bolsig_file("input/Condition1_300K.txt")
    print(data.summary())

    # Load multiple conditions keyed by Tgas
    all_data = load_bolsig_files(["input/Condition1_300K.txt",
                                   "input/Condition1_523K.txt"])
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RateLabel:
    """Metadata for one BOLSIG+ rate coefficient entry."""
    id: str                              # e.g. "C1", "C68"
    species: str                         # e.g. "CH4", "N2", "O2", "CO2"
    process: str                         # e.g. "Attachment", "Excitation", "Ionization", ...
    threshold_eV: Optional[float] = None # excitation/ionization threshold if given

    def __repr__(self) -> str:
        thr = f"  {self.threshold_eV} eV" if self.threshold_eV is not None else ""
        return f"{self.id:>4s}  {self.species:<5s} {self.process}{thr}"


@dataclass
class BolsigData:
    """Parsed BOLSIG+ output for one gas-temperature condition."""

    filepath: str
    tgas_K: float                 # Gas temperature [K]
    eedf_type: str                # "non-Maxwellian" or "Maxwellian"

    # --- Transport coefficients (1-D, indexed by E/N point) ---------------
    EN_Td: np.ndarray             = field(default_factory=lambda: np.array([]))
    mean_energy_eV: np.ndarray    = field(default_factory=lambda: np.array([]))  # A1
    mobility_N: np.ndarray        = field(default_factory=lambda: np.array([]))  # A2
    diffusion_N: np.ndarray       = field(default_factory=lambda: np.array([]))  # A6
    energy_mobility_N: np.ndarray = field(default_factory=lambda: np.array([]))  # A11
    energy_diffusion_N: np.ndarray= field(default_factory=lambda: np.array([]))  # A12
    total_collision_freq_N: np.ndarray = field(default_factory=lambda: np.array([]))  # A13
    momentum_freq_N: np.ndarray   = field(default_factory=lambda: np.array([]))  # A14
    ionization_freq_N: np.ndarray = field(default_factory=lambda: np.array([]))  # A16
    attachment_freq_N: np.ndarray = field(default_factory=lambda: np.array([]))  # A17
    townsend_ioniz_N: np.ndarray  = field(default_factory=lambda: np.array([]))  # A18
    townsend_attach_N: np.ndarray = field(default_factory=lambda: np.array([]))  # A19
    power_N: np.ndarray           = field(default_factory=lambda: np.array([]))  # A20 [eV m3/s]
    elastic_power_N: np.ndarray   = field(default_factory=lambda: np.array([]))  # A21 [eV m3/s]
    inelastic_power_N: np.ndarray = field(default_factory=lambda: np.array([]))  # A22 [eV m3/s]
    growth_power_N: np.ndarray    = field(default_factory=lambda: np.array([]))  # A23 [eV m3/s]

    # --- Rate coefficients ------------------------------------------------
    rate_labels: List[RateLabel]  = field(default_factory=list)
    rate_coefficients: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_EN, n_rates) [m3/s]

    # --- Convenience properties -------------------------------------------

    @property
    def n_EN(self) -> int:
        """Number of E/N grid points."""
        return len(self.EN_Td)

    @property
    def n_rates(self) -> int:
        """Number of rate coefficient processes."""
        return len(self.rate_labels)

    def get_rate_by_id(self, rate_id: str) -> np.ndarray:
        """Return rate coefficient array for a given label (e.g. ``'C10'``)."""
        for i, rl in enumerate(self.rate_labels):
            if rl.id == rate_id:
                return self.rate_coefficients[:, i]
        raise KeyError(f"Rate coefficient '{rate_id}' not found in {self.filepath}")

    def summary(self) -> str:
        """One-line-per-field summary string."""
        lines = [
            f"BOLSIG+ Data: {self.filepath}",
            f"  Tgas = {self.tgas_K:.0f} K, EEDF = {self.eedf_type}",
            f"  E/N range: [{self.EN_Td[0]:.4f}, {self.EN_Td[-1]:.1f}] Td "
            f"({self.n_EN} points)",
            f"  Mean energy range: [{self.mean_energy_eV[0]:.4f}, "
            f"{self.mean_energy_eV[-1]:.2f}] eV",
            f"  Rate coefficients: {self.n_rates} processes",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RE_TGAS = re.compile(r"Tgas\s*=\s*([\d.]+)\s*K", re.IGNORECASE)
_RE_EEDF = re.compile(r"EEDF\s*=\s*(.+)", re.IGNORECASE)

# Matches the transport-coefficients data header:
#   R#    E/N (Td)      A1   ...
_RE_TRANSPORT_HDR = re.compile(r"^\s*R#\s+E/N\s*\(Td\)\s+A1\b")

# Matches the rate-coefficients data header:
#   R#    E/N (Td) Energy (eV)      C1   ...
_RE_RATE_HDR = re.compile(r"^\s*R#\s+E/N\s*\(Td\)\s+Energy\s*\(eV\)\s+C1\b")

# Matches a rate-label description line:  " C10   CH4    Ionization    12.60 eV"
_RE_RATE_LABEL = re.compile(
    r"^\s+(C\d+)\s+(\S+)\s+(.+?)\s*$"
)

# Threshold extractor from process string:  "Ionization    12.60 eV"
_RE_THRESHOLD = re.compile(r"([\d.]+)\s*eV\s*$")


def _find_line(lines: List[str], pattern: re.Pattern, start: int = 0) -> int:
    """Return index of first line matching *pattern* at or after *start*."""
    for i in range(start, len(lines)):
        if pattern.search(lines[i]):
            return i
    return -1


def _parse_data_block(lines: List[str], start: int, n_cols_expected: int = 0):
    """Read consecutive numeric rows starting at *start*.

    Returns a 2-D numpy array (n_rows × n_cols).  Stops at the first
    blank or non-numeric line.
    """
    rows: List[List[float]] = []
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            break
        # Each row starts with an integer row number.  Fast check:
        tokens = stripped.split()
        try:
            # First token must be an integer row number
            int(tokens[0])
            row = [float(t) for t in tokens]
            rows.append(row)
        except (ValueError, IndexError):
            break
    if not rows:
        raise ValueError(f"No numeric data found starting at line {start + 1}")
    arr = np.array(rows, dtype=np.float64)
    if n_cols_expected and arr.shape[1] != n_cols_expected:
        raise ValueError(
            f"Expected {n_cols_expected} columns, got {arr.shape[1]} "
            f"(line {start + 1})"
        )
    return arr


def _parse_rate_labels(lines: List[str], start: int, end: int) -> List[RateLabel]:
    """Parse rate-coefficient label lines between *start* and *end*."""
    labels: List[RateLabel] = []
    for i in range(start, end):
        m = _RE_RATE_LABEL.match(lines[i])
        if not m:
            continue
        cid = m.group(1).strip()
        species = m.group(2).strip()
        process_raw = m.group(3).strip()

        # Extract threshold if present
        threshold: Optional[float] = None
        m_thr = _RE_THRESHOLD.search(process_raw)
        if m_thr:
            threshold = float(m_thr.group(1))
            # Clean process name: remove the threshold suffix
            process_name = process_raw[: m_thr.start()].strip()
        else:
            process_name = process_raw

        labels.append(RateLabel(id=cid, species=species,
                                process=process_name, threshold_eV=threshold))
    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_bolsig_file(filepath: str) -> BolsigData:
    """Parse a single BOLSIG+ output file.

    Parameters
    ----------
    filepath : str
        Path to the BOLSIG+ ``.txt`` output file.

    Returns
    -------
    BolsigData
        Structured object with transport and rate-coefficient data.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file format is not recognised.
    """
    with open(filepath, "r") as fh:
        text = fh.read()
    lines = text.splitlines()

    # --- Config section ---------------------------------------------------
    m_tgas = _RE_TGAS.search(text)
    if not m_tgas:
        raise ValueError(f"Could not find 'Tgas = ... K' in {filepath}")
    tgas_K = float(m_tgas.group(1))

    m_eedf = _RE_EEDF.search(text)
    eedf_type = m_eedf.group(1).strip() if m_eedf else "unknown"

    # --- Section 1: Transport coefficients --------------------------------
    hdr_transport = _find_line(lines, _RE_TRANSPORT_HDR)
    if hdr_transport < 0:
        raise ValueError(
            f"Transport-coefficient header (R# E/N ... A1 ...) not found in {filepath}"
        )
    transport = _parse_data_block(lines, hdr_transport + 1, n_cols_expected=20)
    # Columns: 0=R#, 1=E/N, 2=A1, 3=A2, 4=A6, 5=A11, 6=A12,
    #          7=A13, 8=A14, 9=A16, 10=A17, 11=A18, 12=A19,
    #          13=A20, 14=A21, 15=A22, 16=A23, 17=A27, 18=A28, 19=A29

    EN_Td = transport[:, 1]

    # --- Section 2: Rate coefficients -------------------------------------
    # Find the "Rate coefficients (m3/s)" marker
    rate_section_start = -1
    for i in range(hdr_transport + 1, len(lines)):
        if "Rate coefficients" in lines[i] and "Inverse" not in lines[i]:
            rate_section_start = i
            break
    if rate_section_start < 0:
        raise ValueError(
            f"'Rate coefficients (m3/s)' section not found in {filepath}"
        )

    hdr_rate = _find_line(lines, _RE_RATE_HDR, rate_section_start)
    if hdr_rate < 0:
        raise ValueError(
            f"Rate-coefficient header (R# E/N Energy C1 ...) not found in {filepath}"
        )

    # Parse label lines between section marker and data header
    rate_labels = _parse_rate_labels(lines, rate_section_start + 1, hdr_rate)
    n_rates = len(rate_labels)

    # Expected columns: R# + E/N + Energy + n_rates
    rate_data = _parse_data_block(lines, hdr_rate + 1,
                                  n_cols_expected=3 + n_rates)
    # Columns: 0=R#, 1=E/N, 2=Energy(eV)=A1, 3..3+n_rates-1 = C1..C_n

    rate_coeffs = rate_data[:, 3:]  # (n_EN, n_rates)

    # --- Sanity checks ----------------------------------------------------
    if len(EN_Td) != rate_data.shape[0]:
        raise ValueError(
            f"Transport rows ({len(EN_Td)}) != Rate rows ({rate_data.shape[0]})"
        )
    # Mean energy in rate section should match A1 from transport
    energy_rate = rate_data[:, 2]
    energy_transport = transport[:, 2]
    max_diff = np.max(np.abs(energy_rate - energy_transport))
    if max_diff > 1e-6:
        raise ValueError(
            f"Mean-energy mismatch between transport (A1) and rate sections: "
            f"max |diff| = {max_diff:.2e}"
        )

    # --- Assemble ---------------------------------------------------------
    data = BolsigData(
        filepath=filepath,
        tgas_K=tgas_K,
        eedf_type=eedf_type,
        EN_Td=EN_Td,
        mean_energy_eV=transport[:, 2],       # A1
        mobility_N=transport[:, 3],            # A2
        diffusion_N=transport[:, 4],           # A6
        energy_mobility_N=transport[:, 5],     # A11
        energy_diffusion_N=transport[:, 6],    # A12
        total_collision_freq_N=transport[:, 7],# A13
        momentum_freq_N=transport[:, 8],       # A14
        ionization_freq_N=transport[:, 9],     # A16
        attachment_freq_N=transport[:, 10],    # A17
        townsend_ioniz_N=transport[:, 11],     # A18
        townsend_attach_N=transport[:, 12],    # A19
        power_N=transport[:, 13],              # A20
        elastic_power_N=transport[:, 14],      # A21
        inelastic_power_N=transport[:, 15],    # A22
        growth_power_N=transport[:, 16],       # A23
        rate_labels=rate_labels,
        rate_coefficients=rate_coeffs,
    )
    return data


def load_bolsig_files(filepaths: List[str]) -> Dict[float, BolsigData]:
    """Load multiple BOLSIG+ files, returning a dict keyed by ``Tgas`` [K].

    Parameters
    ----------
    filepaths : list of str
        Paths to BOLSIG+ output files.

    Returns
    -------
    dict
        ``{tgas_K: BolsigData, ...}``
    """
    result: Dict[float, BolsigData] = {}
    for fp in filepaths:
        data = parse_bolsig_file(fp)
        if data.tgas_K in result:
            print(f"  WARNING: duplicate Tgas={data.tgas_K} K — "
                  f"overwriting with {fp}")
        result[data.tgas_K] = data
    return result


# ---------------------------------------------------------------------------
# EEDF parser
# ---------------------------------------------------------------------------

@dataclass
class EEDFBlock:
    """EEDF data for one E/N condition."""
    EN_Td: float
    tgas_K: float
    energy_eV: np.ndarray   # (n_points,) [eV]
    eedf: np.ndarray        # (n_points,) [eV^(-3/2)], normalization: ∫F₀√ε dε = 1


@dataclass
class EEDFData:
    """All EEDF data from one BOLSIG+ EEDF output file."""
    filepath: str
    tgas_K: float
    n_blocks: int
    EN_Td: np.ndarray            # (n_blocks,) [Td]
    blocks: List[EEDFBlock] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"EEDF Data: {self.filepath}\n"
            f"  Tgas = {self.tgas_K:.0f} K, {self.n_blocks} E/N points\n"
            f"  E/N range: [{self.EN_Td[0]:.4f}, {self.EN_Td[-1]:.1f}] Td\n"
            f"  Grid points per block: {len(self.blocks[0].energy_eV)}"
        )


_RE_EEDF_EN = re.compile(r"Electric field / N \(Td\)\s+([\d.eE+\-]+)")
_RE_EEDF_TGAS = re.compile(r"Gas temperature \(K\)\s+([\d.eE+\-]+)")
_RE_EEDF_DATA_HDR = re.compile(r"^\s*Energy \(eV\)\s+EEDF")
_RE_EEDF_BLOCK = re.compile(r"^R(\d+)\s*$")


def parse_eedf_file(filepath: str) -> EEDFData:
    """Parse a BOLSIG+ EEDF output file.

    Each block (R1..RN) contains a header with E/N and gas conditions,
    followed by three-column data: Energy [eV], EEDF [eV^(-3/2)], Anisotropy.

    EEDF normalization (BOLSIG+ convention): ∫ F₀(ε) · √ε dε = 1
    """
    with open(filepath, "r") as fh:
        lines = fh.read().splitlines()

    blocks: List[EEDFBlock] = []
    tgas_K = 0.0
    i = 0

    while i < len(lines):
        m_block = _RE_EEDF_BLOCK.match(lines[i].strip())
        if not m_block:
            i += 1
            continue

        en_td = 0.0
        block_tgas = 0.0
        j = i + 1
        data_start = -1

        while j < len(lines):
            line = lines[j]
            m_en = _RE_EEDF_EN.search(line)
            if m_en:
                en_td = float(m_en.group(1))
            m_tg = _RE_EEDF_TGAS.search(line)
            if m_tg:
                block_tgas = float(m_tg.group(1))
            if _RE_EEDF_DATA_HDR.match(line):
                data_start = j + 1
                break
            j += 1

        if data_start < 0:
            i = j + 1
            continue

        if tgas_K == 0.0:
            tgas_K = block_tgas

        rows = []
        k = data_start
        while k < len(lines):
            stripped = lines[k].strip()
            if not stripped:
                k += 1
                continue
            try:
                tokens = stripped.split()
                float(tokens[0])
                rows.append([float(t) for t in tokens[:3]])
                k += 1
            except (ValueError, IndexError):
                break

        if rows:
            arr = np.array(rows, dtype=np.float64)
            blocks.append(EEDFBlock(
                EN_Td=en_td,
                tgas_K=block_tgas,
                energy_eV=arr[:, 0],
                eedf=arr[:, 1],
            ))

        i = k

    if not blocks:
        raise ValueError(f"No EEDF blocks found in {filepath}")

    en_arr = np.array([b.EN_Td for b in blocks])

    return EEDFData(
        filepath=filepath,
        tgas_K=tgas_K,
        n_blocks=len(blocks),
        EN_Td=en_arr,
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "input/Condition1_300K.txt",
        "input/Condition1_523K.txt",
    ]

    for f in files:
        try:
            data = parse_bolsig_file(f)
        except FileNotFoundError:
            print(f"File not found: {f}")
            continue
        except ValueError as e:
            print(f"Parse error in {f}: {e}")
            continue

        print(data.summary())
        print(f"  A20 (Power/N) at 100 Td: "
              f"{np.interp(100.0, data.EN_Td, data.power_N):.4e} eV m3/s")
        print(f"  A21 (Elastic/N) at 100 Td: "
              f"{np.interp(100.0, data.EN_Td, data.elastic_power_N):.4e} eV m3/s")
        print(f"  A22 (Inelastic/N) at 100 Td: "
              f"{np.interp(100.0, data.EN_Td, data.inelastic_power_N):.4e} eV m3/s")

        # Show a few rate labels
        print(f"  First 5 rate labels:")
        for rl in data.rate_labels[:5]:
            print(f"    {rl}")
        print(f"  Last 3 rate labels:")
        for rl in data.rate_labels[-3:]:
            print(f"    {rl}")
        print()
