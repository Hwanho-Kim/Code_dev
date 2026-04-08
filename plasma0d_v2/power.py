"""Power source module — V-I curve framework for sDBD reactor.

V-I curve (any CSV)
  -> V(t), I(t) interpolation
  -> P_dep(t) = |V(t) * I(t)| / V_eff  [W/m³]

E/N is no longer computed here — ε̄ is a state variable solved by the ODE.
"""

import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .constants import KB, QE, TD_TO_VM2, total_number_density, R_GAS


# ============================================================
#  Pulse Auto-Detection
# ============================================================

@dataclass
class PulseInfo:
    n_pulses: int = 0
    frequency_Hz: float = 0.0
    period_s: float = 0.0
    pulse_width_s: float = 0.0
    duty_cycle: float = 0.0
    V_peak: float = 0.0
    I_peak: float = 0.0
    P_peak: float = 0.0
    E_per_pulse_J: float = 0.0
    waveform_type: str = "unknown"
    recommended_max_step_s: float = 1e-6
    recommended_t_end_s: float = 1e-3


# ============================================================
#  V-I Curve Loader (format-agnostic)
# ============================================================

class VICurve:
    """Voltage-current waveform loader.
    
    Accepts any CSV with at least 3 numeric columns (time, voltage, current).
    Auto-detects header rows, units, and pulse characteristics.
    """
    
    def __init__(self):
        self.time: np.ndarray = np.array([])
        self.voltage: np.ndarray = np.array([])
        self.current: np.ndarray = np.array([])
        self.dt: float = 0.0
        self.duration: float = 0.0
        self.pulse_info: Optional[PulseInfo] = None
        self._loaded = False
    
    def load(self, filepath: str,
             col_time: int = 0, col_voltage: int = 1, col_current: int = 2,
             skip_rows: Optional[int] = None,
             delimiter: str = ',',
             V_scale: float = 1.0,
             I_scale: float = 1.0,
             V_unit: str = 'V',
             I_unit: str = 'A'):
        if skip_rows is None:
            skip_rows = self._detect_header_rows(filepath, delimiter)
        
        data = np.genfromtxt(filepath, delimiter=delimiter, skip_header=skip_rows,
                             filling_values=0.0)
        
        if data.ndim == 1:
            raise ValueError(f"Only 1 column detected. Check delimiter='{delimiter}'")
        if data.shape[1] < 3:
            raise ValueError(f"Need >= 3 columns, got {data.shape[1]}")
        
        self.time = data[:, col_time]
        self.voltage = data[:, col_voltage] * V_scale
        self.current = data[:, col_current] * I_scale
        
        V_mult = {'kV': 1000.0, 'mV': 0.001, 'V': 1.0}.get(V_unit, 1.0)
        I_mult = {'mA': 0.001, 'uA': 1e-6, 'A': 1.0}.get(I_unit, 1.0)
        self.voltage *= V_mult
        self.current *= I_mult
        
        valid = np.isfinite(self.time) & np.isfinite(self.voltage) & np.isfinite(self.current)
        self.time = self.time[valid]
        self.voltage = self.voltage[valid]
        self.current = self.current[valid]
        
        self.dt = float(np.median(np.diff(self.time)))
        self.duration = float(self.time[-1] - self.time[0])
        self._loaded = True
        
        self.pulse_info = self._analyze_pulses()
        self._print_summary()
    
    def _detect_header_rows(self, filepath: str, delimiter: str) -> int:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delimiter)
                try:
                    vals = [float(p.strip().strip('"')) for p in parts if p.strip()]
                    if len(vals) >= 3:
                        return i
                except ValueError:
                    continue
        return 0
    
    def _analyze_pulses(self) -> PulseInfo:
        info = PulseInfo()
        V = self.voltage
        I = self.current
        t = self.time
        
        info.V_peak = float(np.max(np.abs(V)))
        info.I_peak = float(np.max(np.abs(I)))
        info.P_peak = float(np.max(np.abs(V * I)))
        
        V_range = np.max(V) - np.min(V)
        if V_range < 10:
            info.waveform_type = "dc"
            info.recommended_max_step_s = self.duration / 100
            info.recommended_t_end_s = self.duration
            return info
        
        V_peak_pos = np.max(V)
        if V_peak_pos > 10:
            V_threshold = 0.5 * V_peak_pos
            above = V > V_threshold
        else:
            V_threshold = 0.5 * np.max(np.abs(V))
            above = np.abs(V) > V_threshold
        transitions = np.diff(above.astype(int))
        rising_edges = np.where(transitions == 1)[0]
        falling_edges = np.where(transitions == -1)[0]
        
        if len(rising_edges) >= 2:
            periods = np.diff(t[rising_edges])
            info.period_s = float(np.median(periods))
            info.frequency_Hz = 1.0 / info.period_s if info.period_s > 0 else 0
            info.n_pulses = len(rising_edges)
            
            if len(falling_edges) > 0 and len(rising_edges) > 0:
                first_rise = rising_edges[0]
                falls_after = falling_edges[falling_edges > first_rise]
                if len(falls_after) > 0:
                    info.pulse_width_s = float(t[falls_after[0]] - t[first_rise])
            
            info.duty_cycle = info.pulse_width_s / info.period_s if info.period_s > 0 else 0
            info.waveform_type = "sinusoidal" if info.duty_cycle > 0.3 else "pulsed"
        else:
            info.n_pulses = 1 if len(rising_edges) == 1 else 0
            info.waveform_type = "single_pulse" if info.n_pulses == 1 else "unknown"
            info.period_s = self.duration
        
        P = V * I
        if info.n_pulses >= 2 and info.period_s > 0:
            one_period_pts = int(info.period_s / self.dt)
            if one_period_pts > 0 and rising_edges[0] + one_period_pts < len(t):
                idx0 = rising_edges[0]
                idx1 = idx0 + one_period_pts
                _trapz = getattr(np, 'trapezoid', None) or np.trapz
                info.E_per_pulse_J = float(_trapz(P[idx0:idx1], t[idx0:idx1]))
        
        if info.waveform_type == "pulsed":
            info.recommended_max_step_s = max(info.pulse_width_s / 20, self.dt * 2)
            info.recommended_t_end_s = max(5 * info.period_s, 10 * info.pulse_width_s)
        elif info.waveform_type == "sinusoidal":
            info.recommended_max_step_s = info.period_s / 100
            info.recommended_t_end_s = 5 * info.period_s
        else:
            info.recommended_max_step_s = self.duration / 1000
            info.recommended_t_end_s = self.duration
        
        return info
    
    def _print_summary(self):
        p = self.pulse_info
        print(f"  V-I curve loaded: {len(self.time)} points")
        print(f"    Duration    : {self.duration*1e6:.1f} us ({self.duration*1e3:.3f} ms)")
        print(f"    Sample rate : {1/self.dt/1e6:.2f} MHz (dt={self.dt*1e9:.1f} ns)")
        print(f"    Voltage     : [{self.voltage.min():.0f}, {self.voltage.max():.0f}] V "
              f"(peak {p.V_peak:.0f} V = {p.V_peak/1000:.2f} kV)")
        print(f"    Current     : [{self.current.min():.4f}, {self.current.max():.4f}] A "
              f"(peak {p.I_peak:.3f} A)")
        print(f"    Peak power  : {p.P_peak:.1f} W ({p.P_peak/1000:.2f} kW)")
        print(f"    Waveform    : {p.waveform_type}")
        if p.n_pulses > 1:
            print(f"    Pulses      : {p.n_pulses} detected")
            print(f"    Frequency   : {p.frequency_Hz:.1f} Hz (T={p.period_s*1e6:.1f} us)")
            print(f"    Pulse width : {p.pulse_width_s*1e6:.2f} us")
            print(f"    Duty cycle  : {p.duty_cycle*100:.2f}%")
            if p.E_per_pulse_J > 0:
                print(f"    Energy/pulse: {p.E_per_pulse_J*1e3:.3f} mJ")
        print(f"    Recommended : max_step={p.recommended_max_step_s*1e9:.0f} ns, "
              f"t_end={p.recommended_t_end_s*1e6:.0f} us")
    
    def interpolate(self, t: float) -> Tuple[float, float]:
        """Get V(t) and I(t) at arbitrary time (periodic extension)."""
        if not self._loaded:
            return 0.0, 0.0
        
        t0 = self.time[0]
        
        if (self.pulse_info and self.pulse_info.n_pulses >= 2
                and self.pulse_info.period_s > 0):
            t_mod = (t - t0) % self.pulse_info.period_s + t0
            V = float(np.interp(t_mod, self.time, self.voltage))
            I = float(np.interp(t_mod, self.time, self.current))
        else:
            if t < self.time[0] or t > self.time[-1]:
                return 0.0, 0.0
            V = float(np.interp(t, self.time, self.voltage))
            I = float(np.interp(t, self.time, self.current))
        
        return V, I


# ============================================================
#  Power Source (V-I -> P_dep via V_eff)
# ============================================================

class PowerSource:
    """Power source for sDBD reactor.
    
    P_dep(t) = |V(t) * I(t)| / V_eff  [W/m³]
    
    V_eff is the effective discharge volume (fitting parameter).
    E/N is no longer computed here — ε̄ is a state variable.
    """
    
    def __init__(self, V_eff: float, P_gas: float):
        self.V_eff = V_eff
        self.P_gas = P_gas
        self.vi_curve = VICurve()
        self._mode = 'vi_curve'       # 'vi_curve', 'constant', 'vi_envelope', or 'pulsed'
        self._P_constant_Wm3 = 0.0
        self._P_avg_W = 0.0
        self._envelope_scale = 1.0
        self._envelope_threshold = 0.0
        # Parametric pulse state
        self._pulse_period = 0.0
        self._pulse_t_on = 0.0
        self._pulse_rise_time = 0.0
        self._pulse_waveform = 'trapezoidal'
        self._pulse_P_on_Wm3 = 0.0
        self._pulse_PRF = 0.0
        self._pulse_duty_cycle = 0.0
    
    def set_constant_power(self, P_watts: float):
        """Set constant power mode. P_dep = P / V_eff [W/m³].

        Args:
            P_watts: total input power [W] (e.g. from Lissajous measurement)
        """
        self._mode = 'constant'
        self._P_constant_Wm3 = P_watts / self.V_eff
        self._P_avg_W = P_watts
        print(f"  Constant power: P = {P_watts:.2f} W")
        print(f"    P_dep = {self._P_constant_Wm3:.3e} W/m³")

    def set_pulsed_power(self, P_peak_W: float, PRF_Hz: float,
                         duty_cycle: float, rise_time_s: float = 0.0,
                         waveform: str = 'trapezoidal'):
        """Parametric pulse mode — no V-I file needed.

        Power waveform is defined by PRF, duty cycle, and peak power.
        During pulse-ON, P_dep = P_peak_W / V_eff [W/m³].
        During pulse-OFF, P_dep = 0.

        Args:
            P_peak_W: peak power during pulse ON [W]
            PRF_Hz: pulse repetition frequency [Hz]
            duty_cycle: fraction of period where power is ON (0 < dc < 1)
            rise_time_s: linear ramp rise/fall time [s]. 0 = hard switch.
                         Trapezoidal: rise_time must be < t_on/2.
            waveform: 'rectangular' (hard switch) or 'trapezoidal' (soft ramp)
        """
        if not 0 < duty_cycle < 1:
            raise ValueError(f"duty_cycle must be in (0, 1), got {duty_cycle}")
        if PRF_Hz <= 0:
            raise ValueError(f"PRF_Hz must be positive, got {PRF_Hz}")

        self._mode = 'pulsed'
        self._pulse_PRF = PRF_Hz
        self._pulse_period = 1.0 / PRF_Hz
        self._pulse_duty_cycle = duty_cycle
        self._pulse_t_on = duty_cycle * self._pulse_period
        self._pulse_waveform = waveform
        self._pulse_P_on_Wm3 = P_peak_W / self.V_eff

        # Clamp rise_time to physical limits
        if waveform == 'trapezoidal' and rise_time_s > 0:
            max_rise = self._pulse_t_on / 2.0
            if rise_time_s > max_rise:
                rise_time_s = max_rise
        elif waveform == 'rectangular':
            rise_time_s = 0.0
        self._pulse_rise_time = rise_time_s

        self._P_avg_W = P_peak_W * duty_cycle
        P_avg_density = self._P_avg_W / self.V_eff

        print(f"  Pulsed power: P_peak = {P_peak_W:.2f} W, "
              f"PRF = {PRF_Hz:.0f} Hz, dc = {duty_cycle*100:.1f}%")
        print(f"    Period    : {self._pulse_period*1e6:.1f} µs "
              f"(t_on = {self._pulse_t_on*1e6:.1f} µs, "
              f"t_off = {(self._pulse_period - self._pulse_t_on)*1e6:.1f} µs)")
        print(f"    Waveform  : {waveform}"
              + (f", rise/fall = {rise_time_s*1e9:.0f} ns" if rise_time_s > 0 else ""))
        print(f"    P_peak/V  : {self._pulse_P_on_Wm3:.3e} W/m³")
        print(f"    P_avg     : {self._P_avg_W:.3f} W  "
              f"(P_dep_avg = {P_avg_density:.3e} W/m³)")

    def set_vi_envelope(self, P_watts: float, noise_threshold_frac: float = 0.001):
        """Switch V-I curve to envelope mode: P_dep(t) = |V·I| × scale / V_eff.

        The scale factor normalizes so that ⟨P_dep⟩ = P_watts / V_eff.
        Points with |V·I| < noise_threshold_frac × peak are zeroed (afterglow noise).
        Requires V-I curve to be loaded first.

        Args:
            P_watts: measured average power [W] (e.g. from Lissajous)
            noise_threshold_frac: fraction of peak |V·I| below which power is zeroed
        """
        if not self.vi_curve._loaded:
            raise RuntimeError("Load V-I curve before calling set_vi_envelope")
        self._mode = 'vi_envelope'

        V = self.vi_curve.voltage
        I = self.vi_curve.current
        t = self.vi_curve.time
        VI_abs = np.abs(V * I)
        VI_peak = float(np.max(VI_abs))
        self._envelope_threshold = noise_threshold_frac * VI_peak

        # Compute ⟨|V·I|⟩ with threshold applied
        VI_clean = np.where(VI_abs >= self._envelope_threshold, VI_abs, 0.0)
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        vi_clean_avg = float(_trapz(VI_clean, t)) / self.vi_curve.duration

        if vi_clean_avg > 0:
            self._envelope_scale = P_watts / vi_clean_avg
        else:
            self._envelope_scale = 1.0
        self._P_avg_W = P_watts

        n_active = int(np.sum(VI_abs >= self._envelope_threshold))
        duty = n_active / len(V) * 100
        print(f"  Envelope mode: P_input = {P_watts:.2f} W")
        print(f"    Threshold = {self._envelope_threshold:.2f} W "
              f"({noise_threshold_frac*100:.1f}% of peak {VI_peak:.1f} W)")
        print(f"    Active points: {n_active}/{len(V)} ({duty:.1f}%)")
        print(f"    Scale = {self._envelope_scale:.4f}")
        print(f"    P_dep (avg) = {P_watts / self.V_eff:.3e} W/m³")

    def load_vi_curve(self, filepath: str, frequency_Hz: float = None, **kwargs):
        self._mode = 'vi_curve'
        self.vi_curve.load(filepath, **kwargs)

        if frequency_Hz is not None and frequency_Hz > 0:
            self._override_frequency(frequency_Hz)

        self._compute_average_power()

    def _override_frequency(self, frequency_Hz: float):
        p = self.vi_curve.pulse_info
        p.frequency_Hz = frequency_Hz
        p.period_s = 1.0 / frequency_Hz

        n_cycles = max(1, round(self.vi_curve.duration * frequency_Hz))
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        total_energy = float(_trapz(
            self.vi_curve.voltage * self.vi_curve.current,
            self.vi_curve.time))
        p.E_per_pulse_J = total_energy / n_cycles
        p.n_pulses = n_cycles

        print(f"    Frequency (config): {frequency_Hz:.1f} Hz "
              f"(T={p.period_s*1e6:.1f} µs, {n_cycles} cycle(s) in data)")

    def _compute_average_power(self):
        """Compute and report time-averaged power from V-I curve."""
        if not self.vi_curve._loaded:
            return
        p = self.vi_curve.pulse_info
        _trapz = getattr(np, 'trapezoid', None) or np.trapz

        V = self.vi_curve.voltage
        I = self.vi_curve.current
        t = self.vi_curve.time

        self._P_avg_W = float(_trapz(np.abs(V * I), t)) / self.vi_curve.duration

        P_avg_density = self._P_avg_W / self.V_eff
        print(f"    Avg power   : {self._P_avg_W:.3f} W  (⟨|V·I|⟩)")
        print(f"    P_dep (avg) : {P_avg_density:.3e} W/m³")
    
    def _pulsed_power_density(self, t: float) -> float:
        """Parametric pulse waveform evaluation."""
        T = self._pulse_period
        t_on = self._pulse_t_on
        rise = self._pulse_rise_time
        P_on = self._pulse_P_on_Wm3
        t_phase = t % T

        if self._pulse_waveform == 'rectangular' or rise <= 0:
            return P_on if t_phase < t_on else 0.0

        # Trapezoidal: linear ramp up → flat top → linear ramp down → off
        if t_phase < rise:
            return P_on * (t_phase / rise)
        elif t_phase < t_on - rise:
            return P_on
        elif t_phase < t_on:
            return P_on * (t_on - t_phase) / rise
        else:
            return 0.0

    def get_pulse_edges(self, t_start: float, t_end: float) -> List[float]:
        """Return pulse ON/OFF transition times within [t_start, t_end].

        Used by the solver for segmentation at pulse boundaries (avoids BDF
        stiffness at hard discontinuities).

        Returns sorted list of times where power switches state.  Empty for
        constant power or smooth waveforms that don't need segmentation.
        """
        if self._mode == 'pulsed':
            T = self._pulse_period
            t_on = self._pulse_t_on
            rise = self._pulse_rise_time
            edges = []
            k = int(t_start / T)
            while True:
                t0_k = k * T
                if t0_k > t_end:
                    break
                if self._pulse_waveform == 'rectangular' or rise <= 0:
                    # Rectangular: 2 edges per period (ON, OFF)
                    for te in [t0_k, t0_k + t_on]:
                        if t_start < te <= t_end:
                            edges.append(te)
                else:
                    # Trapezoidal: 2 edges per period (ON/OFF boundaries).
                    # The waveform is continuous, so BDF handles derivative
                    # discontinuities at kink points without reinit.
                    for te in [t0_k, t0_k + t_on]:
                        if t_start < te <= t_end:
                            edges.append(te)
                k += 1
            return sorted(set(edges))

        if self._mode in ('vi_envelope', 'vi_curve'):
            p = self.vi_curve.pulse_info
            if p and p.waveform_type == 'pulsed' and p.period_s > 0:
                T = p.period_s
                edges = []
                k = int(t_start / T)
                while True:
                    t_edge = k * T
                    if t_edge > t_end:
                        break
                    if t_start < t_edge <= t_end:
                        edges.append(t_edge)
                    k += 1
                return edges

        return []

    def get_power_density(self, t: float) -> float:
        """Power density [W/m³].

        pulsed:      parametric pulse (rectangular/trapezoidal)
        vi_curve:    signed V·I / V_eff (requires capacitive subtraction for accuracy)
        vi_envelope: |V·I| × scale / V_eff (normalized to P_input, always ≥ 0)
        constant:    P_input / V_eff
        """
        if self._mode == 'constant':
            return self._P_constant_Wm3
        if self._mode == 'pulsed':
            return self._pulsed_power_density(t)
        V, I = self.vi_curve.interpolate(t)
        if self._mode == 'vi_envelope':
            vi_abs = abs(V * I)
            if vi_abs < self._envelope_threshold:
                return 0.0
            return vi_abs * self._envelope_scale / self.V_eff
        return (V * I) / self.V_eff

    def get_voltage(self, t: float) -> float:
        if self._mode == 'constant':
            return 0.0
        V, _ = self.vi_curve.interpolate(t)
        return V

    def get_current(self, t: float) -> float:
        if self._mode == 'constant':
            return 0.0
        _, I = self.vi_curve.interpolate(t)
        return I
    
    def get_recommended_solver_params(self) -> dict:
        if self._mode == 'constant':
            # No pulse structure → no max_step constraint, user sets t_end
            return {'t_end': 1e-4, 'n_points': 2000}

        if self._mode == 'pulsed':
            T = self._pulse_period
            t_on = self._pulse_t_on
            rise = self._pulse_rise_time
            # Resolve rise/fall time: need ~20 steps across the shortest feature
            if self._pulse_waveform == 'trapezoidal' and rise > 0:
                feature_time = rise
            else:
                feature_time = t_on
            max_step = max(feature_time / 20.0, 1e-10)
            t_end = 10 * T  # default: 10 periods
            n_points = max(2000, int(t_end / max_step * 2))
            return {
                'max_step': max_step,
                't_end': t_end,
                'n_points': n_points,
            }

        p = self.vi_curve.pulse_info
        if p is None:
            return {'max_step': 1e-7, 't_end': 1e-4, 'n_points': 2000}
        return {
            'max_step': p.recommended_max_step_s,
            't_end': p.recommended_t_end_s,
            'n_points': max(2000, int(p.recommended_t_end_s / p.recommended_max_step_s * 2)),
        }

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def P_avg_W(self) -> float:
        """Time-averaged power [W]."""
        return self._P_avg_W

    @property
    def frequency(self):
        if self._mode == 'constant':
            return 0.0
        if self._mode == 'pulsed':
            return self._pulse_PRF
        p = self.vi_curve.pulse_info
        return p.frequency_Hz if p else 0.0

    @property
    def period(self):
        if self._mode == 'constant':
            return 0.0
        if self._mode == 'pulsed':
            return self._pulse_period
        p = self.vi_curve.pulse_info
        return p.period_s if p else self.vi_curve.duration
