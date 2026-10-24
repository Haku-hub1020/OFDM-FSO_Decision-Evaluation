"""
fso_physics_model.py
=====================
Phase 2A physics-grounded analytical model for OFDM-FSO configuration selection.

Locked methodology reference: see project methodology audit v2 + final
implementation audit. This module implements ONLY the locked equations —
no turbulence, no APD excess noise, no laser phase noise, no diffraction-
limited (wavelength-dependent) divergence.

Pipeline:
    (V, R, L, lambda, M)
      -> atmospheric attenuation (Kim fog OR Carbonneau rain, mutually exclusive)
      -> geometric loss (fixed divergence)
      -> received optical power (link budget)
      -> coherent (heterodyne) SNR
      -> M-QAM BER
      -> Q-factor (derived only)

All physical constants are SI. Wavelength is handled in nanometers at the
API boundary and converted to meters internally where required (Part A/§9
of the audit: never mix nm and m in the photon-energy calculation).
"""

from __future__ import annotations
import numpy as np
from scipy import special
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Dict, Tuple, Literal

# ----------------------------------------------------------------------
# Physical constants (SI)
# ----------------------------------------------------------------------
H_PLANCK = 6.62607015e-34   # J*s
C_LIGHT = 2.99792458e8      # m/s
Q_ELECTRON = 1.602176634e-19  # C


# ----------------------------------------------------------------------
# Configurable system parameters (Final Implementation Audit, Section 16)
# ----------------------------------------------------------------------
@dataclass
class SystemParams:
    """All parameters that are configurable / not yet fully sourced are
    exposed here explicitly rather than hard-coded inside functions, per
    the audit's requirement not to silently choose unsupported values."""

    # Transmitter (Sec 1.2, 1.3 of Final Implementation Audit) — LOCKED
    D_t_m: float = 0.05          # transmit aperture diameter, m (5 cm)
    D_r_m: float = 0.15          # receive aperture diameter, m (15 cm)
    theta_rad: float = 0.005     # full-angle divergence, rad (5 mrad) — FIXED, not wavelength-dependent

    # Transmit power per wavelength (Sec 1.1) — Phase 2A.2 update.
    # 850nm: LOCKED, real commercial datasheet (TereScope 700/G, 16mW).
    # 1250/1550nm: STILL UNRESOLVED as single numbers. IEC 60825-1 itself
    # cannot be reduced to one exact multiplier for our specific expanded-
    # beam geometry (5cm aperture, 5mrad divergence) without applying the
    # full subtense-angle/MHP algorithm from Soltani et al. 2021 (arXiv
    # 2102.08707), whose own worked examples use bare micron-scale beam
    # waists, not an expanded aperture -- applying it to our geometry
    # from scratch would itself be a new, unverified derivation. Instead
    # of picking one number, the eye-safety advantage is represented as
    # a CONFIGURABLE MULTIPLIER over the 850nm baseline, with credible
    # sources disagreeing on its size:
    #   - qualitative commercial claim (EC System FSO vendor): "several
    #     times higher power" at 1550nm for equal eye-safety class
    #   - cited secondary literature (arXiv:1812.11233): 50-65x
    #   - authoritative IEC-based tutorial (arXiv:2102.08707, Tables V/X):
    #     confirms 850/1250nm are BOTH governed by the much stricter
    #     retinal-hazard MPE (700-1400nm band), while 1550nm falls under
    #     the far more permissive cornea/skin MPE (>1400nm band) -- this
    #     structurally supports a LARGE ratio but doesn't pin an exact
    #     number for our geometry.
    # Default below uses the literature-cited value as a working default;
    # Part C sensitivity analysis varies this explicitly across the full
    # disputed range rather than hiding the disagreement.
    eye_safety_multiplier: float = 50.0  # UNRESOLVED -- see comment above
    Pt_dBm: Dict[int, float] = field(default_factory=lambda: {
        850: 12.04,   # 16 mW, TereScope 700/G datasheet -> LOCKED
        1250: 12.04,  # placeholder; use apply_eye_safety_scaling() below
        1550: 12.04,  # placeholder; use apply_eye_safety_scaling() below
    })

    # System margin (Sec 7) — UNRESOLVED, defaults to 0 dB for unit testing
    L_margin_dB: float = 0.0

    # Detector responsivity, A/W (Sec 8, updated Phase 2A.2).
    # 850nm: LOCKED (Si, Hamamatsu-class datasheets, ~0.5 A/W typical).
    # 1550nm: LOCKED (InGaAs, OSI Optoelectronics FCI-INGAAS-1500 datasheet,
    #   0.95 A/W at 1550nm exactly).
    # 1250nm: DERIVED (not a free choice) via R(lambda) = eta*q*lambda/(hc);
    #   for roughly constant quantum efficiency across InGaAs's flat
    #   response region (900-1600nm per Edmund Optics), R scales linearly
    #   with lambda. Anchored to the SAME datasheet's 1310nm point
    #   (0.9 A/W): R(1250) = 0.9 * (1250/1310) = 0.8588 A/W. This is a
    #   physically-derived interpolation, not an arbitrary number, but it
    #   is still NOT a direct datasheet read at 1250nm -- flagged as such.
    responsivity_AW: Dict[int, float] = field(default_factory=lambda: {
        850: 0.5,      # Si, Hamamatsu-class datasheets -> LOCKED
        1250: 0.9 * (1250.0 / 1310.0),  # derived, constant-QE interpolation -> 0.8588
        1550: 0.95,    # InGaAs, OSI Optoelectronics datasheet -> LOCKED
    })

    # Environmental/system measurement uncertainty (Sec A3, Phase 2A.2).
    # Separated explicitly into ENVIRONMENTAL (sourced from real instrument
    # datasheets this session) and SYSTEM (not independently sourced this
    # session -- kept as engineering-typical placeholders, flagged).
    uncertainty: Dict[str, float] = field(default_factory=lambda: {
        # ENVIRONMENTAL -- sourced:
        "visibility_fraction": 0.10,  # Vaisala PWD10/12/22 datasheet:
                                       # "Accuracy: +/-10% at 10-2000m".
                                       # Note: a separate IEEE Xplore
                                       # comparison study found +/-20% at
                                       # 10,000-20,000m for 1-min data,
                                       # improving to +/-10% for 10-min
                                       # averages -- +/-10% used here as
                                       # the more conservative near-
                                       # threshold (fog-relevant) figure,
                                       # not the best case.
        "rain_fraction": 0.05,        # R.M. Young tipping-bucket spec:
                                       # +/-2% up to 25mm/h, +/-3% up to
                                       # 50mm/h (calibrated). Hydrological
                                       # Services notes UNCORRECTED gauges
                                       # can err up to 15% at high
                                       # intensity. 0.05 is a stated
                                       # mid-range compromise across our
                                       # full 0-100mm/h range, not a
                                       # single-source number -- document
                                       # this compromise explicitly.
        # SYSTEM -- NOT independently sourced this session, engineering-
        # typical placeholders only:
        "power_fraction": 0.05,       # UNRESOLVED -- typical commercial
                                       # laser diode power stability specs
                                       # are commonly ~+/-0.5-1 dB, but no
                                       # specific datasheet was verified
                                       # for THIS assumption this session.
        "responsivity_fraction": 0.05,  # UNRESOLVED -- same caveat.
    })

    # Fixed symbol-rate / fixed receiver-bandwidth model (Phase 2A.1
    # consistency correction). Rs = B = 2.5 GHz for BOTH 4-QAM and 16-QAM
    # -- this models a receiver of fixed physical bandwidth serving both
    # modulation orders, so bit rate (not bandwidth) is what varies with
    # M. This replaces the earlier fixed-DATA-RATE model, which was
    # inconsistent with treating log2(M) as "more throughput" (a fixed
    # data rate by definition cannot yield more throughput at higher M).
    symbol_rate_hz: float = 2.5e9

    # BER decision threshold (Final Implementation Audit Sec 5.1) — LOCKED
    ber_threshold: float = 1e-9

    # Candidate configuration space, in a FIXED order (used for
    # deterministic tie-breaking in select_configuration)
    wavelengths_nm: Tuple[int, ...] = (850, 1250, 1550)
    modulation_orders: Tuple[int, ...] = (4, 16)


DEFAULT_PARAMS = SystemParams()


def apply_eye_safety_scaling(params: SystemParams, multiplier: float = None
                              ) -> SystemParams:
    """Returns a NEW SystemParams with Pt(1250) and Pt(1550) scaled up from
    the LOCKED 850nm baseline by `multiplier` (linear power ratio, not dB).
    Does not mutate the input. If multiplier is None, uses
    params.eye_safety_multiplier. This keeps the unresolved multiplier
    explicit and swappable rather than baked into a hard-coded constant --
    used directly by the Part C sensitivity analysis to sweep the disputed
    range (1x / ~5x / 50x / 65x) rather than silently picking one.
    """
    import copy
    m = multiplier if multiplier is not None else params.eye_safety_multiplier
    new_params = copy.deepcopy(params)
    baseline_W = dbm_to_watt(params.Pt_dBm[850])
    new_params.Pt_dBm[1250] = watt_to_dbm(baseline_W * m)
    new_params.Pt_dBm[1550] = watt_to_dbm(baseline_W * m)
    new_params.eye_safety_multiplier = m
    return new_params


def verify_quantum_efficiencies(params: SystemParams = DEFAULT_PARAMS
                                 ) -> Dict[int, float]:
    """Returns {wavelength: eta} for every configured responsivity, so
    physical plausibility (0 < eta < ~1, allowing some margin for
    avalanche-adjacent or imperfect-antireflection-coating real detectors)
    can be checked directly rather than assumed (Part A2 requirement)."""
    return {wl: responsivity_to_quantum_efficiency(r, wl)
            for wl, r in params.responsivity_AW.items()}


# ----------------------------------------------------------------------
# 1. Fog attenuation (Kim model)
# ----------------------------------------------------------------------
def kim_P_coefficient(V_km: float) -> float:
    """Piecewise Kim-model scattering coefficient P(V). V in km."""
    if V_km > 50:
        return 1.6
    elif V_km > 6:
        return 1.3
    elif V_km > 1:
        return 0.16 * V_km + 0.34
    elif V_km > 0.5:
        return V_km - 0.5
    else:
        return 0.0


def kim_fog_attenuation(V_km: float, wavelength_nm: float) -> float:
    """Kim fog attenuation coefficient, dB/km.

    beta_fog(lambda) = (3.91/V) * (lambda/550)^(-P)

    V in km, wavelength in nm (550 nm reference is dimensionally consistent
    since it's a ratio of two nm quantities).
    """
    if V_km <= 0:
        raise ValueError("Visibility must be positive.")
    P = kim_P_coefficient(V_km)
    beta = (3.91 / V_km) * (wavelength_nm / 550.0) ** (-P)
    return beta  # dB/km


# ----------------------------------------------------------------------
# 2. Rain attenuation (Carbonneau/France parametrization)
# ----------------------------------------------------------------------
def rain_attenuation(R_mmh: float) -> float:
    """Carbonneau rain attenuation coefficient, dB/km.
    gamma_rain = 1.076 * R^0.67  (R in mm/h)
    Wavelength-independent per ITU-R P.1814-1 for 850/1550 nm; extended to
    1250 nm as an explicit modeling assumption (Final Implementation Audit).
    """
    if R_mmh < 0:
        raise ValueError("Rain rate cannot be negative.")
    if R_mmh == 0:
        return 0.0
    return 1.076 * (R_mmh ** 0.67)  # dB/km


# ----------------------------------------------------------------------
# 3. Atmospheric loss (mutually-exclusive scenario dispatch)
# ----------------------------------------------------------------------
Scenario = Literal["clear", "fog", "rain"]


def atmospheric_loss(scenario: Scenario, V_km: float, R_mmh: float,
                      L_m: float, wavelength_nm: float) -> float:
    """Total atmospheric loss in dB for ONE mutually-exclusive scenario.
    Never adds fog and rain losses together (locked v1 simplification)."""
    L_km = L_m / 1000.0
    if scenario == "clear":
        return 0.0
    elif scenario == "fog":
        beta = kim_fog_attenuation(V_km, wavelength_nm)
        return beta * L_km
    elif scenario == "rain":
        gamma = rain_attenuation(R_mmh)
        return gamma * L_km
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# ----------------------------------------------------------------------
# 4. Geometric loss (FIXED divergence, not wavelength-dependent)
# ----------------------------------------------------------------------
def geometric_loss(L_m: float, D_t_m: float = DEFAULT_PARAMS.D_t_m,
                    D_r_m: float = DEFAULT_PARAMS.D_r_m,
                    theta_rad: float = DEFAULT_PARAMS.theta_rad) -> float:
    """GL(dB) = 10*log10[ D_r^2 / (D_t + L*theta)^2 ]  (Alkholidi 2014).

    Returns the SIGNED value of this expression directly, which is
    NEGATIVE for any realistic link (D_r < D_t + L*theta once L is more
    than a few tens of centimetres). This signed negative dB quantity is
    a loss already in the correct sign convention for the link budget:
    received_power_dbm() ADDS it directly (Pr = Pt - L_atm + GL - margin),
    since adding a negative number reduces Pr, exactly as a loss should.
    Do not negate this value before use.
    """
    denom = (D_t_m + L_m * theta_rad) ** 2
    ratio = (D_r_m ** 2) / denom
    return 10 * np.log10(ratio)


# ----------------------------------------------------------------------
# 5. Photon energy / frequency
# ----------------------------------------------------------------------
def wavelength_to_frequency(wavelength_nm: float) -> float:
    """nu = c / lambda. Converts nm -> m internally. Returns Hz."""
    wavelength_m = wavelength_nm * 1e-9
    return C_LIGHT / wavelength_m


def photon_energy(wavelength_nm: float) -> float:
    """h*nu in Joules."""
    nu = wavelength_to_frequency(wavelength_nm)
    return H_PLANCK * nu


# ----------------------------------------------------------------------
# 6. Responsivity <-> quantum efficiency conversion
# ----------------------------------------------------------------------
def responsivity_to_quantum_efficiency(responsivity_AW: float,
                                        wavelength_nm: float) -> float:
    """R = eta*q/(h*nu)  =>  eta = R*h*nu/q   (dimensionless).

    Dimension check: [A/W] * [J] / [C]
      = [C/s/W] * [J] / [C] = [J/(s*W)] = [W/W] = dimensionless.  OK.
    """
    hv = photon_energy(wavelength_nm)
    eta = responsivity_AW * hv / Q_ELECTRON
    return eta


# ----------------------------------------------------------------------
# 7. Receiver bandwidth / bit rate (Phase 2A.1: FIXED symbol-rate model)
# ----------------------------------------------------------------------
def receiver_bandwidth(symbol_rate_hz: float) -> float:
    """B = Rs, held FIXED across modulation orders (Phase 2A.1 correction).

    This is the convention that makes SNR (as computed by coherent_snr)
    equal to Es/N0 exactly, for EITHER modulation order, since both share
    the same symbol rate / receiver bandwidth. M no longer appears here --
    that is the point of this correction (bandwidth does not shrink as M
    grows; bit rate grows instead, see bit_rate() below).
    """
    return symbol_rate_hz


def bit_rate(symbol_rate_hz: float, M: int) -> float:
    """Rb = Rs * log2(M) -- the actual information bit rate delivered by
    a given modulation order at the fixed symbol rate. THIS, not log2(M)
    alone, is the correct throughput quantity for configuration selection
    (Phase 2A.1 correction, Sec 1)."""
    return symbol_rate_hz * np.log2(M)


# ----------------------------------------------------------------------
# 8. dBm <-> Watt conversions
# ----------------------------------------------------------------------
def dbm_to_watt(p_dbm: float) -> float:
    return 10 ** ((p_dbm - 30) / 10)


def watt_to_dbm(p_watt: float) -> float:
    if p_watt <= 0:
        raise ValueError("Power must be positive to convert to dBm.")
    return 10 * np.log10(p_watt) + 30


# ----------------------------------------------------------------------
# 9. Link budget -> received power
# ----------------------------------------------------------------------
def received_power_dbm(Pt_dBm: float, L_atm_dB: float, GL_dB_signed: float,
                        L_margin_dB: float = 0.0) -> float:
    """P_r(dBm) = P_t(dBm) - L_atm + GL_dB_signed - L_margin.

    GL_dB_signed is the signed (negative) output of geometric_loss().
    Because it is already negative, ADDING it here correctly reduces Pr
    by its magnitude, in the same way L_atm (a positive dB quantity) is
    SUBTRACTED to reduce Pr. Both terms reduce received power; they just
    carry opposite signs by construction, and this function adds each
    term exactly once with the sign it already has -- no double negation.
    """
    return Pt_dBm - L_atm_dB + GL_dB_signed - L_margin_dB


# ----------------------------------------------------------------------
# 10. Coherent (heterodyne) SNR
# ----------------------------------------------------------------------
def coherent_snr(Pr_W: float, eta: float, wavelength_nm: float,
                  B_Hz: float) -> float:
    """SNR_het = eta * Pr / (h*nu*B).  Dimensionless.
    Assumptions (Final Implementation Audit Sec 3/11): strong LO,
    LO-shot-noise-limited, thermal noise & LO RIN neglected, PIN detector,
    no phase noise, no APD excess-noise factor.
    """
    hv = photon_energy(wavelength_nm)
    return (eta * Pr_W) / (hv * B_Hz)


def coherent_snr_homodyne(snr_heterodyne: float) -> float:
    """Optional sensitivity function ONLY — not used in the primary
    dataset per the locked methodology (heterodyne is primary)."""
    return 2.0 * snr_heterodyne


# ----------------------------------------------------------------------
# 11. M-QAM BER (numerically stable via log-domain Q-function)
# ----------------------------------------------------------------------
def mqam_ber(M: int, SNR: float) -> Tuple[float, float]:
    """Square M-QAM AWGN BER (Proakis), Gray-coded.

    BER ~= [4/log2(M)] * (1 - 1/sqrt(M)) * Q( sqrt(3/(M-1) * SNR) )

    SNR here is defined at symbol-rate bandwidth (Es/N0 convention, see
    receiver_bandwidth()) -- under the Phase 2A.1 fixed-Rs model this is
    the SAME bandwidth for both M=4 and M=16, so this function's SNR
    argument is directly comparable across modulation orders at a given
    link condition (Test 5 verifies this explicitly).

    Returns (BER, log10_BER_raw). log10_BER_raw is the numerically stable,
    UNCENSORED physical value, computed via the log-survival-function path
    (scipy.stats.norm.logsf) so it remains finite and meaningful even when
    BER underflows to exactly 0.0 in double precision (Test 10). This is
    the raw physical quantity -- see censor_log10_ber() for the SEPARATE
    ML/reporting target derived from it. Do NOT confuse the two: raw is
    physical (unbounded, can reach absurd magnitudes like -500,000 for
    very high SNR, as the sanity plots showed); target is a clipped
    quantity for regression/reporting only.
    """
    if SNR < 0:
        raise ValueError("SNR cannot be negative.")
    if M not in (4, 16, 64, 256):
        raise ValueError("M must be a valid square-QAM order (4,16,64,256,...).")

    log2M = np.log2(M)
    prefactor = (4.0 / log2M) * (1.0 - 1.0 / np.sqrt(M))
    x = np.sqrt((3.0 / (M - 1)) * SNR)

    # log(Q(x)) = logsf(x) for standard normal -- numerically stable for
    # large x (avoids computing Q(x) directly, which underflows to 0.0
    # long before its logarithm becomes uninformative).
    log_Q = norm.logsf(x)  # natural log

    log10_BER_raw = (np.log(prefactor) + log_Q) / np.log(10.0)

    # BER itself: may legitimately underflow to 0.0 for very large x;
    # that's a correct float64 result, not a bug -- log10_BER_raw remains
    # the numerically meaningful (raw, uncensored) quantity in that regime.
    with np.errstate(under="ignore"):
        BER = 10.0 ** log10_BER_raw

    return float(BER), float(log10_BER_raw)


def censor_log10_ber(log10_ber_raw: float, floor: float = -20.0) -> float:
    """ML/reporting target: log10_BER_target = max(log10_BER_raw, floor).

    This is an ML-TARGET CENSORING/CLIPPING LIMIT, not a physical model.
    It is NOT a thermal-noise floor, receiver-saturation model, or claim
    that real hardware cannot achieve BER below 10^-20 -- it exists purely
    because (a) the research decision boundary is BER<=1e-9, so anything
    at or below ~1e-20 is operationally indistinguishable from "passes
    comfortably" for that decision, and (b) uncensored values (which can
    reach -500,000+ per the sanity-check findings) would dominate a
    regression loss function and provide no genuine learning signal in
    that regime. log10_BER_raw remains available, unmodified, alongside
    this target -- both are returned by run_scenario().
    """
    return max(log10_ber_raw, floor)


# ----------------------------------------------------------------------
# 12. Q-factor (derived only, per locked methodology -- never a separate
#     ML target in Phase 2A)
# ----------------------------------------------------------------------
def q_factor(BER: float) -> float:
    """BER = 0.5*erfc(Q/sqrt(2))  =>  Q = sqrt(2)*erfcinv(2*BER).

    Returns np.inf when BER underflows to exactly 0.0 (mathematically
    correct: BER->0 implies Q->infinity). Prefer log10_BER_raw over Q for any
    downstream numerical/ML use in the extreme-low-BER regime.
    """
    if BER < 0 or BER > 1:
        raise ValueError("BER must be in [0, 1].")
    if BER == 0.0:
        return float(np.inf)
    return float(np.sqrt(2) * special.erfcinv(2 * BER))


# ----------------------------------------------------------------------
# 13. Weather regime classification (Kim breakpoints + AMS rain standard)
# ----------------------------------------------------------------------
def classify_weather_regime(V_km: float, R_mmh: float) -> str:
    """Returns a neutral, numerically-anchored label. Regime label is for
    LOWCO experiment bookkeeping ONLY -- never fed to the ML model as a
    feature (locked dataset schema)."""
    if R_mmh > 0:
        if R_mmh < 2.5:
            return "Rain: Light (<2.5 mm/h)"
        elif R_mmh <= 7.6:
            return "Rain: Moderate (2.5-7.6 mm/h)"
        else:
            return "Rain: Heavy (>7.6 mm/h)"
    # Fog/clear branch (R == 0)
    if V_km > 50:
        return "Clear (V>50km)"
    elif V_km > 6:
        return "Visibility Regime 1 (6-50km)"
    elif V_km > 1:
        return "Visibility Regime 2 (1-6km)"
    elif V_km > 0.5:
        return "Dense Fog (0.5-1km)"
    else:
        return "Very Dense Fog (<0.5km)"


# ----------------------------------------------------------------------
# 14. Configuration selection (deterministic, NOT learned)
# ----------------------------------------------------------------------
def select_configuration(ber_by_config: Dict[Tuple[int, int], float],
                          params: SystemParams = DEFAULT_PARAMS
                          ) -> Tuple[Tuple[int, int], str]:
    """ber_by_config: {(wavelength_nm, M): BER, ...} for all 6 candidates
    at ONE environmental state.

    Decision rule (Phase 2A.1 correction):
        argmax_(lambda,M) Rb(M)   subject to   BER <= threshold
    where Rb(M) = bit_rate(params.symbol_rate_hz, M) is the ACTUAL bit
    rate at the fixed symbol rate -- NOT log2(M) alone, per the
    consistency correction (a fixed data-rate model cannot simultaneously
    treat log2(M) as "more throughput"; the fixed-symbol-rate model
    resolves this, since Rb genuinely increases with M here).

    Deterministic tie-break: among configurations tied on Rb AND
    satisfying the constraint, select the one that appears FIRST in the
    fixed candidate order (params.wavelengths_nm x params.modulation_orders,
    wavelength-major). Test 9 verifies this.

    Returns (selected_config, reason_string). If NO configuration meets
    the threshold, returns the configuration with the lowest BER instead
    (best-effort fallback), with reason string noting this.
    """
    fixed_order = [(lam, m) for lam in params.wavelengths_nm
                   for m in params.modulation_orders]

    passing = [(lam, m) for (lam, m) in fixed_order
               if ber_by_config[(lam, m)] <= params.ber_threshold]

    if passing:
        rb = {cfg: bit_rate(params.symbol_rate_hz, cfg[1]) for cfg in passing}
        best_rb = max(rb.values())
        tied = [cfg for cfg in passing if rb[cfg] == best_rb]
        selected = tied[0]  # first in fixed_order due to list comprehension order
        reason = (f"max bit rate ({best_rb/1e9:.2f} Gbps at Rs="
                  f"{params.symbol_rate_hz/1e9:.2f} Gbaud) among configs "
                  f"meeting BER<={params.ber_threshold:.0e}; tie-break = "
                  f"first in fixed wavelength-major order")
    else:
        selected = min(fixed_order, key=lambda cfg: ber_by_config[cfg])
        reason = ("NO configuration met the BER threshold; "
                  "fallback = lowest-BER configuration")

    return selected, reason


# ----------------------------------------------------------------------
# Convenience: full pipeline for one (scenario, V, R, L, lambda, M)
# ----------------------------------------------------------------------
def run_scenario(scenario: Scenario, V_km: float, R_mmh: float, L_m: float,
                  wavelength_nm: int, M: int,
                  params: SystemParams = DEFAULT_PARAMS) -> dict:
    """Runs the full locked pipeline for one configuration and returns
    every intermediate quantity (useful for unit tests and sanity plots)."""
    L_atm = atmospheric_loss(scenario, V_km, R_mmh, L_m, wavelength_nm)
    GL = geometric_loss(L_m, params.D_t_m, params.D_r_m, params.theta_rad)

    Pt_dBm = params.Pt_dBm[wavelength_nm]
    Pr_dBm = received_power_dbm(Pt_dBm, L_atm, GL, params.L_margin_dB)
    Pr_W = dbm_to_watt(Pr_dBm)

    responsivity = params.responsivity_AW[wavelength_nm]
    eta = responsivity_to_quantum_efficiency(responsivity, wavelength_nm)

    # Phase 2A.1: B is FIXED (=symbol_rate_hz) regardless of M; bit rate
    # Rb, not bandwidth, is what varies with M.
    B = receiver_bandwidth(params.symbol_rate_hz)
    Rb = bit_rate(params.symbol_rate_hz, M)
    SNR = coherent_snr(Pr_W, eta, wavelength_nm, B)

    BER, log10_BER_raw = mqam_ber(M, SNR)
    log10_BER_target = censor_log10_ber(log10_BER_raw)
    Q = q_factor(BER)

    return dict(
        scenario=scenario, V_km=V_km, R_mmh=R_mmh, L_m=L_m,
        wavelength_nm=wavelength_nm, M=M,
        L_atm_dB=L_atm, GL_dB=GL, Pt_dBm=Pt_dBm, Pr_dBm=Pr_dBm, Pr_W=Pr_W,
        responsivity_AW=responsivity, eta=eta, B_Hz=B, Rb_bps=Rb, SNR=SNR,
        BER=BER, log10_BER_raw=log10_BER_raw,
        log10_BER_target=log10_BER_target, Q_factor=Q,
    )
