from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "images"


FS_OUT = 18.78
F_NATURAL = 50.0 / 60.0  # Hz
ZETA = 0.05  # assumed damping ratio for a trial model
BFD_B = 20.0
BFMULT = 10 ** (BFD_B / 20.0)

FIR_COEFF = np.array(
    [
        0.012325749757,
        0.026308035066,
        0.047864432383,
        0.073557450816,
        0.099301953437,
        0.120152272037,
        0.131681883090,
        0.131301713951,
        0.119096929920,
        0.097799496799,
        0.071908943807,
        0.046358288197,
        0.025177758950,
        0.011494677450,
        -0.000382643068,
        0.000003073960,
    ]
)

DETREND_SECTIONS = [
    (0.9917049902712847, [1.0, -1.0, 0.0], [1.0, -0.9834099805425692, 0.0]),
]

BOOST_SECTIONS = [
    (
        9.7663354168330612e-05,
        [1.0, 2.0, 1.0],
        [1.0, -1.9844039826314028, 9.8532892360982915e-01],
    ),
    (
        1.0,
        [1.0, -2.0, 1.0],
        [1.0, -1.9908268678021406, 9.9116152920232115e-01],
    ),
]


def biquad_response(freq_hz: np.ndarray, fs_hz: float, sections):
    zinv = np.exp(-1j * 2 * np.pi * freq_hz / fs_hz)
    response = np.ones_like(freq_hz, dtype=complex)
    for gain, b, a in sections:
        num = b[0] + b[1] * zinv + b[2] * zinv**2
        den = a[0] + a[1] * zinv + a[2] * zinv**2
        response *= gain * num / den
    return response


def fir_response(freq_hz: np.ndarray, fs_hz: float, coeff: np.ndarray):
    n = np.arange(len(coeff))
    expo = np.exp(-1j * 2 * np.pi * freq_hz[:, None] * n[None, :] / fs_hz)
    return np.sum(expo * coeff[None, :], axis=1)


def boxcar_decimation_response(freq_hz: np.ndarray):
    # The ADC stage effectively averages over a 2048-sample window before
    # outputting data at 18.78 Hz. The first zero is therefore near 18.78/4 Hz.
    first_zero = FS_OUT / 4.0
    return np.sinc(freq_hz / first_zero)


def analog_feedback_response(freq_hz: np.ndarray, r_feedback=866e3, c_feedback=330e-12):
    fc = 1.0 / (2.0 * np.pi * r_feedback * c_feedback)
    return 1.0 / np.sqrt(1.0 + (freq_hz / fc) ** 2)


def mechanical_velocity_response(freq_hz: np.ndarray, f_natural: float, zeta: float):
    r = freq_hz / f_natural
    denom = np.sqrt((1.0 - r**2) ** 2 + (2.0 * zeta * r) ** 2)
    # Relative displacement for harmonic base displacement of fixed amplitude.
    rel_disp = r**2 / denom
    # Coil voltage is proportional to relative velocity of magnet and coil.
    return 2.0 * np.pi * freq_hz * rel_disp


def make_assumption_scaled_mechanical(freq_hz: np.ndarray, f_natural: float, zeta: float, assumption: str):
    """
    Return relative mechanical sensitivity under different input-normalization assumptions.

    assumption = "equal_displacement": fixed base displacement amplitude Y
    assumption = "equal_velocity": fixed base velocity amplitude, so Y ~ 1/omega
    assumption = "equal_acceleration": fixed base acceleration amplitude, so Y ~ 1/omega^2
    """
    base = mechanical_velocity_response(freq_hz, f_natural, zeta)
    omega = 2.0 * np.pi * freq_hz
    if assumption == "equal_displacement":
        return base
    if assumption == "equal_velocity":
        return base / omega
    if assumption == "equal_acceleration":
        return base / (omega**2)
    raise ValueError(f"Unknown assumption: {assumption}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    freq = np.geomspace(0.01, FS_OUT / 2.0, 2000)

    h_analog = analog_feedback_response(freq)
    h_decim = np.abs(boxcar_decimation_response(freq))
    h_fir = np.abs(fir_response(freq, FS_OUT, FIR_COEFF))
    h_detrend = np.abs(biquad_response(freq, FS_OUT, DETREND_SECTIONS))
    h_boost = biquad_response(freq, FS_OUT, BOOST_SECTIONS)
    h_digital = h_decim * h_fir * h_detrend * np.abs(1.0 + BFMULT * h_boost)
    h_mech = mechanical_velocity_response(freq, F_NATURAL, ZETA)
    h_total = h_mech * h_analog * h_digital

    mode1 = h_mech * h_analog * h_decim
    mode2 = h_mech * h_analog * h_decim * h_fir
    mode3 = h_mech * h_analog * h_decim * h_fir * h_detrend
    mode4 = h_total

    # Normalize for relative comparison.
    h_mech_norm = h_mech / np.max(h_mech)
    h_chain_norm = (h_analog * h_digital) / np.max(h_analog * h_digital)
    h_total_norm = h_total / np.max(h_total)

    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    ax.semilogx(freq, h_mech_norm, label="Mechanical stage", color="#8c4f2b", linewidth=2.0)
    ax.semilogx(freq, h_chain_norm, label="Acquisition chain", color="#2d6a9f", linewidth=2.0)
    ax.semilogx(freq, h_total_norm, label="Combined relative sensitivity", color="#111111", linewidth=2.4)

    ax.axvline(F_NATURAL, color="#8c4f2b", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(2.5, color="#2d6a9f", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.text(F_NATURAL * 1.04, 0.87, r"$f_n \approx 0.83$ Hz", color="#8c4f2b", fontsize=10)
    ax.text(2.5 * 1.04, 0.33, "Digital LP scale", color="#2d6a9f", fontsize=10)

    ax.set_xlim(freq.min(), freq.max())
    ax.set_ylim(1e-4, 1.15)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized sensitivity")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, loc="lower left")

    out = OUTPUT_DIR / "sensitivity_model.png"
    fig.savefig(out, dpi=220)
    print(f"Saved {out}")

    fig2, ax2 = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    ax2.semilogx(freq, mode1 / np.max(mode1), label="Mode 1: oversampled ADC", linewidth=2.0, color="#5b8c5a")
    ax2.semilogx(freq, mode4 / np.max(mode4), label="Mode 4: default processing", linewidth=2.2, color="#111111")
    ax2.axvline(F_NATURAL, color="0.4", linestyle="--", linewidth=1.0)
    ax2.set_xlim(freq.min(), freq.max())
    ax2.set_ylim(1e-4, 1.15)
    ax2.set_yscale("log")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Normalized sensitivity")
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(frameon=False, loc="lower left")
    out2 = OUTPUT_DIR / "sensitivity_modes_comparison.png"
    fig2.savefig(out2, dpi=220)
    print(f"Saved {out2}")

    # assumptions = [
    #     ("Equal displacement", "equal_displacement", "#8c4f2b"),
    #     ("Equal velocity", "equal_velocity", "#2d6a9f"),
    #     ("Equal acceleration", "equal_acceleration", "#b24a2c"),
    # ]
    # fig3, axes3 = plt.subplots(3, 1, figsize=(8.6, 8.6), sharex=True, constrained_layout=True)
    # for ax, (label, key, color) in zip(axes3, assumptions):
    #     h_mech_assump = make_assumption_scaled_mechanical(freq, F_NATURAL, ZETA, key)
    #     h_total_assump = h_mech_assump * h_analog * h_digital
    #     h_mech_assump /= np.max(h_mech_assump)
    #     h_total_assump /= np.max(h_total_assump)

    #     ax.semilogx(freq, h_mech_assump, color=color, linewidth=2.0, label=f"Mechanical: {label.lower()}")
    #     ax.semilogx(freq, h_total_assump, color="#111111", linewidth=2.2, linestyle="--", label="Combined sensitivity")
    #     ax.axvline(F_NATURAL, color="0.45", linestyle=":", linewidth=1.0)
    #     ax.set_ylabel("Normalized")
    #     ax.set_yscale("log")
    #     ax.set_ylim(1e-4, 1.2)
    #     ax.grid(True, which="both", alpha=0.22)
    #     ax.legend(frameon=False, loc="lower left")
    #     ax.text(0.012, 0.55, label, fontsize=10)

    # axes3[-1].set_xlabel("Frequency (Hz)")
    # out3 = OUTPUT_DIR / "sensitivity_assumption_comparison.png"
    # fig3.savefig(out3, dpi=220)
    # print(f"Saved {out3}")

    peak_idx = int(np.argmax(h_total_norm))
    print(f"Peak combined sensitivity at {freq[peak_idx]:.3f} Hz")
    print(f"Combined sensitivity at 0.83 Hz: {np.interp(F_NATURAL, freq, h_total_norm):.3f}")
    print(f"Combined sensitivity at 2.5 Hz: {np.interp(2.5, freq, h_total_norm):.3f}")


if __name__ == "__main__":
    main()
