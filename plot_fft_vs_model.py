from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import read

import plot_sensitivity_model as model


ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = ROOT / "project"
OUTPUT_DIR = Path(__file__).resolve().parent / "images"


def load_trace(path: Path):
    trace = read(str(path))[0]
    data = trace.data.astype(float)
    times = np.arange(trace.stats.npts) * trace.stats.delta
    return trace, times, data


def centered_window(times, data, center_index, width_s):
    half_width = width_s / 2
    center_time = times[center_index]
    mask = (times >= center_time - half_width) & (times <= center_time + half_width)
    window_t = times[mask] - center_time
    window_y = data[mask] - np.mean(data[mask])
    return window_t, window_y


def fft_amplitude(window_y, dt):
    n = len(window_y)
    taper = np.hanning(n)
    y = window_y * taper
    freq = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(y))
    return freq[1:], amp[1:]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("Bridge jumping test", PROJECT_DIR / "2026_03_26_195042-bridge1_B10.sac", "#1b6ca8"),
        ("Bus loop field test", PROJECT_DIR / "2026_03_27_150136_B10.sac", "#b24a2c"),
    ]

    spectra = []
    for label, path, color in datasets:
        trace, times, data = load_trace(path)
        peak_index = int(np.argmax(np.abs(data - np.mean(data))))
        window_t, window_y = centered_window(times, data, peak_index, width_s=80)
        freq, amp = fft_amplitude(window_y, trace.stats.delta)
        spectra.append((label, color, freq, amp / np.max(amp)))

        band = (freq >= 0.04) & (freq <= 0.3)
        band_freq = freq[band]
        band_amp = amp[band]
        peak_f = float(band_freq[np.argmax(band_amp)])
        print(f"{label}: strongest FFT peak in 0.04-0.30 Hz band at {peak_f:.3f} Hz (period {1/peak_f:.2f} s)")

    freq_model = np.geomspace(0.01, model.FS_OUT / 2.0, 3000)
    h_analog = model.analog_feedback_response(freq_model)
    h_decim = np.abs(model.boxcar_decimation_response(freq_model))
    h_fir = np.abs(model.fir_response(freq_model, model.FS_OUT, model.FIR_COEFF))
    h_detrend = np.abs(model.biquad_response(freq_model, model.FS_OUT, model.DETREND_SECTIONS))
    h_boost = model.biquad_response(freq_model, model.FS_OUT, model.BOOST_SECTIONS)
    h_digital = h_decim * h_fir * h_detrend * np.abs(1.0 + model.BFMULT * h_boost)
    h_mech = model.mechanical_velocity_response(freq_model, model.F_NATURAL, model.ZETA)
    h_total = h_mech * h_analog * h_digital

    h_digital_norm = h_digital / np.max(h_digital)
    h_total_norm = h_total / np.max(h_total)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True, constrained_layout=True)

    for ax, (label, color, freq, amp_norm) in zip(axes, spectra):
        ax.semilogx(freq, amp_norm, color=color, linewidth=2.0, label=f"{label} FFT")
        ax.semilogx(freq_model, h_digital_norm, color="#7a3e9d", linestyle="--", linewidth=1.7, label="Acquisition-chain model")
        ax.semilogx(freq_model, h_total_norm, color="#111111", linestyle="-.", linewidth=1.5, label="Combined sensitivity model")
        ax.axvline(1 / 10.5, color="0.45", linestyle=":", linewidth=1.0)
        ax.axvline(model.F_NATURAL, color="0.45", linestyle="--", linewidth=1.0)
        ax.set_ylabel("Normalized amplitude")
        ax.set_ylim(1e-3, 1.2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(frameon=False, loc="upper right")

    axes[0].text(1 / 10.5 * 1.03, 0.4, "10-11 s band", fontsize=9, color="0.35")
    axes[0].text(model.F_NATURAL * 1.03, 0.07, r"$f_n \approx 0.83$ Hz", fontsize=9, color="0.35")
    axes[-1].set_xlabel("Frequency (Hz)")

    out = OUTPUT_DIR / "fft_vs_sensitivity_model.png"
    fig.savefig(out, dpi=220)
    print(f"Saved {out}")

    assumptions = [
        ("Equal displacement", "equal_displacement", "#8c4f2b"),
        ("Equal velocity", "equal_velocity", "#2d6a9f"),
        ("Equal acceleration", "equal_acceleration", "#b24a2c"),
    ]

    fig2, axes2 = plt.subplots(3, 1, figsize=(8.8, 9.0), sharex=True, constrained_layout=True)
    bridge_label, bridge_color, bridge_freq, bridge_amp = spectra[0]
    bus_label, bus_color, bus_freq, bus_amp = spectra[1]

    for ax, (label, key, mech_color) in zip(axes2, assumptions):
        h_mech_assump = model.make_assumption_scaled_mechanical(freq_model, model.F_NATURAL, model.ZETA, key)
        h_total_assump = h_mech_assump * h_analog * h_digital
        h_total_assump /= np.max(h_total_assump)

        ax.semilogx(bridge_freq, bridge_amp, color=bridge_color, linewidth=1.9, label=f"{bridge_label} FFT")
        ax.semilogx(bus_freq, bus_amp, color=bus_color, linewidth=1.9, label=f"{bus_label} FFT")
        ax.semilogx(freq_model, h_digital_norm, color="#7a3e9d", linestyle="--", linewidth=1.6, label="Acquisition-chain model")
        ax.semilogx(freq_model, h_total_assump, color=mech_color, linestyle="-.", linewidth=1.8, label=f"Combined model: {label.lower()}")
        ax.axvline(1 / 10.5, color="0.45", linestyle=":", linewidth=1.0)
        ax.axvline(model.F_NATURAL, color="0.45", linestyle="--", linewidth=1.0)
        ax.text(0.012, 0.45, label, fontsize=10)
        ax.set_ylabel("Normalized amplitude")
        ax.set_ylim(1e-3, 1.2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(frameon=False, loc="upper right")

    axes2[0].text(1 / 10.5 * 1.03, 0.22, "10-11 s band", fontsize=9, color="0.35")
    axes2[0].text(model.F_NATURAL * 1.03, 0.03, r"$f_n \approx 0.83$ Hz", fontsize=9, color="0.35")
    axes2[-1].set_xlabel("Frequency (Hz)")

    out2 = OUTPUT_DIR / "fft_vs_sensitivity_assumptions.png"
    fig2.savefig(out2, dpi=220)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
