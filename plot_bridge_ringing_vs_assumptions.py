from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import read

import plot_sensitivity_model as model


ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = ROOT / "project"
OUTPUT_DIR = Path(__file__).resolve().parent / "images"
BRIDGE_PATH = PROJECT_DIR / "2026_03_26_195042-bridge1_B10.sac"


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


def segment(window_t, window_y, t0, t1):
    mask = (window_t >= t0) & (window_t <= t1)
    seg_t = window_t[mask]
    seg_y = window_y[mask] - np.mean(window_y[mask])
    return seg_t, seg_y


def fft_amplitude(y, dt):
    n = len(y)
    taper = np.hanning(n)
    y = y * taper
    freq = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(y))
    return freq[1:], amp[1:]


def strongest_peak(freq, amp, fmin=0.04, fmax=2.0):
    mask = (freq >= fmin) & (freq <= fmax)
    f = freq[mask]
    a = amp[mask]
    idx = np.argmax(a)
    return float(f[idx]), float(a[idx])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trace, times, data = load_trace(BRIDGE_PATH)
    peak_index = int(np.argmax(np.abs(data - np.mean(data))))
    window_t, window_y = centered_window(times, data, peak_index, width_s=80)

    # Same ringing window used in the bridge segment analysis.
    post_t, post_y = segment(window_t, window_y, 2, 38)
    freq_post, amp_post = fft_amplitude(post_y, trace.stats.delta)
    amp_post_norm = amp_post / np.max(amp_post)

    post_peak_f, _ = strongest_peak(freq_post, amp_post)

    freq_model = np.geomspace(0.01, model.FS_OUT / 2.0, 3000)
    h_analog = model.analog_feedback_response(freq_model)
    h_decim = np.abs(model.boxcar_decimation_response(freq_model))
    h_fir = np.abs(model.fir_response(freq_model, model.FS_OUT, model.FIR_COEFF))
    h_detrend = np.abs(model.biquad_response(freq_model, model.FS_OUT, model.DETREND_SECTIONS))
    h_boost = model.biquad_response(freq_model, model.FS_OUT, model.BOOST_SECTIONS)
    h_digital = h_decim * h_fir * h_detrend * np.abs(1.0 + model.BFMULT * h_boost)
    h_chain_norm = h_digital / np.max(h_digital)

    assumptions = [
        ("Equal displacement", "equal_displacement", "#8c4f2b"),
        ("Equal velocity", "equal_velocity", "#2d6a9f"),
        ("Equal acceleration", "equal_acceleration", "#b24a2c"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.0), sharex=True, constrained_layout=True)

    for ax, (label, key, color) in zip(axes, assumptions):
        h_mech = model.make_assumption_scaled_mechanical(freq_model, model.F_NATURAL, model.ZETA, key)
        h_total = h_mech * h_analog * h_digital
        h_total_norm = h_total / np.max(h_total)

        ax.semilogx(freq_post, amp_post_norm, color="#1b6ca8", linewidth=2.0, label="Bridge post-event ringing FFT")
        ax.semilogx(freq_model, h_chain_norm, color="#7a3e9d", linestyle="--", linewidth=1.7, label="Acquisition-chain model")
        ax.semilogx(freq_model, h_total_norm, color=color, linestyle="-.", linewidth=1.8, label=f"Combined model: {label.lower()}")
        ax.axvline(post_peak_f, color="#1b6ca8", linestyle=":", linewidth=1.0)
        ax.axvline(1 / 10.5, color="0.45", linestyle=":", linewidth=1.0)
        ax.axvline(model.F_NATURAL, color="0.45", linestyle="--", linewidth=1.0)
        ax.text(0.012, 0.42, label, fontsize=10)
        ax.set_ylabel("Normalized amplitude")
        ax.set_ylim(1e-3, 1.2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(frameon=False, loc="upper right")

    axes[0].text(post_peak_f * 1.03, 0.23, f"ringing peak = {post_peak_f:.3f} Hz", fontsize=9, color="#1b6ca8")
    axes[0].text(1 / 10.5 * 1.03, 0.12, "10-11 s band", fontsize=9, color="0.35")
    axes[0].text(model.F_NATURAL * 1.03, 0.03, r"$f_n \approx 0.83$ Hz", fontsize=9, color="0.35")
    axes[-1].set_xlabel("Frequency (Hz)")

    out = OUTPUT_DIR / "bridge_ringing_vs_assumptions.png"
    fig.savefig(out, dpi=220)

    print(f"Bridge post-event ringing peak: {post_peak_f:.3f} Hz (period {1/post_peak_f:.2f} s)")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
