from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import read


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


def summarize(data):
    centered = data - np.mean(data)
    rms = np.sqrt(np.mean(centered**2))
    p2p = centered.max() - centered.min()
    return rms, p2p


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("Bridge jumping test", PROJECT_DIR / "2026_03_26_195042-bridge1_B10.sac", "#1b6ca8"),
        ("Bus loop field test", PROJECT_DIR / "2026_03_27_150136_B10.sac", "#b24a2c"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, constrained_layout=True)

    for ax, (label, path, color) in zip(axes, datasets):
        trace, times, data = load_trace(path)
        peak_index = int(np.argmax(np.abs(data - np.mean(data))))
        window_t, window_y = centered_window(times, data, peak_index, width_s=80)
        rms, p2p = summarize(window_y)

        ax.plot(window_t, window_y, color=color, linewidth=1.1)
        ax.axvline(0, color="0.35", linestyle="--", linewidth=0.9)
        ax.set_ylabel("ADC counts")
        ax.set_title(
            f"{label}: 80 s window around the strongest transient\n"
            f"peak-to-peak = {p2p:.0f} counts, RMS = {rms:.0f} counts",
            fontsize=11,
        )
        ax.grid(alpha=0.25)

        peak_time = trace.stats.starttime + peak_index * trace.stats.delta
        print(f"{label}: {path.name}")
        print(f"  peak time: {peak_time}")
        print(f"  window RMS: {rms:.2f} counts")
        print(f"  window peak-to-peak: {p2p:.2f} counts")

    axes[-1].set_xlabel("Time relative to strongest transient (s)")
    # fig.suptitle("Representative traces from the electromagnetic seismometer", fontsize=13)

    out_path = OUTPUT_DIR / "bridge_bus_trace_comparison.png"
    fig.savefig(out_path, dpi=220)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
