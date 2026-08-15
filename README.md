# Repo Guide

This repository documents the building, modeling, and field testing of a low-frequency electromagnetic seismometer. It contains the report source, generated figures, analysis scripts, and compiled outputs.

## Folder Structure

### Main report files

- `project.tex`
  Main LaTeX source for the final project report.

- [`project.pdf`](project.pdf)
  Compiled PDF of the current report draft.

- `electromagnetic_seismometer_report.pdf`
  Portfolio-ready copy of the compiled report.

### Sensitivity-model sandbox

- `sensitivity_model.tex`
  Separate LaTeX note used to test and explain the sensitivity model without cluttering the main report.

- `sensitivity_model.pdf`
  Compiled PDF of that modeling note.

### Images

All report figures live in `images/`.

- `breadboard.png`
  Photo of the breadboard implementation of the amplifier.

- `initializing.png`
  Startup oscillation image showing the settling behavior after acquisition begins.

- `bridge_bus_trace_comparison.png`
  Time-domain comparison of the bridge and bus-loop traces around the strongest transient.

- `sensitivity_model.png`
  Plot of mechanical response, acquisition-chain response, and combined relative sensitivity.

- `sensitivity_modes_comparison.png`
  Plot comparing the relative sensitivity of `nerdaqII` mode 1 and mode 4.

- `fft_vs_sensitivity_model.png`
  FFT of the 80 s bridge and bus-loop windows overlaid with the sensitivity model.

- `bridge_segment_fft.png`
  Bridge-only FFT comparison between pre-event background and post-event ringing, also overlaid with the model.

- `bus_loop.jpeg`
  Annotated time-domain bus-loop measurement used in the report.

- `bus_loop_110500.png`
  Additional bus-loop jAmaSeis screenshot.

- `bus_loop_frequency.png`
  Frequency-domain screenshot from the bus-loop experiment.

- `bus_loop_period.png`
  Period-domain screenshot from the bus-loop experiment.

## Python Scripts

### `plot_project_data.py`

Purpose:
- Reads the raw `.sac` data for the bridge test and bus-loop test.
- Finds the strongest transient in each dataset.
- Extracts an 80 s window centered on that transient.
- Plots the time-domain traces and reports RMS and peak-to-peak values.

Main output:
- `images/bridge_bus_trace_comparison.png`

### `plot_sensitivity_model.py`

Purpose:
- Builds a simplified theoretical sensitivity model of the whole instrument.
- Combines:
  - the mechanical spring-mass response,
  - the analog amplifier response,
  - the Arduino oversampling/averaging response,
  - the FIR filter,
  - the detrend filter,
  - the long-period boost filter.
- Produces normalized relative sensitivity curves rather than absolute calibration.

Main outputs:
- `images/sensitivity_model.png`
- `images/sensitivity_modes_comparison.png`

### `plot_fft_vs_model.py`

Purpose:
- Computes FFTs of the same 80 s bridge and bus-loop windows used in the time-domain comparison.
- Overlays those FFTs with:
  - the acquisition-chain model,
  - the full combined sensitivity model.
- Helps show whether the observed dominant periods are better explained by the mechanical resonance or by the acquisition chain.

Main output:
- `images/fft_vs_sensitivity_model.png`

### `plot_bridge_segment_fft.py`

Purpose:
- Uses only the controlled bridge dataset.
- Splits the 80 s bridge window into:
  - a pre-event background segment,
  - a post-event ringing segment.
- Computes FFTs for both segments.
- Overlays them with the acquisition-chain model and the full combined sensitivity model.
- Helps separate filter-shaped background behavior from the actual post-impact mechanical ringing.

Main output:
- `images/bridge_segment_fft.png`

## Generated Build Files

These are created by LaTeX during compilation and usually do not need manual editing:

- `*.aux`
- `*.fdb_latexmk`
- `*.fls`
- `*.log`
- `*.out`
- `*.synctex.gz`

## Suggested Workflow

1. Edit `project.tex` for the main report.
2. If the amplifier schematic changes, rebuild it with `latexmk -pdf -outdir=images amplifier_schematic.tex`.
3. Run the relevant Python script if a plotted figure needs updating.
4. Recompile with `latexmk -pdf project.tex`.
5. If testing modeling ideas separately, edit `sensitivity_model.tex` instead of the main report.
