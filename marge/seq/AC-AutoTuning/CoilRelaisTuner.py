#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from GPIODevice import GPIODevice, GpioState, Pin, PinLevel
from nanovna_capture import CaptureResult, NanoVNACapture


ARDUINO_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
NANOVNA_PORT = "/dev/serial/by-id/usb-nanovna.com_NanoVNA-H4_400-if00"
NANOVNA_INDEX = 0

OUT_DIR = Path("tm_scan")
F_START_HZ = 1_000_000
F_STOP_HZ = 20_000_000
N_POINTS = 401
F0_HZ: int | None = None
SETTLE_S = 0.02
UPDATE_PROGRESS_PLOT = True
PROGRESS_PLOT_FILENAME = "scan_progress.png"
PROGRESS_PLOT_EVERY = 1
SCAN_STATE_FILENAME = "scan_state.json"
REFERENCE_IMPEDANCE_OHM = 50.0

ROUTE_TO_VNA_PIN = Pin.A5
MATCH_CAPS: tuple[Pin, ...] = (Pin.D10, Pin.D11, Pin.D12, Pin.D13, Pin.A4)
TUNE_CAPS: tuple[Pin, ...] = (Pin.D5, Pin.D6, Pin.D7, Pin.D8, Pin.D9)
SERIES_CAPS: tuple[Pin, ...] = (Pin.A1, Pin.A2, Pin.D2, Pin.D3, Pin.D4)
SCAN_PINS: tuple[Pin, ...] = MATCH_CAPS + TUNE_CAPS + SERIES_CAPS


@dataclass(frozen=True)
class SwitchConfig:
    match_mask: int
    tune_mask: int
    series_mask: int

    def active_set(
        self,
        match_caps: Sequence[Pin],
        tune_caps: Sequence[Pin],
        series_caps: Sequence[Pin],
    ) -> set[Pin]:
        active: set[Pin] = set()

        for index, pin in enumerate(match_caps):
            if (self.match_mask >> index) & 1:
                active.add(pin)

        for index, pin in enumerate(tune_caps):
            if (self.tune_mask >> index) & 1:
                active.add(pin)

        for index, pin in enumerate(series_caps):
            if (self.series_mask >> index) & 1:
                active.add(pin)

        return active


class FileScanPlot:
    def __init__(self, *, enabled: bool, path: Path, save_every: int = 1) -> None:
        self.enabled = enabled
        self.path = path
        self.save_every = max(1, save_every)
        self.best_min_s11_db: float | None = None
        self.steps: list[int] = []
        self.min_history_db: list[float] = []
        self.best_history_db: list[float] = []
        self.best_freqs_mhz: list[float] = []
        self.best_magnitudes_db: list[float] = []
        self.best_config: SwitchConfig | None = None
        self.best_impedance: complex | None = None
        self.best_match_freq_mhz: float | None = None

        if not self.enabled:
            return

        self.figure, axes = plt.subplots(2, 1, figsize=(11, 8))
        self.sweep_axis, self.progress_axis = axes
        self.current_line, = self.sweep_axis.plot([], [], color="#1f77b4", linewidth=1.0, label="Current")
        self.best_line, = self.sweep_axis.plot([], [], color="#d62728", linewidth=1.4, label="Best")
        self.min_line, = self.progress_axis.plot([], [], color="#1f77b4", linewidth=1.0, label="Current min")
        self.best_progress_line, = self.progress_axis.plot([], [], color="#2ca02c", linewidth=1.2, label="Best so far")
        self.current_match_marker, = self.sweep_axis.plot([], [], "o", color="#1f77b4", markersize=5)
        self.best_match_marker, = self.sweep_axis.plot([], [], "o", color="#d62728", markersize=6)
        self.match_text = self.sweep_axis.text(
            0.02,
            0.02,
            "",
            transform=self.sweep_axis.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )

        self.sweep_axis.set_title("S11 Magnitude During Scan")
        self.sweep_axis.set_xlabel("Frequency (MHz)")
        self.sweep_axis.set_ylabel("Magnitude (dB)")
        self.sweep_axis.grid(True, alpha=0.35)
        self.sweep_axis.legend(loc="best")

        self.progress_axis.set_title("Running Best Min S11")
        self.progress_axis.set_xlabel("Configuration Index")
        self.progress_axis.set_ylabel("Min S11 (dB)")
        self.progress_axis.grid(True, alpha=0.35)
        self.progress_axis.legend(loc="best")

        self.figure.tight_layout()

    def update(
        self,
        *,
        step: int,
        total_steps: int,
        config: SwitchConfig,
        freqs_hz: Sequence[int],
        s11: Sequence[complex],
        magnitudes_db: Sequence[float],
        min_s11_db: float,
    ) -> None:
        if not self.enabled:
            return

        freqs_mhz = [frequency / 1e6 for frequency in freqs_hz]
        self.current_line.set_data(freqs_mhz, magnitudes_db)
        current_match_index, current_match_freq_mhz, current_impedance = best_match_summary(freqs_hz, s11)
        self.current_match_marker.set_data(
            [freqs_mhz[current_match_index]],
            [magnitudes_db[current_match_index]],
        )

        if self.best_min_s11_db is None or min_s11_db < self.best_min_s11_db:
            self.best_min_s11_db = min_s11_db
            self.best_freqs_mhz = list(freqs_mhz)
            self.best_magnitudes_db = list(magnitudes_db)
            self.best_config = config
            self.best_impedance = current_impedance
            self.best_match_freq_mhz = current_match_freq_mhz
            self.best_line.set_data(self.best_freqs_mhz, self.best_magnitudes_db)
            self.best_match_marker.set_data(
                [freqs_mhz[current_match_index]],
                [magnitudes_db[current_match_index]],
            )

        self.steps.append(step)
        self.min_history_db.append(min_s11_db)
        self.best_history_db.append(
            self.best_min_s11_db if self.best_min_s11_db is not None else min_s11_db
        )

        self.min_line.set_data(self.steps, self.min_history_db)
        self.best_progress_line.set_data(self.steps, self.best_history_db)

        self.sweep_axis.set_title(
            "S11 Magnitude During Scan "
            f"({step}/{total_steps}) mm={config.match_mask:02x} tm={config.tune_mask:02x} sm={config.series_mask:02x}"
        )
        if self.best_min_s11_db is not None:
            best_title = f"{self.best_min_s11_db:.2f} dB"
            if self.best_config is not None:
                best_title += (
                    f" at mm={self.best_config.match_mask:02x}"
                    f" tm={self.best_config.tune_mask:02x}"
                    f" sm={self.best_config.series_mask:02x}"
                )
            self.progress_axis.set_title(
                f"Running Best Min S11: {best_title}"
            )
        match_lines = [
            (
                f"Current match: {current_impedance.real:.1f}"
                f" + j{current_impedance.imag:.1f} ohm @ {current_match_freq_mhz:.3f} MHz"
            )
        ]
        if self.best_impedance is not None and self.best_match_freq_mhz is not None:
            match_lines.append(
                (
                    f"Best match: {self.best_impedance.real:.1f}"
                    f" + j{self.best_impedance.imag:.1f} ohm @ {self.best_match_freq_mhz:.3f} MHz"
                )
            )
        match_lines.append(f"Reference: {REFERENCE_IMPEDANCE_OHM:.0f} ohm")
        self.match_text.set_text("\n".join(match_lines))

        self.sweep_axis.relim()
        self.sweep_axis.autoscale_view()
        self.progress_axis.relim()
        self.progress_axis.autoscale_view()

        if step == 1 or step % self.save_every == 0 or step == total_steps:
            self.save()

    def save(self) -> None:
        if not self.enabled:
            return
        self.figure.tight_layout()
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self.figure.savefig(temp_path, dpi=150, format="png")
        temp_path.replace(self.path)

    def finish(self) -> None:
        if not self.enabled:
            return
        self.save()
        plt.close(self.figure)


def gpio_state_from_levels(pin_levels: dict[Pin, PinLevel]) -> GpioState:
    return GpioState(**{pin.name.lower(): level for pin, level in pin_levels.items()})


def route_coil_to_vna() -> GpioState:
    return gpio_state_from_levels({ROUTE_TO_VNA_PIN: PinLevel.HIGH})


def managed_state_subset(state: GpioState) -> GpioState:
    pin_levels = {
        pin: level
        for pin in (ROUTE_TO_VNA_PIN,) + SCAN_PINS
        if (level := state.level_for(pin)) is not None
    }
    return gpio_state_from_levels(pin_levels)


def iter_configs(
    n_match: int,
    n_tune: int,
    n_series: int,
    require_match_nonzero: bool = True,
    require_series_nonzero: bool = True,
) -> Iterable[SwitchConfig]:
    for match_mask in range(1 << n_match):
        if require_match_nonzero and match_mask == 0:
            continue
        for tune_mask in range(1 << n_tune):
            for series_mask in range(1 << n_series):
                if require_series_nonzero and series_mask == 0:
                    continue
                yield SwitchConfig(
                    match_mask=match_mask,
                    tune_mask=tune_mask,
                    series_mask=series_mask,
                )


def count_configs(
    n_match: int,
    n_tune: int,
    n_series: int,
    require_match_nonzero: bool = True,
    require_series_nonzero: bool = True,
) -> int:
    match_count = (1 << n_match) - int(require_match_nonzero)
    tune_count = 1 << n_tune
    series_count = (1 << n_series) - int(require_series_nonzero)
    return match_count * tune_count * series_count


def nearest_index(values: Sequence[int], target: int) -> int:
    best_index = 0
    best_delta = abs(values[0] - target)
    for index in range(1, len(values)):
        delta = abs(values[index] - target)
        if delta < best_delta:
            best_delta = delta
            best_index = index
    return best_index


def magnitude_db_from_s11(s11: Sequence[complex]) -> list[float]:
    return [20.0 * math.log10(max(abs(gamma), 1e-12)) for gamma in s11]


def estimate_fres_hz(freqs_hz: Sequence[int], s11: Sequence[complex]) -> int:
    magnitudes = [abs(gamma) for gamma in s11]
    minimum_index = min(range(len(magnitudes)), key=magnitudes.__getitem__)
    return int(freqs_hz[minimum_index])


def impedance_from_gamma(gamma: complex, z0_ohm: float = REFERENCE_IMPEDANCE_OHM) -> complex:
    denominator = 1.0 - gamma
    if abs(denominator) < 1e-12:
        return complex(float("inf"), float("inf"))
    return z0_ohm * (1.0 + gamma) / denominator


def best_match_summary(
    freqs_hz: Sequence[int],
    s11: Sequence[complex],
    z0_ohm: float = REFERENCE_IMPEDANCE_OHM,
) -> tuple[int, float, complex]:
    magnitudes = [abs(gamma) for gamma in s11]
    best_index = min(range(len(magnitudes)), key=magnitudes.__getitem__)
    return best_index, freqs_hz[best_index] / 1e6, impedance_from_gamma(s11[best_index], z0_ohm)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    return str(value)


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def write_scan_state(
    *,
    path: Path,
    status: str,
    results_path: Path,
    progress_plot_path: Path,
    total_configs: int,
    completed_configs: int,
    remaining_configs: int,
    last_completed: tuple[int, int, int] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "resume_supported": True,
        "updated_at_epoch_s": time.time(),
        "total_configs": total_configs,
        "completed_configs": completed_configs,
        "remaining_configs": remaining_configs,
        "results_csv": str(results_path),
        "progress_plot": str(progress_plot_path),
        "resume_hint": f"Re-run python3 {Path(__file__).name} to continue from the existing results.",
    }
    if last_completed is not None:
        payload["last_completed"] = {
            "match_mask": last_completed[0],
            "tune_mask": last_completed[1],
            "series_mask": last_completed[2],
        }
    if error is not None:
        payload["error"] = error
    write_json_file(path, payload)


def apply_switch_state(relais: GPIODevice, active: set[Pin]) -> GpioState:
    pin_levels = {
        pin: PinLevel.HIGH if pin in active else PinLevel.LOW
        for pin in SCAN_PINS
    }
    return relais.sync(gpio_state_from_levels(pin_levels))


def capture_once(
    nanovna: NanoVNACapture,
    f_start_hz: int,
    f_stop_hz: int,
    n_points: int,
) -> CaptureResult:
    return nanovna.capture(start_hz=f_start_hz, stop_hz=f_stop_hz, points=n_points)


def scan_all_configs(
    *,
    relais: GPIODevice,
    nanovna: NanoVNACapture,
    out_dir: Path,
    match_caps: Sequence[Pin],
    tune_caps: Sequence[Pin],
    series_caps: Sequence[Pin],
    f_start_hz: int,
    f_stop_hz: int,
    n_points: int,
    f0_hz: Optional[int] = None,
    require_match_nonzero: bool = True,
    require_series_nonzero: bool = True,
    settle_s: float = 0.02,
    resume: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sweeps_dir = out_dir / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.csv"
    progress_plot_path = out_dir / PROGRESS_PLOT_FILENAME
    scan_state_path = out_dir / SCAN_STATE_FILENAME
    done_set: set[tuple[int, int, int]] = set()
    last_completed: tuple[int, int, int] | None = None

    if resume and results_path.exists():
        with results_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                last_completed = (
                    int(row["match_mask"]),
                    int(row["tune_mask"]),
                    int(row["series_mask"]),
                )
                done_set.add(last_completed)

    total_configs = count_configs(
        len(match_caps),
        len(tune_caps),
        len(series_caps),
        require_match_nonzero=require_match_nonzero,
        require_series_nonzero=require_series_nonzero,
    )
    total_remaining = total_configs - len(done_set)
    if done_set:
        print(
            f"Resume mode: found {len(done_set)} completed configurations in {results_path}; "
            f"continuing with {total_remaining} remaining"
        )
    else:
        print(f"Scanning {total_remaining} configurations into {out_dir}")
    plotter = FileScanPlot(
        enabled=UPDATE_PROGRESS_PLOT,
        path=progress_plot_path,
        save_every=PROGRESS_PLOT_EVERY,
    )
    if UPDATE_PROGRESS_PLOT:
        print(f"Updating progress plot at {progress_plot_path}")

    completed_count = len(done_set)
    write_scan_state(
        path=scan_state_path,
        status="running",
        results_path=results_path,
        progress_plot_path=progress_plot_path,
        total_configs=total_configs,
        completed_configs=completed_count,
        remaining_configs=total_remaining,
        last_completed=last_completed,
    )

    try:
        write_header = not results_path.exists()
        with results_path.open("a", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "match_mask",
                "tune_mask",
                "series_mask",
                "active_count_match",
                "active_count_tune",
                "active_count_series",
                "min_s11_db",
                "f_res_hz",
                "s11_db_at_f0",
                "f0_hz",
                "sweep_file",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for config in iter_configs(
                n_match=len(match_caps),
                n_tune=len(tune_caps),
                n_series=len(series_caps),
                require_match_nonzero=require_match_nonzero,
                require_series_nonzero=require_series_nonzero,
            ):
                key = (config.match_mask, config.tune_mask, config.series_mask)
                if resume and key in done_set:
                    continue

                active = config.active_set(match_caps, tune_caps, series_caps)
                actual_state = apply_switch_state(relais, active)
                time.sleep(settle_s)

                capture = capture_once(
                    nanovna,
                    f_start_hz=f_start_hz,
                    f_stop_hz=f_stop_hz,
                    n_points=n_points,
                )

                freqs = list(capture.frequencies_hz)
                s11 = list(capture.s11)
                magnitudes_db = magnitude_db_from_s11(s11)
                min_s11_db = float(min(magnitudes_db))
                f_res_hz = estimate_fres_hz(freqs, s11)
                plotter.update(
                    step=completed_count + 1,
                    total_steps=total_configs,
                    config=config,
                    freqs_hz=freqs,
                    s11=s11,
                    magnitudes_db=magnitudes_db,
                    min_s11_db=min_s11_db,
                )

                s11_db_at_f0: float | None = None
                if f0_hz is not None:
                    target_index = nearest_index(freqs, f0_hz)
                    s11_db_at_f0 = float(magnitudes_db[target_index])

                sweep_file = sweeps_dir / (
                    f"mm{config.match_mask:02x}_tm{config.tune_mask:02x}_sm{config.series_mask:02x}.json"
                )
                sweep_payload = {
                    "match_mask": config.match_mask,
                    "tune_mask": config.tune_mask,
                    "series_mask": config.series_mask,
                    "active_switches": [pin.value for pin in sorted(active, key=lambda pin: pin.value)],
                    "gpio_state": actual_state.lines(),
                    "capture_info": json_safe(capture.info),
                    "frequencies_hz": freqs,
                    "s11_re_im": [[float(value.real), float(value.imag)] for value in s11],
                }
                sweep_file.write_text(json.dumps(sweep_payload, indent=2), encoding="utf-8")

                writer.writerow(
                {
                    "match_mask": config.match_mask,
                    "tune_mask": config.tune_mask,
                    "series_mask": config.series_mask,
                        "active_count_match": bin(config.match_mask).count("1"),
                        "active_count_tune": bin(config.tune_mask).count("1"),
                        "active_count_series": bin(config.series_mask).count("1"),
                        "min_s11_db": min_s11_db,
                        "f_res_hz": f_res_hz,
                        "s11_db_at_f0": "" if s11_db_at_f0 is None else s11_db_at_f0,
                        "f0_hz": "" if f0_hz is None else f0_hz,
                        "sweep_file": str(sweep_file.relative_to(out_dir)),
                    }
                )
                handle.flush()
                last_completed = key
                completed_count += 1
                write_scan_state(
                    path=scan_state_path,
                    status="running",
                    results_path=results_path,
                    progress_plot_path=progress_plot_path,
                    total_configs=total_configs,
                    completed_configs=completed_count,
                    remaining_configs=total_configs - completed_count,
                    last_completed=last_completed,
                )
    except KeyboardInterrupt:
        write_scan_state(
            path=scan_state_path,
            status="interrupted",
            results_path=results_path,
            progress_plot_path=progress_plot_path,
            total_configs=total_configs,
            completed_configs=completed_count,
            remaining_configs=total_configs - completed_count,
            last_completed=last_completed,
            error="Interrupted by user",
        )
        print("Scan interrupted. Re-run the script to resume from the existing results.")
        raise
    except Exception as exc:
        write_scan_state(
            path=scan_state_path,
            status="failed",
            results_path=results_path,
            progress_plot_path=progress_plot_path,
            total_configs=total_configs,
            completed_configs=completed_count,
            remaining_configs=total_configs - completed_count,
            last_completed=last_completed,
            error=str(exc),
        )
        print("Scan failed. Re-run the script to resume from the existing results.")
        raise
    else:
        write_scan_state(
            path=scan_state_path,
            status="completed",
            results_path=results_path,
            progress_plot_path=progress_plot_path,
            total_configs=total_configs,
            completed_configs=completed_count,
            remaining_configs=0,
            last_completed=last_completed,
        )
    finally:
        plotter.finish()


def main() -> None:
    relais = GPIODevice(port=ARDUINO_PORT, baudrate=115200)
    nanovna = NanoVNACapture(port=NANOVNA_PORT, index=NANOVNA_INDEX)
    initial_gpio_state: GpioState | None = None
    opened = False

    try:
        relais.open()
        opened = True
        initial_gpio_state = managed_state_subset(relais.read_state())
        relais.sync(route_coil_to_vna())

        scan_all_configs(
            relais=relais,
            nanovna=nanovna,
            out_dir=OUT_DIR,
            match_caps=MATCH_CAPS,
            tune_caps=TUNE_CAPS,
            series_caps=SERIES_CAPS,
            f_start_hz=F_START_HZ,
            f_stop_hz=F_STOP_HZ,
            n_points=N_POINTS,
            f0_hz=F0_HZ,
            require_match_nonzero=True,
            require_series_nonzero=False,
            settle_s=SETTLE_S,
            resume=True,
        )
    finally:
        if opened:
            try:
                if initial_gpio_state is not None:
                    relais.sync(initial_gpio_state)
            finally:
                relais.close()


if __name__ == "__main__":
    main()
