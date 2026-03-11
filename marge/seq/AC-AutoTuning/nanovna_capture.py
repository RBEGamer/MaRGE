#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pynanovna import VNA, get_interfaces


def resolve_vna_index(index: int | None, port: str | None) -> int:
    interfaces = get_interfaces()
    if not interfaces:
        raise RuntimeError("No NanoVNA detected")

    if port:
        resolved_port = str(Path(port).resolve())
        for candidate_index, iface in enumerate(interfaces):
            iface_port = getattr(iface, "port", None) or getattr(iface, "name", None)
            if iface_port and Path(iface_port).resolve() == Path(resolved_port):
                return candidate_index
        raise RuntimeError(f"No NanoVNA interface matched port {port}")

    if index is not None:
        if index < 0 or index >= len(interfaces):
            raise RuntimeError(f"NanoVNA index {index} is out of range; found {len(interfaces)} device(s)")
        return index

    return 0


def validate_sweep(start_hz: int, stop_hz: int, points: int) -> None:
    if start_hz <= 0 or stop_hz <= 0:
        raise ValueError("Frequencies must be positive")
    if start_hz >= stop_hz:
        raise ValueError("Start frequency must be smaller than stop frequency")
    if points < 2:
        raise ValueError("At least 2 points are required")


@dataclass(frozen=True)
class CaptureResult:
    info: dict[str, object]
    frequencies_hz: list[int]
    s11: list[complex]

    @property
    def interface_name(self) -> str:
        interface = self.info.get("Interface")
        return str(interface) if interface is not None else "unknown"

    def magnitude_db(self) -> list[float]:
        return [20.0 * math.log10(max(abs(gamma), 1e-12)) for gamma in self.s11]

    def frequencies_mhz(self) -> list[float]:
        return [value / 1e6 for value in self.frequencies_hz]


class NanoVNACapture:
    def __init__(
        self,
        *,
        port: str | None = None,
        index: int | None = None,
        logging_level: str = "critical",
    ) -> None:
        self.port = port
        self.index = index
        self.logging_level = logging_level

    def capture(self, start_hz: int, stop_hz: int, points: int = 101) -> CaptureResult:
        validate_sweep(start_hz, stop_hz, points)
        vna_index = resolve_vna_index(self.index, self.port)

        previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        vna = VNA(vna_index=vna_index, logging_level=self.logging_level)
        if not vna.connected:
            logging.disable(previous_disable)
            raise RuntimeError("Failed to connect to the NanoVNA")

        try:
            info = dict(vna.info())
            vna.set_sweep(start_hz, stop_hz, points)
            s11, _s21, frequencies_hz = vna.sweep()
        finally:
            vna.kill()
            logging.disable(previous_disable)

        normalized_frequencies_hz = [int(value) for value in frequencies_hz]
        normalized_s11 = [complex(value) for value in s11]

        if len(normalized_frequencies_hz) != len(normalized_s11):
            raise RuntimeError(
                f"Point mismatch: got {len(normalized_frequencies_hz)} frequencies "
                f"and {len(normalized_s11)} S11 samples"
            )

        if len(normalized_frequencies_hz) != points:
            raise RuntimeError(f"Expected {points} points, got {len(normalized_frequencies_hz)}")

        return CaptureResult(
            info=info,
            frequencies_hz=normalized_frequencies_hz,
            s11=normalized_s11,
        )

    def capture_to_files(
        self,
        *,
        start_hz: int,
        stop_hz: int,
        points: int = 101,
        csv_path: str | Path | None = None,
        plot_path: str | Path | None = None,
    ) -> CaptureResult:
        result = self.capture(start_hz=start_hz, stop_hz=stop_hz, points=points)
        if csv_path is not None:
            self.save_csv(csv_path, result)
        if plot_path is not None:
            self.save_plot(plot_path, result)
        return result

    @staticmethod
    def save_csv(path: str | Path, result: CaptureResult) -> Path:
        csv_path = Path(path)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frequency_hz", "s11_real", "s11_imag", "s11_mag_db"])
            for frequency_hz, gamma, magnitude_db in zip(
                result.frequencies_hz,
                result.s11,
                result.magnitude_db(),
            ):
                writer.writerow([frequency_hz, gamma.real, gamma.imag, magnitude_db])
        return csv_path

    @staticmethod
    def draw_smith_background(ax) -> None:
        ax.add_patch(Circle((0.0, 0.0), 1.0, fill=False, color="0.3", linewidth=1.2))
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.axvline(0.0, color="0.75", linewidth=0.8)
        for radius in (0.25, 0.5, 0.75):
            ax.add_patch(Circle((0.0, 0.0), radius, fill=False, color="0.9", linewidth=0.5))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Real(S11)")
        ax.set_ylabel("Imag(S11)")
        ax.set_title("Smith Chart")
        ax.grid(False)

    @classmethod
    def save_plot(cls, path: str | Path, result: CaptureResult) -> Path:
        plot_path = Path(path)
        magnitudes_db = result.magnitude_db()
        reals = [gamma.real for gamma in result.s11]
        imags = [gamma.imag for gamma in result.s11]

        fig, (ax_mag, ax_smith) = plt.subplots(1, 2, figsize=(12, 5.5))

        ax_mag.plot(result.frequencies_mhz(), magnitudes_db, color="#1f77b4", linewidth=1.5)
        ax_mag.set_title("S11 Magnitude")
        ax_mag.set_xlabel("Frequency (MHz)")
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.grid(True, alpha=0.35)

        cls.draw_smith_background(ax_smith)
        ax_smith.plot(reals, imags, color="#d62728", linewidth=1.2)
        ax_smith.scatter(reals[0], imags[0], color="#2ca02c", s=25, label="Start")
        ax_smith.scatter(reals[-1], imags[-1], color="#9467bd", s=25, label="Stop")
        ax_smith.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture S11 from a connected NanoVNA-H4")
    parser.add_argument("--port", default=None, help="Serial port for the NanoVNA")
    parser.add_argument("--index", type=int, default=None, help="NanoVNA index when multiple devices exist")
    parser.add_argument("--start", type=int, required=True, help="Start frequency in Hz")
    parser.add_argument("--stop", type=int, required=True, help="Stop frequency in Hz")
    parser.add_argument("--points", type=int, default=101, help="Number of sweep points")
    parser.add_argument("--csv", default="nanovna_capture.csv", help="CSV output path")
    parser.add_argument("--plot", default="nanovna_capture.png", help="Plot output path")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = NanoVNACapture(port=args.port, index=args.index)
    result = capture.capture_to_files(
        start_hz=args.start,
        stop_hz=args.stop,
        points=args.points,
        csv_path=args.csv,
        plot_path=args.plot,
    )

    print(f"Connected to: {result.interface_name}")
    for key, value in result.info.items():
        print(f"{key}: {value}")
    print(f"Saved CSV: {Path(args.csv)}")
    print(f"Saved plot: {Path(args.plot)}")

    if args.show:
        image = plt.imread(args.plot)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.imshow(image)
        ax.axis("off")
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
