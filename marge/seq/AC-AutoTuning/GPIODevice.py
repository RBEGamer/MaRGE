#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum

from SerialDevice import SerialDevice


class Pin(str, Enum):
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"
    D5 = "D5"
    D6 = "D6"
    D7 = "D7"
    D8 = "D8"
    D9 = "D9"
    D10 = "D10"
    D11 = "D11"
    D12 = "D12"
    D13 = "D13"
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"


class PinLevel(str, Enum):
    LOW = "LOW"
    HIGH = "HIGH"


class PinDirection(str, Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    INPUT_PULLUP = "INPUT_PULLUP"


PinMap = dict[Pin, PinLevel]
PIN_FIELDS: tuple[tuple[Pin, str], ...] = tuple((pin, pin.name.lower()) for pin in Pin)
PIN_TO_FIELD = {pin: field_name for pin, field_name in PIN_FIELDS}


class ProtocolError(RuntimeError):
    pass


def parse_pin(token: str) -> Pin:
    try:
        return Pin(token.upper())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"unsupported pin: {token}") from exc


def parse_level(token: str) -> PinLevel:
    try:
        return PinLevel(token.upper())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"unsupported level: {token}") from exc


def parse_assignment(token: str) -> tuple[Pin, PinLevel]:
    if "=" not in token:
        raise argparse.ArgumentTypeError(
            f"invalid assignment {token!r}; expected PIN=LEVEL such as D13=HIGH"
        )

    pin_token, level_token = token.split("=", 1)
    return parse_pin(pin_token), parse_level(level_token)


@dataclass(frozen=True)
class GpioState:
    d2: PinLevel | None = None
    d3: PinLevel | None = None
    d4: PinLevel | None = None
    d5: PinLevel | None = None
    d6: PinLevel | None = None
    d7: PinLevel | None = None
    d8: PinLevel | None = None
    d9: PinLevel | None = None
    d10: PinLevel | None = None
    d11: PinLevel | None = None
    d12: PinLevel | None = None
    d13: PinLevel | None = None
    a0: PinLevel | None = None
    a1: PinLevel | None = None
    a2: PinLevel | None = None
    a3: PinLevel | None = None
    a4: PinLevel | None = None
    a5: PinLevel | None = None

    @classmethod
    def empty(cls) -> "GpioState":
        return cls()

    @classmethod
    def all(cls, level: PinLevel) -> "GpioState":
        return cls(**{field_name: level for _, field_name in PIN_FIELDS})

    @classmethod
    def from_pin_map(
        cls,
        pin_map: PinMap,
        available_pins: set[Pin] | None = None,
    ) -> "GpioState":
        expected_pins = available_pins or set(Pin)
        missing_pins = [pin.value for pin in Pin if pin in expected_pins and pin not in pin_map]
        if missing_pins:
            raise ProtocolError("missing pins in state: " + ", ".join(missing_pins))
        values = {
            PIN_TO_FIELD[pin]: pin_map[pin]
            for pin in Pin
            if pin in pin_map
        }
        return cls(**values)

    @classmethod
    def from_assignments(cls, assignments: list[tuple[Pin, PinLevel]]) -> "GpioState":
        values: dict[str, PinLevel] = {}
        for pin, level in assignments:
            field_name = PIN_TO_FIELD[pin]
            existing = values.get(field_name)
            if existing is not None and existing != level:
                raise ProtocolError(
                    f"conflicting assignments for {pin.value}: {existing.value} vs {level.value}"
                )
            values[field_name] = level
        return cls(**values)

    def assigned_pins(self) -> PinMap:
        assigned: PinMap = {}
        for pin, field_name in PIN_FIELDS:
            level = getattr(self, field_name)
            if level is not None:
                assigned[pin] = level
        return assigned

    def level_for(self, pin: Pin) -> PinLevel | None:
        return getattr(self, PIN_TO_FIELD[pin])

    def with_pin(self, pin: Pin, level: PinLevel) -> "GpioState":
        values = {field_name: getattr(self, field_name) for _, field_name in PIN_FIELDS}
        values[PIN_TO_FIELD[pin]] = level
        return GpioState(**values)

    def lines(self, include_unset: bool = False) -> list[str]:
        output: list[str] = []
        for pin, field_name in PIN_FIELDS:
            level = getattr(self, field_name)
            if level is None and not include_unset:
                continue
            rendered = level.value if level is not None else "UNSET"
            output.append(f"{pin.value}={rendered}")
        return output


@dataclass(frozen=True)
class PinInfo:
    pin: Pin
    direction: PinDirection

    @property
    def is_output(self) -> bool:
        return self.direction == PinDirection.OUTPUT


@dataclass(frozen=True)
class GpioCapabilities:
    pins: dict[Pin, PinInfo]

    def available_pins(self) -> set[Pin]:
        return set(self.pins)

    def require_available(self, pin: Pin) -> PinInfo:
        pin_info = self.pins.get(pin)
        if pin_info is None:
            raise ProtocolError(f"{pin.value} is not advertised by the Arduino")
        return pin_info

    def require_output(self, pin: Pin) -> PinInfo:
        pin_info = self.require_available(pin)
        if not pin_info.is_output:
            raise ProtocolError(
                f"{pin.value} is configured as {pin_info.direction.value} and cannot be written"
            )
        return pin_info

    def lines(self) -> list[str]:
        return [f"{pin.value}={self.pins[pin].direction.value}" for pin in Pin if pin in self.pins]


class GPIODevice(SerialDevice):
    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 115200,
        timeout: float = 0.5,
        startup_delay: float = 2.0,
    ) -> None:
        self.connection_string = port
        super().__init__(
            baudrate=baudrate,
            timeout=timeout,
            startup_delay=startup_delay,
        )
        self._capabilities: GpioCapabilities | None = None

    def __enter__(self) -> "GPIODevice":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        port = self.connection_string
        if not self.connect(port):
            raise FileNotFoundError(
                f"failed to connect to Arduino on {port}"
            )
        self._capabilities = self.read_capabilities()

    def close(self) -> None:
        self.disconnect()
        self._capabilities = None

    def set_connection_string(self, port: str) -> None:
        self.connection_string = port

    @property
    def capabilities(self) -> GpioCapabilities:
        if self._capabilities is None:
            self._capabilities = self.read_capabilities()
        return self._capabilities

    def read_capabilities(self) -> GpioCapabilities:
        line = self._send_command("PINS", expected_prefix="PINS ")
        return self._parse_pins_line(line)

    def read_state(self) -> GpioState:
        line = self._send_command("STATUS", expected_prefix="STATUS ")
        return self._parse_status_line(line)

    def set_pin(self, pin: Pin, level: PinLevel, verify: bool = True) -> GpioState | None:
        self.capabilities.require_output(pin)
        return self.sync(GpioState.empty().with_pin(pin, level), verify=verify)

    def set_all(self, level: PinLevel, verify: bool = True) -> GpioState | None:
        for pin in self.capabilities.available_pins():
            self.capabilities.require_output(pin)
        self._send_command(f"ALL {level.value}", expected_exact="OK")
        if not verify:
            return None
        full_state = GpioState(
            **{
                PIN_TO_FIELD[pin]: level
                for pin in self.capabilities.available_pins()
            }
        )
        return self._verify_requested_state(full_state)

    def sync(self, desired_state: GpioState, verify: bool = True) -> GpioState | None:
        desired_updates = desired_state.assigned_pins()
        current_state = self.read_state()

        if not desired_updates:
            return current_state if verify else None

        for pin in desired_updates:
            self.capabilities.require_output(pin)

        current_levels = current_state.assigned_pins()
        for pin, level in desired_updates.items():
            if current_levels[pin] == level:
                continue
            self._send_command(f"SET {pin.value} {level.value}", expected_exact="OK")

        if not verify:
            return None
        return self._verify_requested_state(desired_state)

    def execute_cli(self, args: argparse.Namespace) -> int:
        if args.command == "pins":
            print_capabilities(self.capabilities)
            return 0

        if args.command == "status":
            print_state(self.read_state())
            return 0

        if args.command == "set":
            status = self.set_pin(args.pin, args.level, verify=not args.no_verify)
            if status is not None:
                print_state(status)
            return 0

        if args.command == "all":
            status = self.set_all(args.level, verify=not args.no_verify)
            if status is not None:
                print_state(status)
            return 0

        if args.command == "sync":
            desired_state = GpioState.from_assignments(args.assignments)
            status = self.sync(desired_state, verify=not args.no_verify)
            if status is not None:
                print_state(status)
            return 0

        raise ProtocolError(f"unsupported command: {args.command}")

    def _verify_requested_state(self, desired_state: GpioState) -> GpioState:
        actual_state = self.read_state()
        mismatches = []
        for pin, expected_level in desired_state.assigned_pins().items():
            actual_level = actual_state.level_for(pin)
            if actual_level != expected_level:
                mismatches.append(
                    f"{pin.value}: expected {expected_level.value}, got {actual_level.value}"
                )
        if mismatches:
            raise ProtocolError("verification failed: " + "; ".join(mismatches))
        return actual_state

    def _parse_status_line(self, line: str) -> GpioState:
        tokens = line.split()
        if not tokens or tokens[0] != "STATUS":
            raise ProtocolError(f"unexpected status line: {line}")

        pin_map: PinMap = {}
        for token in tokens[1:]:
            if "=" not in token:
                raise ProtocolError(f"malformed status token: {token}")
            pin_token, level_token = token.split("=", 1)
            try:
                pin = Pin(pin_token)
                level = PinLevel(level_token)
            except ValueError as exc:
                raise ProtocolError(f"unsupported status token: {token}") from exc
            if pin in pin_map:
                raise ProtocolError(f"duplicate status entry for {pin.value}")
            pin_map[pin] = level

        capabilities = self.capabilities
        advertised_pins = capabilities.available_pins()
        if any(pin not in advertised_pins for pin in pin_map):
            unexpected = [pin.value for pin in pin_map if pin not in advertised_pins]
            raise ProtocolError("unexpected pins in status: " + ", ".join(unexpected))

        if len(pin_map) != len(advertised_pins):
            missing_pins = [pin.value for pin in advertised_pins if pin not in pin_map]
            raise ProtocolError("missing pins in status: " + ", ".join(missing_pins))

        return GpioState.from_pin_map(pin_map, available_pins=advertised_pins)

    @staticmethod
    def _parse_pins_line(line: str) -> GpioCapabilities:
        tokens = line.split()
        if not tokens or tokens[0] != "PINS":
            raise ProtocolError(f"unexpected pins line: {line}")

        pin_info: dict[Pin, PinInfo] = {}
        for token in tokens[1:]:
            if "=" not in token:
                raise ProtocolError(f"malformed pin token: {token}")
            pin_token, direction_token = token.split("=", 1)
            try:
                pin = Pin(pin_token)
                direction = PinDirection(direction_token)
            except ValueError as exc:
                raise ProtocolError(f"unsupported pin token: {token}") from exc
            if pin in pin_info:
                raise ProtocolError(f"duplicate pin entry for {pin.value}")
            pin_info[pin] = PinInfo(pin=pin, direction=direction)

        if not pin_info:
            raise ProtocolError("arduino reported no gpio metadata")

        return GpioCapabilities(pin_info)

    def _send_command(
        self,
        command: str,
        *,
        expected_exact: str | None = None,
        expected_prefix: str | None = None,
        deadline_seconds: float = 5.0,
    ) -> str:
        if self.device is None:
            raise RuntimeError("serial port is not open")

        self.reset_buffers()
        self.write_line(command)

        while True:
            raw = self.read_line(deadline_seconds=deadline_seconds)
            if raw is False:
                raise TimeoutError(f"timed out waiting for response to {command!r}")

            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            if line.startswith("Commands:") or line.startswith("UNO GPIO"):
                continue
            if line.startswith("PINS"):
                if expected_prefix == "PINS " and line.startswith(expected_prefix):
                    return line
                continue
            if line.startswith("SET ") or line.startswith("ALL ") or line == "STATUS":
                continue
            if line.startswith("ERR "):
                raise ProtocolError(line)
            if expected_exact is not None and line == expected_exact:
                return line
            if expected_prefix is not None and line.startswith(expected_prefix):
                return line


ArduinoGpioClient = GPIODevice




def print_state(state: GpioState, include_unset: bool = False) -> None:
    for line in state.lines(include_unset=include_unset):
        print(line)


def print_capabilities(capabilities: GpioCapabilities) -> None:
    for line in capabilities.lines():
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Control Arduino Uno GPIO pins over the serial controller protocol."
    )
    parser.add_argument("--port", default=None, help="Serial device path")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baud rate")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip reading STATUS after write commands",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pins", help="Read and print the GPIO metadata advertised by the Arduino")
    subparsers.add_parser("status", help="Read and print all managed pin states")

    set_parser = subparsers.add_parser("set", help="Set one pin to HIGH or LOW")
    set_parser.add_argument("pin", type=parse_pin)
    set_parser.add_argument("level", type=parse_level)

    all_parser = subparsers.add_parser("all", help="Set all managed pins to the same level")
    all_parser.add_argument("level", type=parse_level)

    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync one or more PIN=LEVEL assignments to the Arduino",
    )
    sync_parser.add_argument(
        "assignments",
        nargs="+",
        type=parse_assignment,
        help="One or more assignments such as D13=HIGH or A0=LOW",
    )

    arguments = parser.parse_args()
    device = GPIODevice(
        port=arguments.port,
        baudrate=arguments.baudrate,
    )

    try:
        with device:
            raise SystemExit(device.execute_cli(arguments))
    except (FileNotFoundError, OSError, ProtocolError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
