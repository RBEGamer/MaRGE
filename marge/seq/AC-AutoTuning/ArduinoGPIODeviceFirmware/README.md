# Arduino Uno GPIO Serial Controller

This PlatformIO project targets an Arduino Uno and exposes simple serial
commands for controlling the usable GPIO pins:

- Digital pins `D2` through `D13`
- Analog pins `A0` through `A5` as digital outputs

Pins `D0` and `D1` are intentionally excluded because they are used by the USB
serial interface.

## Commands

Send newline-terminated commands at `115200` baud:

```text
SET D13 HIGH
SET D13 LOW
SET A0 HIGH
ALL HIGH
ALL LOW
PINS
STATUS
HELP
```

`STATUS` returns a single machine-readable line containing every managed pin:

```text
STATUS D2=LOW D3=LOW D4=LOW D5=LOW D6=LOW D7=LOW D8=LOW D9=LOW D10=LOW D11=LOW D12=LOW D13=LOW A0=LOW A1=LOW A2=LOW A3=LOW A4=LOW A5=LOW
```

`PINS` returns the GPIOs advertised by the Arduino and their direction:

```text
PINS D2=OUTPUT D3=OUTPUT D4=OUTPUT D5=OUTPUT D6=OUTPUT D7=OUTPUT D8=OUTPUT D9=OUTPUT D10=OUTPUT D11=OUTPUT D12=OUTPUT D13=OUTPUT A0=OUTPUT A1=OUTPUT A2=OUTPUT A3=OUTPUT A4=OUTPUT A5=OUTPUT
```

## Build and upload

```bash
python3 -m venv .venv
.venv/bin/pip install platformio
.venv/bin/pio run
./flash_arduino.sh /dev/ttyUSB0
```

If you omit the port, `flash_arduino.sh` tries `/dev/ttyUSB0` first and then
`/dev/ttyACM0`.

## Serial monitor

```bash
.venv/bin/pio device monitor --baud 115200
```

## Python GPIO client

Install the serial dependency:

```bash
.venv/bin/pip install -r requirements-arduino.txt
```

Read the current state of every managed pin:

```bash
.venv/bin/python GPIODevice.py --port /dev/ttyUSB0 status
```

Read the GPIO metadata advertised by the Arduino:

```bash
.venv/bin/python GPIODevice.py --port /dev/ttyUSB0 pins
```

Set a single pin and verify it with a follow-up `STATUS` read:

```bash
.venv/bin/python GPIODevice.py --port /dev/ttyUSB0 set D13 HIGH
```

Set every managed pin and verify the full snapshot:

```bash
.venv/bin/python GPIODevice.py --port /dev/ttyUSB0 all LOW
```

Sync a partial target state and verify the requested pins against the returned
snapshot:

```bash
.venv/bin/python GPIODevice.py --port /dev/ttyUSB0 sync D13=HIGH A0=LOW A1=HIGH
```

The Python API is centered around `GPIODevice` and a `GpioState` dataclass where
`None` means "leave this pin unchanged":

```python
from GPIODevice import GPIODevice, GpioState, PinLevel

target = GpioState(d13=PinLevel.HIGH, a0=PinLevel.LOW)

with GPIODevice(port="/dev/ttyUSB0") as gpio:
    print(gpio.capabilities.lines())
    confirmed = gpio.sync(target)
    print(confirmed.d13)
```

`GPIODevice` accepts the same connection string styles as `SerialDevice`, for
example `/dev/ttyUSB0`, `COM3`, `serial:55731323736351611260`,
`socket://192.168.1.100:5000`, or `rfc2217://host:port`.

The Python client validates writes against the board metadata before sending
them, so it will reject pins that are not advertised by the Arduino or that are
reported as inputs.

`GPIODevice` inherits from `SerialDevice`, so the GPIO protocol methods live on
the serial device itself instead of wrapping a separate transport object.

## NanoVNA capture tool

The Python capture tool uses the `pynanovna` library, performs a sweep, saves the
measured S11 data to CSV, and generates a PNG containing:

- S11 magnitude in dB over frequency
- A Smith-chart style plot of the complex reflection coefficient

Install the NanoVNA dependencies:

```bash
.venv/bin/pip install -r requirements-nanovna.txt
```

Example:

```bash
.venv/bin/python nanovna_capture.py \
  --start 1000000 \
  --stop 30000000 \
  --points 101
```

Optional arguments:

```text
--port /dev/serial/by-id/usb-nanovna.com_NanoVNA-H4_400-if00
--index 0
--csv nanovna_capture.csv
--plot nanovna_capture.png
--show
```

If multiple NanoVNAs are connected, `pynanovna` selects by device index. You can
still pass `--port` and the script will resolve that port to the matching
`pynanovna` device index.

The CLI is now backed by a reusable `NanoVNACapture` class, so you can call it
from Python code directly:

```python
from nanovna_capture import NanoVNACapture

capture = NanoVNACapture(port="/dev/serial/by-id/usb-nanovna.com_NanoVNA-H4_400-if00")
result = capture.capture_to_files(
    start_hz=1_000_000,
    stop_hz=30_000_000,
    points=101,
    csv_path="nanovna_capture.csv",
    plot_path="nanovna_capture.png",
)

print(result.interface_name)
```

`CoilRelaisTuner` accepts the same NanoVNA configuration and exposes
`capture_smithchart(...)` as a convenience wrapper:

```python
from CoilRelaisTuner import CoilRelaisTuner

with CoilRelaisTuner(
    port="/dev/ttyUSB0",
    nanovna_port="/dev/serial/by-id/usb-nanovna.com_NanoVNA-H4_400-if00",
) as tuner:
    tuner.setup()
    tuner.capture_smithchart(
        start_hz=1_000_000,
        stop_hz=30_000_000,
        points=101,
        plot_path="smithchart.png",
    )
```
