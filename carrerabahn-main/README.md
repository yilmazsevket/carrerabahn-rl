# CarreraBahn RL – RealSense + Arduino

A university group project (Machine Learning track): A Carrera slot car is speed‑controlled by an Arduino (digital potentiometer via SPI) and tracked by an Intel RealSense camera. A Reinforcement Learning agent (NFQ, in PyTorch) learns a driving policy based on speed, progress, and off‑track status.

## Features

- Real‑time tracking of a green marker (the car) using Intel RealSense (color stream)
- Recording one complete lap as a list of 2D points (reference track)
- Off‑track detection and progress estimation along the recorded track
- Speed control via Arduino → MCP41X1 (digital potentiometer) → Carrera controller
- NFQ agent (PyTorch) with ε‑greedy exploration and replay buffer

## Architecture

- `Camera.py` – RealSense tracking, lap recording, off‑track check, speed (px/s), progress (% of lap)
- `Arduino.py` – Serial interface to Arduino, sets speed (0–100 recommended; see note below)
- `CarreraTrackEnvQ.py` – NFQ network, agent, and a minimal environment (expects precomputed observations)
- `main.py` – Orchestration: start Arduino + camera, record lap, run RL training
- `arduino_geschwindigkeit_skaliert.ino` – Arduino sketch: reads 0–100 from Serial, maps to 60–255 (SPI → MCP41X1)
- `MCamera.py`, `temp.py` – experimental/alternative modules (not required for the standard run)

## Requirements

- Hardware

  - Arduino‑compatible board (Serial, SPI), MCP41X1 digital potentiometer (or similar)
  - Intel RealSense camera (tested: color 848×480 @ 60 FPS)
  - Windows PC (project developed on Windows / COM port)

- Software
  - Python 3.10+ (tested with 3.12)
  - Dependencies: see `requirements.txt`

## Installation

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Flash the Arduino sketch `arduino_geschwindigkeit_skaliert.ino` with the Arduino IDE. Adjust the COM port in code if needed (default: `COM5`).

## Usage

1. Connect the RealSense camera; connect the Arduino via USB; wire the Carrera controller through the MCP41X1.
2. If needed, set `port="COMx"` in `Arduino.py` to your actual COM port.
3. Run:

```cmd
python main.py
```

Interactive flow:

- Press Enter to capture the starting point
- Let the car complete one full lap (the script detects the full lap and stops recording)
- Training then starts for multiple episodes; the agent continuously sets speed

Notes:

- The camera segments "green" in HSV space. Under different lighting you may need to adjust the thresholds.
- If no position is detected, a robust fallback/retry should be used in production (see Improvements).

## Known Limitations & Improvements

- Host ↔ Arduino protocol alignment:
  - The Arduino sketch expects values 0–100 and internally maps to 60–255 for the potentiometer.
  - The Python agent currently produces actions 0–255. Recommendation: either normalize host output to 0–100, or change the sketch to accept 0–255 directly.
- `record_round()` currently returns `None`; the recorded track is stored in `camera.positions`. For clarity, return the positions explicitly or adapt the call site to use `camera.positions`.
- Depth stream is not used → can be disabled to save resources.
- Reward design can include progress/lap time more strongly (currently mainly speed with an off‑track penalty).

## Project Structure

```
├─ Arduino.py                      # Serial interface
├─ Camera.py                       # RealSense tracking & track logic
├─ CarreraTrackEnvQ.py             # NFQ network, agent, environment
├─ main.py                         # Entry point
├─ MCamera.py                      # alternative camera variant (optional)
├─ temp.py                         # experimental Gym environment (optional)
├─ arduino_geschwindigkeit_skaliert.ino  # Arduino sketch (SPI → MCP41X1)
├─ requirements.txt                # Python dependencies
└─ .gitignore
```

## Contributors

University group project (Machine Learning track).
Mustafa Turhal
Sevket Yilmaz
Nick Justus
Giacomo Schomburg
Anas Karah
