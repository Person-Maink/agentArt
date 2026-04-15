# agentArt

`agentArt` is a small image-reconstruction experiment driven by Codex agents. It starts from a white canvas and repeatedly asks an agent to propose either a single pixel edit or a filled rectangle, then periodically asks a judge agent whether the canvas is good enough to stop.

The script records each accepted edit as a frame, writes a JSONL manifest of the run, and saves a final summary.

## What it does

- Loads a source image
- Creates a blank white canvas with the same dimensions
- Repeatedly requests drawing proposals from Codex
- Applies accepted edits to the canvas
- Saves each step under `runs/latest/frames/`
- Runs a judge at a configurable interval to decide whether to continue

## Requirements

- Python 3.12+
- `ffmpeg`
- `ffprobe`
- The `codex` CLI available on your `PATH`

The script uses `ffprobe` to inspect image dimensions and `ffmpeg` to read and write RGB image data.

## Project Layout

- `agent_recreate.py` - main entrypoint and reconstruction logic
- `pixel_schema.json` - schema for single-pixel agent responses
- `rectangle_schema.json` - schema for rectangle agent responses
- `judge_schema.json` - schema for judge responses
- `tests/` - unit tests using mocked responses
- `image` - default source image used by the script
- `runs/latest/` - default output directory for the latest run

## Usage

Run the default rectangle-fill mode:

```bash
python agent_recreate.py
```

Use pixel mode instead of rectangle mode:

```bash
python agent_recreate.py --pixel-mode
```

Specify a different source image and output directory:

```bash
python agent_recreate.py --source path/to/source.png --output-dir runs/my-run
```

## CLI Options

- `--source` - source image path, default: `./image`
- `--output-dir` - output directory, default: `./runs/latest`
- `--judge-interval` - run the judge after this many accepted edits, default: `25`
- `--max-steps` - maximum number of accepted-edit attempts, default: `2000`
- `--poll-interval` - how long to wait between checks if the source file does not exist yet, default: `1.0`
- `--agent-retries` - retries for malformed agent output, default: `3`
- `--pixel-mode` - use single-pixel edits instead of rectangle fills
- `--model` - Codex model for drawing agents, default: `gpt-5.4-mini`
- `--judge-model` - optional separate model for the judge
- `--reasoning-effort` - reasoning effort for both agents, default: `low`
- `--codex-bin` - Codex executable name or path, default: `codex`
- `--mock-agent-responses` - JSON or JSONL file of mocked agent outputs for local testing
- `--mock-judge-responses` - JSON or JSONL file of mocked judge outputs for local testing

## Output Files

Each run writes a few artifacts to the output directory:

- `source_image` - copy of the source image used for the run
- `initial_canvas.png` - blank white starting canvas
- `frames/step_000001.png`, `frames/step_000002.png`, ... - saved canvas after each accepted edit
- `manifest.jsonl` - event log for the run
- `summary.json` - final run summary returned by the script

The manifest includes events such as:

- `run_started`
- `edit`
- `judge`
- `skip`

## Local Testing

The test suite uses mocked agent and judge outputs so it can run without calling Codex.

```bash
python -m unittest
```

## Response Formats

The script expects strict JSON responses from the agent and judge:

- Pixel mode: `{ "x": 0, "y": 0, "r": 255, "g": 255, "b": 255 }`
- Rectangle mode: `{ "x": 0, "y": 0, "w": 10, "h": 10, "r": 255, "g": 255, "b": 255 }`
- Judge: `{ "status": "continue", "reason": "..." }` or `{ "status": "stop", "reason": "..." }`

## Notes

- The source image path may exist later than the script starts; the runner will wait for it.
- If an agent returns malformed output, the runner retries up to `--agent-retries` times before skipping that attempt.
- Repeated writes to the same pixel or rectangle region are blended by averaging the old and new color values.
