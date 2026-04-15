#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("./image")
DEFAULT_OUTPUT_DIR = Path("./runs/latest")
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "low"
PIXEL_SCHEMA_PATH = Path(__file__).with_name("pixel_schema.json")
RECTANGLE_SCHEMA_PATH = Path(__file__).with_name("rectangle_schema.json")
JUDGE_SCHEMA_PATH = Path(__file__).with_name("judge_schema.json")


class ReconstructionError(Exception):
    """Raised when the reconstruction run cannot continue."""


def run_command(command: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=input_bytes,
        capture_output=True,
        check=False,
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def wait_for_source(path: Path, poll_interval: float) -> Path:
    while not path.exists():
        time.sleep(poll_interval)
    return path


def probe_image_dimensions(path: Path) -> tuple[int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReconstructionError(f"ffprobe failed for {path}: {stderr}")

    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams") or []
    if not streams:
        raise ReconstructionError(f"No image stream found in {path}")

    stream = streams[0]
    width = stream.get("width")
    height = stream.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ReconstructionError(f"Missing width/height in ffprobe output for {path}")
    return width, height


def load_rgb_image(path: Path) -> tuple[int, int, bytearray]:
    width, height = probe_image_dimensions(path)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = run_command(command)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReconstructionError(f"ffmpeg decode failed for {path}: {stderr}")

    pixels = bytearray(result.stdout)
    expected_length = width * height * 3
    if len(pixels) != expected_length:
        raise ReconstructionError(
            f"Decoded image length mismatch for {path}: got {len(pixels)}, expected {expected_length}"
        )
    return width, height, pixels


def save_rgb_image(path: Path, width: int, height: int, pixels: bytes | bytearray) -> None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        str(path),
    ]
    result = run_command(command, input_bytes=bytes(pixels))
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReconstructionError(f"ffmpeg encode failed for {path}: {stderr}")


def make_white_canvas(width: int, height: int) -> bytearray:
    return bytearray([255]) * (width * height * 3)


def sanitize_response_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def load_sequence(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def coerce_pixel_response(payload: Any, width: int, height: int) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise ValueError("pixel response must be a JSON object")

    required = ["x", "y", "r", "g", "b"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")

    try:
        x = int(payload["x"])
        y = int(payload["y"])
        r = int(payload["r"])
        g = int(payload["g"])
        b = int(payload["b"])
    except (TypeError, ValueError) as exc:
        raise ValueError("x, y, r, g, b must be integers") from exc

    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"pixel coordinate ({x}, {y}) is out of bounds for {width}x{height}")

    return {
        "x": x,
        "y": y,
        "r": max(0, min(255, r)),
        "g": max(0, min(255, g)),
        "b": max(0, min(255, b)),
    }


def coerce_rectangle_response(payload: Any, width: int, height: int) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise ValueError("rectangle response must be a JSON object")

    required = ["x", "y", "w", "h", "r", "g", "b"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")

    try:
        x = int(payload["x"])
        y = int(payload["y"])
        w = int(payload["w"])
        h = int(payload["h"])
        r = int(payload["r"])
        g = int(payload["g"])
        b = int(payload["b"])
    except (TypeError, ValueError) as exc:
        raise ValueError("x, y, w, h, r, g, b must be integers") from exc

    if w <= 0 or h <= 0:
        raise ValueError("rectangle width and height must be positive")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"rectangle ({x}, {y}, {w}, {h}) is out of bounds for {width}x{height}")

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "r": max(0, min(255, r)),
        "g": max(0, min(255, g)),
        "b": max(0, min(255, b)),
    }


def coerce_judge_response(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")

    status = payload.get("status")
    reason = payload.get("reason")
    if status not in {"continue", "stop"}:
        raise ValueError("judge status must be 'continue' or 'stop'")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("judge reason must be a non-empty string")
    return {"status": status, "reason": reason.strip()}


def apply_pixel_update(
    pixels: bytearray,
    write_counts: list[int],
    width: int,
    proposal: dict[str, int],
) -> dict[str, Any]:
    pixel_index = proposal["y"] * width + proposal["x"]
    base = pixel_index * 3
    previous = tuple(pixels[base : base + 3])
    had_previous_write = write_counts[pixel_index] > 0
    proposed = (proposal["r"], proposal["g"], proposal["b"])

    if had_previous_write:
        updated = tuple((old + new) // 2 for old, new in zip(previous, proposed))
    else:
        updated = proposed

    pixels[base : base + 3] = bytes(updated)
    write_counts[pixel_index] += 1

    return {
        "mode": "pixel",
        "previous_color": list(previous),
        "new_color": list(updated),
        "averaged": had_previous_write,
        "write_count": write_counts[pixel_index],
    }


def apply_rectangle_update(
    pixels: bytearray,
    write_counts: list[int],
    width: int,
    proposal: dict[str, int],
) -> dict[str, Any]:
    fill_color = (proposal["r"], proposal["g"], proposal["b"])
    touched_pixels = 0
    averaged_pixels = 0
    write_count_min: int | None = None
    write_count_max = 0

    for row in range(proposal["y"], proposal["y"] + proposal["h"]):
        for col in range(proposal["x"], proposal["x"] + proposal["w"]):
            pixel_index = row * width + col
            base = pixel_index * 3
            previous = tuple(pixels[base : base + 3])
            had_previous_write = write_counts[pixel_index] > 0

            if had_previous_write:
                updated = tuple((old + new) // 2 for old, new in zip(previous, fill_color))
                averaged_pixels += 1
            else:
                updated = fill_color

            pixels[base : base + 3] = bytes(updated)
            write_counts[pixel_index] += 1
            write_count_min = write_counts[pixel_index] if write_count_min is None else min(write_count_min, write_counts[pixel_index])
            write_count_max = max(write_count_max, write_counts[pixel_index])
            touched_pixels += 1

    return {
        "mode": "rectangle",
        "fill_color": list(fill_color),
        "touched_pixels": touched_pixels,
        "averaged_pixels": averaged_pixels,
        "write_count_min": write_count_min,
        "write_count_max": write_count_max,
    }


def pixel_agent_prompt(width: int, height: int, step_index: int) -> str:
    return (
        "You are one agent in a collaborative image reconstruction process.\n"
        "Two images are attached: the target image and the latest canvas image.\n"
        f"The image dimensions are width={width} and height={height} pixels.\n"
        f"This is proposal step {step_index}.\n"
        "Use only visual inspection of the attached images.\n"
        "Do not use code, tools, shell commands, calculations, or analytical methods.\n"
        "Choose exactly one pixel coordinate on the canvas and exactly one RGB color to place there.\n"
        "Coordinates must be zero-based integers within the image bounds.\n"
        "Return only JSON with keys x, y, r, g, b and no extra text."
    )


def rectangle_agent_prompt(width: int, height: int, step_index: int) -> str:
    return (
        "You are one agent in a collaborative image reconstruction process.\n"
        "Two images are attached: the target image and the latest canvas image.\n"
        f"The image dimensions are width={width} and height={height} pixels.\n"
        f"This is proposal step {step_index}.\n"
        "Use only visual inspection of the attached images.\n"
        "Do not use code, tools, shell commands, calculations, or analytical methods.\n"
        "Choose exactly one filled rectangle on the canvas and exactly one RGB color to fill it with.\n"
        "Coordinates and sizes must be zero-based integers in bounds, using keys x, y, w, h.\n"
        "The rectangle must fit entirely inside the image.\n"
        "Return only JSON with keys x, y, w, h, r, g, b and no extra text."
    )


def judge_prompt(width: int, height: int, step_index: int) -> str:
    return (
        "You are the completion judge for a collaborative image reconstruction process.\n"
        "Two images are attached: the target image and the latest canvas image.\n"
        f"The image dimensions are width={width} and height={height} pixels.\n"
        f"The canvas currently contains {step_index} accepted drawing edits.\n"
        "Use only visual inspection of the attached images.\n"
        "Do not use code, tools, shell commands, calculations, or analytical methods.\n"
        "If the canvas looks sufficiently complete, return status='stop'. Otherwise return status='continue'.\n"
        "Return only JSON with keys status and reason and no extra text."
    )


@dataclass
class RunArtifacts:
    output_dir: Path
    frames_dir: Path
    manifest_path: Path
    summary_path: Path
    source_copy_path: Path
    initial_canvas_path: Path


class ResponseProvider:
    def get_agent_choice(
        self,
        original_image: Path,
        current_canvas: Path,
        width: int,
        height: int,
        step_index: int,
        *,
        pixel_mode: bool,
    ) -> dict[str, int]:
        raise NotImplementedError

    def get_judge_decision(
        self, original_image: Path, current_canvas: Path, width: int, height: int, step_index: int
    ) -> dict[str, str]:
        raise NotImplementedError


class CodexResponseProvider(ResponseProvider):
    def __init__(
        self,
        *,
        working_dir: Path,
        codex_bin: str,
        pixel_schema_path: Path,
        rectangle_schema_path: Path,
        judge_schema_path: Path,
        model: str | None,
        judge_model: str | None,
        reasoning_effort: str | None,
    ) -> None:
        self.working_dir = working_dir
        self.codex_bin = codex_bin
        self.pixel_schema_path = pixel_schema_path
        self.rectangle_schema_path = rectangle_schema_path
        self.judge_schema_path = judge_schema_path
        self.model = model
        self.judge_model = judge_model
        self.reasoning_effort = reasoning_effort

    def _invoke(self, *, prompt: str, schema_path: Path, images: list[Path], model: str | None) -> Any:
        with tempfile.NamedTemporaryFile(prefix="codex-output-", suffix=".json", delete=False) as handle:
            output_path = Path(handle.name)

        try:
            command = [
                self.codex_bin,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "-s",
                "read-only",
                "-C",
                str(self.working_dir),
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
            ]
            if model:
                command.extend(["-m", model])
            if self.reasoning_effort:
                command.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
            for image in images:
                command.extend(["-i", str(image)])
            command.append("-")

            result = run_command(command, input_bytes=prompt.encode("utf-8"))
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                stdout = result.stdout.decode("utf-8", errors="replace").strip()
                raise ReconstructionError(
                    f"codex exec failed with code {result.returncode}: {stderr or stdout or 'no output'}"
                )

            if not output_path.exists():
                raise ReconstructionError("codex exec did not write the output file")

            text = sanitize_response_text(output_path.read_text(encoding="utf-8"))
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ReconstructionError(f"codex exec returned non-JSON output: {exc}") from exc
        finally:
            output_path.unlink(missing_ok=True)

    def get_agent_choice(
        self,
        original_image: Path,
        current_canvas: Path,
        width: int,
        height: int,
        step_index: int,
        *,
        pixel_mode: bool,
    ) -> dict[str, int]:
        if pixel_mode:
            payload = self._invoke(
                prompt=pixel_agent_prompt(width, height, step_index),
                schema_path=self.pixel_schema_path,
                images=[original_image, current_canvas],
                model=self.model,
            )
            return coerce_pixel_response(payload, width, height)

        payload = self._invoke(
            prompt=rectangle_agent_prompt(width, height, step_index),
            schema_path=self.rectangle_schema_path,
            images=[original_image, current_canvas],
            model=self.model,
        )
        return coerce_rectangle_response(payload, width, height)

    def get_judge_decision(
        self, original_image: Path, current_canvas: Path, width: int, height: int, step_index: int
    ) -> dict[str, str]:
        payload = self._invoke(
            prompt=judge_prompt(width, height, step_index),
            schema_path=self.judge_schema_path,
            images=[original_image, current_canvas],
            model=self.judge_model or self.model,
        )
        return coerce_judge_response(payload)


class MockResponseProvider(ResponseProvider):
    def __init__(self, *, agent_responses: list[Any], judge_responses: list[Any]) -> None:
        self.agent_responses = agent_responses
        self.judge_responses = judge_responses
        self.agent_index = 0
        self.judge_index = 0

    def get_agent_choice(
        self,
        original_image: Path,
        current_canvas: Path,
        width: int,
        height: int,
        step_index: int,
        *,
        pixel_mode: bool,
    ) -> dict[str, int]:
        if self.agent_index >= len(self.agent_responses):
            raise ReconstructionError("mock agent responses exhausted")
        payload = self.agent_responses[self.agent_index]
        self.agent_index += 1
        if pixel_mode:
            return coerce_pixel_response(payload, width, height)
        return coerce_rectangle_response(payload, width, height)

    def get_judge_decision(
        self, original_image: Path, current_canvas: Path, width: int, height: int, step_index: int
    ) -> dict[str, str]:
        if self.judge_index >= len(self.judge_responses):
            raise ReconstructionError("mock judge responses exhausted")
        payload = self.judge_responses[self.judge_index]
        self.judge_index += 1
        return coerce_judge_response(payload)


def prepare_artifacts(output_dir: Path, source: Path, width: int, height: int) -> RunArtifacts:
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    source_copy_path = output_dir / "source_image"
    initial_canvas_path = output_dir / "initial_canvas.png"
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    shutil.copyfile(source, source_copy_path)
    save_rgb_image(initial_canvas_path, width, height, make_white_canvas(width, height))
    manifest_path.write_text("", encoding="utf-8")

    return RunArtifacts(
        output_dir=output_dir,
        frames_dir=frames_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        source_copy_path=source_copy_path,
        initial_canvas_path=initial_canvas_path,
    )


def run_reconstruction(
    *,
    source: Path,
    output_dir: Path,
    provider: ResponseProvider,
    judge_interval: int,
    max_steps: int,
    poll_interval: float,
    agent_retries: int,
    pixel_mode: bool,
) -> dict[str, Any]:
    if judge_interval <= 0:
        raise ReconstructionError("judge_interval must be positive")
    if max_steps <= 0:
        raise ReconstructionError("max_steps must be positive")
    if agent_retries <= 0:
        raise ReconstructionError("agent_retries must be positive")

    source = wait_for_source(source, poll_interval)
    width, height = probe_image_dimensions(source)
    canvas = make_white_canvas(width, height)
    write_counts = [0] * (width * height)

    artifacts = prepare_artifacts(output_dir, source, width, height)
    latest_canvas_path = artifacts.initial_canvas_path
    accepted_steps = 0
    total_agent_calls = 0
    judge_calls = 0
    stop_reason = "max_steps_reached"

    append_jsonl(
        artifacts.manifest_path,
        {
            "type": "run_started",
            "source": str(source),
            "width": width,
            "height": height,
            "mode": "pixel" if pixel_mode else "rectangle",
            "initial_canvas_path": str(artifacts.initial_canvas_path),
        },
    )

    for attempt_index in range(1, max_steps + 1):
        retry_errors: list[str] = []
        proposal: dict[str, int] | None = None
        for _ in range(1, agent_retries + 1):
            total_agent_calls += 1
            try:
                proposal = provider.get_agent_choice(
                    source,
                    latest_canvas_path,
                    width,
                    height,
                    accepted_steps + 1,
                    pixel_mode=pixel_mode,
                )
                break
            except (ReconstructionError, ValueError, json.JSONDecodeError) as exc:
                retry_errors.append(str(exc))

        if proposal is None:
            append_jsonl(
                artifacts.manifest_path,
                {
                    "type": "skip",
                    "attempt": attempt_index,
                    "step": accepted_steps,
                    "errors": retry_errors,
                },
            )
            continue

        if pixel_mode:
            update = apply_pixel_update(canvas, write_counts, width, proposal)
        else:
            update = apply_rectangle_update(canvas, write_counts, width, proposal)
        accepted_steps += 1
        frame_path = artifacts.frames_dir / f"step_{accepted_steps:06d}.png"
        save_rgb_image(frame_path, width, height, canvas)
        latest_canvas_path = frame_path

        append_jsonl(
            artifacts.manifest_path,
            {
                "type": "edit",
                "mode": "pixel" if pixel_mode else "rectangle",
                "attempt": attempt_index,
                "step": accepted_steps,
                "proposal": proposal,
                "retries_before_success": len(retry_errors),
                "retry_errors": retry_errors,
                "frame_path": str(frame_path),
                **update,
            },
        )

        if accepted_steps % judge_interval != 0:
            continue

        judge_calls += 1
        decision = provider.get_judge_decision(source, latest_canvas_path, width, height, accepted_steps)
        append_jsonl(
            artifacts.manifest_path,
            {
                "type": "judge",
                "step": accepted_steps,
                "decision": decision,
            },
        )
        if decision["status"] == "stop":
            stop_reason = decision["reason"]
            break
    else:
        stop_reason = "max_steps_reached"

    summary = {
        "source": str(source),
        "source_copy_path": str(artifacts.source_copy_path),
        "width": width,
        "height": height,
        "accepted_steps": accepted_steps,
        "total_agent_calls": total_agent_calls,
        "judge_calls": judge_calls,
        "judge_interval": judge_interval,
        "max_steps": max_steps,
        "agent_retries": agent_retries,
        "mode": "pixel" if pixel_mode else "rectangle",
        "final_canvas_path": str(latest_canvas_path),
        "manifest_path": str(artifacts.manifest_path),
        "stop_reason": stop_reason,
    }
    write_json(artifacts.summary_path, summary)
    return summary


def build_provider(args: argparse.Namespace) -> ResponseProvider:
    if args.mock_agent_responses or args.mock_judge_responses:
        if not args.mock_agent_responses or not args.mock_judge_responses:
            raise ReconstructionError("both --mock-agent-responses and --mock-judge-responses are required together")
        return MockResponseProvider(
            agent_responses=load_sequence(args.mock_agent_responses),
            judge_responses=load_sequence(args.mock_judge_responses),
        )

    return CodexResponseProvider(
        working_dir=Path.cwd(),
        codex_bin=args.codex_bin,
        pixel_schema_path=PIXEL_SCHEMA_PATH,
        rectangle_schema_path=RECTANGLE_SCHEMA_PATH,
        judge_schema_path=JUDGE_SCHEMA_PATH,
        model=args.model,
        judge_model=args.judge_model,
        reasoning_effort=args.reasoning_effort,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct an image with Codex agents using rectangle or pixel edits.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to the source image file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where frames, manifest, and summary will be written.",
    )
    parser.add_argument("--judge-interval", type=int, default=25, help="Run the judge after this many accepted edits.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Hard cap on accepted-edit attempts.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds to wait between checks when the source file is missing.",
    )
    parser.add_argument("--agent-retries", type=int, default=3, help="Retries for malformed agent output.")
    parser.add_argument(
        "--pixel-mode",
        action="store_true",
        help="Use single-pixel proposals instead of the default rectangle-fill mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Codex model for drawing agents (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Optional Codex model override for the judge; defaults to `--model`.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=DEFAULT_REASONING_EFFORT,
        choices=["low", "medium", "high", "xhigh"],
        help=f"Reasoning effort for both agents (default: {DEFAULT_REASONING_EFFORT}).",
    )
    parser.add_argument("--codex-bin", type=str, default="codex", help="Codex executable to invoke.")
    parser.add_argument(
        "--mock-agent-responses",
        type=Path,
        default=None,
        help="JSON or JSONL file of mocked agent outputs for local testing.",
    )
    parser.add_argument(
        "--mock-judge-responses",
        type=Path,
        default=None,
        help="JSON or JSONL file of mocked judge outputs for local testing.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        provider = build_provider(args)
        summary = run_reconstruction(
            source=args.source,
            output_dir=args.output_dir,
            provider=provider,
            judge_interval=args.judge_interval,
            max_steps=args.max_steps,
            poll_interval=args.poll_interval,
            agent_retries=args.agent_retries,
            pixel_mode=args.pixel_mode,
        )
    except ReconstructionError as exc:
        parser.exit(1, f"error: {exc}\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
