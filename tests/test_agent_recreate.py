from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from agent_recreate import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    MockResponseProvider,
    apply_rectangle_update,
    apply_pixel_update,
    build_parser,
    load_rgb_image,
    make_white_canvas,
    probe_image_dimensions,
    run_reconstruction,
    save_rgb_image,
)


def create_test_image(path: Path, width: int, height: int, color: str) -> None:
    png_path = path.with_suffix(".png")
    colors = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    if color not in colors:
        raise ValueError(f"unsupported test color: {color}")
    r, g, b = colors[color]
    pixels = bytearray()
    for _ in range(width * height):
        pixels.extend([r, g, b])
    save_rgb_image(png_path, width, height, pixels)
    shutil.copyfile(png_path, path)


class AgentRecreateTests(unittest.TestCase):
    def test_probe_handles_extensionless_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "image"
            create_test_image(source, 3, 2, "red")

            self.assertEqual(probe_image_dimensions(source), (3, 2))

    def test_white_canvas_matches_dimensions(self) -> None:
        width, height = 4, 3
        canvas = make_white_canvas(width, height)

        self.assertEqual(len(canvas), width * height * 3)
        self.assertEqual(canvas[:6], bytes([255, 255, 255, 255, 255, 255]))

    def test_first_write_sets_pixel_directly(self) -> None:
        width, height = 2, 2
        canvas = make_white_canvas(width, height)
        write_counts = [0] * (width * height)

        update = apply_pixel_update(canvas, write_counts, width, {"x": 1, "y": 0, "r": 10, "g": 20, "b": 30})

        self.assertFalse(update["averaged"])
        self.assertEqual(update["new_color"], [10, 20, 30])
        self.assertEqual(list(canvas[3:6]), [10, 20, 30])

    def test_repeated_write_averages_existing_pixel(self) -> None:
        width, height = 1, 1
        canvas = make_white_canvas(width, height)
        write_counts = [0]

        apply_pixel_update(canvas, write_counts, width, {"x": 0, "y": 0, "r": 20, "g": 40, "b": 60})
        update = apply_pixel_update(canvas, write_counts, width, {"x": 0, "y": 0, "r": 100, "g": 120, "b": 140})

        self.assertTrue(update["averaged"])
        self.assertEqual(update["new_color"], [60, 80, 100])
        self.assertEqual(list(canvas[:3]), [60, 80, 100])

    def test_rectangle_update_fills_region(self) -> None:
        width, height = 3, 2
        canvas = make_white_canvas(width, height)
        write_counts = [0] * (width * height)

        update = apply_rectangle_update(
            canvas,
            write_counts,
            width,
            {"x": 1, "y": 0, "w": 2, "h": 2, "r": 10, "g": 20, "b": 30},
        )

        self.assertEqual(update["mode"], "rectangle")
        self.assertEqual(update["touched_pixels"], 4)
        self.assertEqual(list(canvas[3:6]), [10, 20, 30])
        self.assertEqual(list(canvas[6:9]), [10, 20, 30])
        self.assertEqual(list(canvas[12:15]), [10, 20, 30])
        self.assertEqual(list(canvas[15:18]), [10, 20, 30])

    def test_runner_defaults_to_rectangle_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "image"
            output_dir = root / "run"
            create_test_image(source, 2, 2, "blue")

            provider = MockResponseProvider(
                agent_responses=[
                    {"x": 0, "y": 0, "w": 2, "h": 1, "r": 255, "g": 0, "b": 0},
                    {"x": 0, "y": 1, "w": 2, "h": 1, "r": 0, "g": 255, "b": 0},
                ],
                judge_responses=[{"status": "continue", "reason": "keep going"}],
            )

            summary = run_reconstruction(
                source=source,
                output_dir=output_dir,
                provider=provider,
                judge_interval=2,
                max_steps=2,
                poll_interval=0.01,
                agent_retries=3,
                pixel_mode=False,
            )

            self.assertEqual(summary["accepted_steps"], 2)
            self.assertEqual(summary["mode"], "rectangle")
            self.assertTrue((output_dir / "initial_canvas.png").exists())
            self.assertTrue((output_dir / "frames" / "step_000001.png").exists())
            self.assertTrue((output_dir / "frames" / "step_000002.png").exists())

            lines = (output_dir / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
            payloads = [json.loads(line) for line in lines]
            event_types = [item["type"] for item in payloads]
            self.assertEqual(event_types, ["run_started", "edit", "edit", "judge"])
            self.assertEqual(payloads[0]["mode"], "rectangle")
            self.assertEqual(payloads[1]["proposal"]["w"], 2)

    def test_runner_skips_after_retries_for_invalid_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "image"
            output_dir = root / "run"
            create_test_image(source, 2, 2, "white")

            provider = MockResponseProvider(
                agent_responses=[
                    {"x": 9, "y": 9, "w": 1, "h": 1, "r": 0, "g": 0, "b": 0},
                    {"x": 9, "y": 9, "w": 1, "h": 1, "r": 0, "g": 0, "b": 0},
                    {"oops": True},
                ],
                judge_responses=[],
            )

            summary = run_reconstruction(
                source=source,
                output_dir=output_dir,
                provider=provider,
                judge_interval=1,
                max_steps=1,
                poll_interval=0.01,
                agent_retries=3,
                pixel_mode=False,
            )

            self.assertEqual(summary["accepted_steps"], 0)
            lines = (output_dir / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
            skip_event = json.loads(lines[-1])
            self.assertEqual(skip_event["type"], "skip")
            self.assertEqual(len(skip_event["errors"]), 3)

    def test_judge_can_stop_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "image"
            output_dir = root / "run"
            create_test_image(source, 1, 1, "black")

            provider = MockResponseProvider(
                agent_responses=[
                    {"x": 0, "y": 0, "r": 0, "g": 0, "b": 0},
                    {"x": 0, "y": 0, "r": 255, "g": 255, "b": 255},
                ],
                judge_responses=[{"status": "stop", "reason": "looks complete"}],
            )

            summary = run_reconstruction(
                source=source,
                output_dir=output_dir,
                provider=provider,
                judge_interval=1,
                max_steps=2,
                poll_interval=0.01,
                agent_retries=3,
                pixel_mode=True,
            )

            self.assertEqual(summary["accepted_steps"], 1)
            self.assertEqual(summary["judge_calls"], 1)
            self.assertEqual(summary["stop_reason"], "looks complete")
            self.assertEqual(summary["mode"], "pixel")

    def test_saved_frame_contains_updated_rectangle_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "image"
            output_dir = root / "run-rect"
            create_test_image(source, 2, 1, "white")

            provider = MockResponseProvider(
                agent_responses=[{"x": 0, "y": 0, "w": 2, "h": 1, "r": 12, "g": 34, "b": 56}],
                judge_responses=[{"status": "stop", "reason": "done"}],
            )

            run_reconstruction(
                source=source,
                output_dir=output_dir,
                provider=provider,
                judge_interval=1,
                max_steps=1,
                poll_interval=0.01,
                agent_retries=3,
                pixel_mode=False,
            )

            width, height, pixels = load_rgb_image(output_dir / "frames" / "step_000001.png")
            self.assertEqual((width, height), (2, 1))
            self.assertEqual(list(pixels[:3]), [12, 34, 56])
            self.assertEqual(list(pixels[3:6]), [12, 34, 56])

    def test_saved_frame_contains_updated_pixel_in_pixel_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "image"
            output_dir = root / "run-pixel"
            create_test_image(source, 2, 1, "white")

            provider = MockResponseProvider(
                agent_responses=[{"x": 1, "y": 0, "r": 12, "g": 34, "b": 56}],
                judge_responses=[{"status": "stop", "reason": "done"}],
            )

            run_reconstruction(
                source=source,
                output_dir=output_dir,
                provider=provider,
                judge_interval=1,
                max_steps=1,
                poll_interval=0.01,
                agent_retries=3,
                pixel_mode=True,
            )

            width, height, pixels = load_rgb_image(output_dir / "frames" / "step_000001.png")
            self.assertEqual((width, height), (2, 1))
            self.assertEqual(list(pixels[3:6]), [12, 34, 56])

    def test_parser_defaults_to_rectangle_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        self.assertFalse(args.pixel_mode)
        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertEqual(args.reasoning_effort, DEFAULT_REASONING_EFFORT)


if __name__ == "__main__":
    unittest.main()
