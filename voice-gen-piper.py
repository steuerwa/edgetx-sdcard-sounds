#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from piper import PiperVoice, SynthesisConfig
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [FILE] [LANGDIR] -m MODEL [-s LENGTH_SCALE]",
        description="Generate voice packs from CSV list using local Piper TTS (https://github.com/rhasspy/piper).",
    )

    parser.add_argument("-v", "--version", action="version", version=f"{parser.prog} version 1.0.0")

    parser.add_argument("file", type=str, help="CSV Translation file")

    parser.add_argument("langdir", type=str, help="Language subfolder")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to Piper voice model (*.onnx), config defaults to <model>.json",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to Piper voice config (*.onnx.json). Defaults to <model>.json",
        required=False,
        default=None,
    )

    parser.add_argument(
        "-s",
        "--length-scale",
        type=float,
        help="Piper length scale (higher = slower speech)",
        required=False,
        default=1.0,
    )

    return parser


def process_sample(infile: str, outfile: str):
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            infile,
            "-ar",
            "32000",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            outfile,
        ],
        check=True,
    )


def synthesize_sample(voice: PiperVoice, text: str, outfile: str, syn_config: SynthesisConfig):
    with wave.open(outfile, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    # Print which script is running and the file being processed
    print(f"Running {os.path.basename(__file__)} to process file: {args.file}")

    csv_file = args.file
    csv_rows = 0
    langdir = args.langdir
    basedir = Path(__file__).resolve().parent
    outdir = Path()
    model_path = Path(args.model).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else Path(str(model_path) + ".json")
    syn_config = SynthesisConfig(length_scale=args.length_scale)

    in_ci = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"

    csv_path = Path(csv_file).resolve()
    voices_root = Path(__file__).resolve().parent / "voices"

    if not model_path.is_file():
        print(f"Error: Piper model not found: {model_path}")
        sys.exit(1)

    if not config_path.is_file():
        print(f"Error: Piper model config not found: {config_path}")
        sys.exit(1)

    all_csvs = sorted(voices_root.glob("*.csv")) if voices_root.exists() else []
    if not all_csvs:
        all_csvs = sorted(csv_path.parent.glob("*.csv"))

    total_files = len(all_csvs) if all_csvs else 1
    processed_files = next((idx + 1 for idx, f in enumerate(all_csvs) if f.resolve() == csv_path), 1)

    if not os.path.isfile(csv_file):
        print("Error: voice file not found")
        sys.exit(1)

    # Get number of rows in CSV file
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        reader = ((field.strip() for field in row) for row in reader)  # Strip spaces
        csv_rows = sum(1 for row in reader)

    # Drop header row from progress count if present
    csv_rows = max(csv_rows - 1, 0)

    console = Console(force_terminal=not in_ci, no_color=in_ci)
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        TextColumn("{task.fields[status]}", justify="left"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=True,
    )

    class StatusLine:
        def __init__(self) -> None:
            self.message = ""

        def update(self, message: str) -> None:
            self.message = message

        def __rich_console__(self, console, options):
            yield Text(self.message)

    status_line = StatusLine()
    layout = Group(status_line, progress)

    report_console = console
    with Live(layout, console=report_console, refresh_per_second=10, transient=False):

        def report(msg: str) -> None:
            if in_ci:
                progress.console.print(msg)
            else:
                status_line.update(msg)
                progress.refresh()

        report(f"Loading Piper voice: {model_path}")
        voice = PiperVoice.load(model_path, config_path=config_path)
        phoneme_type = getattr(voice.config.phoneme_type, "value", voice.config.phoneme_type)
        report(f"Voice loaded (phoneme type: {phoneme_type}, {voice.config.sample_rate} Hz)")

        # Process CSV file with progress bar
        with open(csv_file, "rt") as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quotechar='"')
            reader = ((field.strip() for field in row) for row in reader)  # Strip spaces
            task_id = progress.add_task("Synthesizing", total=csv_rows or None, status="")

            line_count = 0
            processed_count = 0

            try:
                fail_streak = 0
                for row in reader:
                    row = list(row)  # Convert the generator to a list
                    line_count += 1
                    if line_count == 1:
                        # absorb header row
                        continue
                    try:
                        path_part = row[4] if len(row) > 4 else ""
                        outdir = basedir / "SOUNDS" / langdir / path_part if path_part else basedir / "SOUNDS" / langdir
                        en_text = row[1] if len(row) > 1 else ""
                        text = row[2] if len(row) > 2 else ""
                        filename = row[5] if len(row) > 5 else ""
                        outfile = outdir / filename if filename else None

                        if not filename:
                            report(f"[{line_count}/{csv_rows}] Skipping row with no filename")
                            progress.update(task_id, advance=1)
                            processed_count += 1
                            continue

                        tmpfile_fd, tmpfile = tempfile.mkstemp(suffix=".wav")
                        os.close(tmpfile_fd)

                        outdir.mkdir(parents=True, exist_ok=True)

                        if not text:
                            report(f"[{line_count}/{csv_rows}] Skipping as no text to translate")
                            progress.update(task_id, advance=1)
                            processed_count += 1
                            os.unlink(tmpfile)
                            continue

                        if not outfile.exists():
                            report(
                                f'[{line_count}/{csv_rows}] Synthesize "{en_text}" as "{text}", save as "{outfile}".'
                            )

                            synthesize_sample(voice, text, tmpfile, syn_config)
                            process_sample(tmpfile, str(outfile))
                            os.unlink(tmpfile)

                        else:
                            report(f'[{line_count}/{csv_rows}] Skipping "{filename}" as already exists.')
                            os.unlink(tmpfile)

                        progress.update(task_id, advance=1)
                        processed_count += 1
                        fail_streak = 0
                    except Exception as e:
                        report(f"[{line_count}/{csv_rows}] Error processing row: {e}")
                        progress.update(task_id, advance=1)
                        processed_count += 1
                        fail_streak += 1
                        if fail_streak >= 3:
                            report("Aborting after 3 consecutive failures")
                            raise SystemExit(1)
                        continue
            except KeyboardInterrupt:
                report(
                    f"Interrupted. Processed {processed_files}/{total_files} files; "
                    f"{processed_count}/{csv_rows} entries in current file."
                )
                progress.update(task_id, completed=processed_count)
                raise SystemExit(1)

            report(
                f"Finished processing {processed_files}/{total_files} files ({processed_count}/{csv_rows} entries) "
                f'from "{csv_file}" using {os.path.basename(__file__)}.'
            )


if __name__ == "__main__":
    main()
