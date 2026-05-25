"""Stem separation via Demucs. Splits audio into vocals, drums, bass, and other."""

import argparse
import subprocess
import sys
from pathlib import Path


def check_demucs():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "demucs", "--help"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def separate_track(input_path: str, output_dir: str = None, vocals_only: bool = False, model: str = "htdemucs"):
    path = Path(input_path)

    if not path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    cmd = [sys.executable, "-m", "demucs"]

    # Model selection
    cmd.extend(["-n", model])

    # Vocals-only mode (2 stems: vocals + accompaniment)
    if vocals_only:
        cmd.append("--two-stems=vocals")

    # Output directory
    if output_dir:
        cmd.extend(["-o", output_dir])
    else:
        cmd.extend(["-o", "separated"])

    # Handle single file or directory
    if path.is_dir():
        files = list(path.glob("*.mp3")) + list(path.glob("*.wav")) + list(path.glob("*.flac"))
        if not files:
            print(f"No audio files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(files)} audio files to process.")
        for f in files:
            cmd_file = cmd + [str(f)]
            _run_separation(cmd_file, f.name)
    else:
        cmd.append(str(path))
        _run_separation(cmd, path.name)


def _run_separation(cmd: list, filename: str):
    print(f"Separating: {filename}")

    try:
        process = subprocess.run(cmd, text=True)
        if process.returncode == 0:
            print(f"Done: {filename} > vocals.wav, drums.wav, bass.wav, other.wav")
        else:
            print(f"Error processing {filename}. Run: pip install demucs")
    except FileNotFoundError:
        print("demucs not found. Run: pip install demucs")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Separate any song into stems (vocals, drums, bass, other)"
    )
    parser.add_argument("input", help="Audio file or folder of audio files")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: ./separated)")
    parser.add_argument("--vocals-only", "-v", action="store_true",
                        help="Only separate vocals vs instrumental (faster)")
    parser.add_argument("--model", "-m", default="htdemucs",
                        choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
                        help="Demucs model (htdemucs=fast, htdemucs_ft=best quality, mdx_extra=alt)")

    args = parser.parse_args()

    if not check_demucs():
        print("Demucs is not installed. Install with:")
        print("  pip install demucs")
        sys.exit(1)

    separate_track(args.input, args.output, args.vocals_only, args.model)


if __name__ == "__main__":
    main()
