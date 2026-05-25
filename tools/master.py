"""
OS Music Pipeline — Tool #1: Reference Mastering
Uses matchering to master your track against a reference (Drake, Weeknd, etc.)

Usage:
    python tools/master.py my_raw_mix.wav --ref drake_reference.wav
    python tools/master.py my_raw_mix.wav --ref weeknd_reference.wav -o mastered_output.wav
    python tools/master.py my_raw_mix.wav --ref nav_reference.wav --format mp3
"""

import argparse
import sys
from pathlib import Path

try:
    import matchering as mg
except ImportError:
    print("matchering not installed. Run: pip install matchering")
    sys.exit(1)


def master_track(target: str, reference: str, output: str = None, fmt: str = "wav"):
    target_path = Path(target)
    ref_path = Path(reference)

    if not target_path.exists():
        print(f"Error: Target file not found: {target}")
        sys.exit(1)
    if not ref_path.exists():
        print(f"Error: Reference file not found: {reference}")
        sys.exit(1)

    if output is None:
        output = str(target_path.stem) + "_mastered"

    results = []
    if fmt == "wav":
        out_file = f"{output}.wav"
        results.append(mg.pcm24(out_file))
    elif fmt == "mp3":
        out_file = f"{output}.mp3"
        results.append(mg.mp3(out_file))
    else:
        out_file = f"{output}.wav"
        results.append(mg.pcm24(out_file))

    print(f"Mastering '{target_path.name}' against '{ref_path.name}'...")
    print(f"This matches loudness, EQ curve, and dynamics to the reference.")
    print()

    mg.process(
        target=str(target_path),
        reference=str(ref_path),
        results=results,
    )

    print(f"Done! Mastered file: {out_file}")
    print()
    print("What happened:")
    print("  - Analyzed the frequency spectrum of your reference track")
    print("  - Matched your track's EQ curve to the reference")
    print("  - Matched loudness (LUFS) to the reference")
    print("  - Applied limiting to prevent clipping")
    print()
    print("Tip: Use a lossless (.wav/.flac) reference for best results.")


def main():
    parser = argparse.ArgumentParser(
        description="Master your track against a reference (Drake, Weeknd, Nav, etc.)"
    )
    parser.add_argument("target", help="Your raw mix file (.wav)")
    parser.add_argument("--ref", "-r", required=True, help="Reference track to match (.wav)")
    parser.add_argument("--output", "-o", default=None, help="Output filename (without extension)")
    parser.add_argument("--format", "-f", default="wav", choices=["wav", "mp3"], help="Output format")

    args = parser.parse_args()
    master_track(args.target, args.ref, args.output, args.format)


if __name__ == "__main__":
    main()
