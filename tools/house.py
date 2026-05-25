"""House music production toolkit: kick/hat synthesis, vocal chops, arrangement structures."""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Required: pip install numpy soundfile")
    sys.exit(1)


# House sub-genre definitions
STYLES = {
    "tech-house": {
        "bpm_range": (124, 128),
        "default_bpm": 126,
        "kick_character": "punchy",
        "hat_pattern": "offbeat",
        "bass_style": "rolling",
        "structure": [
            ("intro", 32, "kick + hats build in"),
            ("build", 16, "bass enters, filter opens"),
            ("drop", 32, "full energy, vocal chop loops"),
            ("break", 16, "strip to vocal + pad"),
            ("build2", 16, "tension, riser, snare roll"),
            ("drop2", 32, "full energy, variation on drop 1"),
            ("outro", 32, "elements remove, kick + hats"),
        ],
        "key_suggestion": "F minor or G minor",
        "reference_artists": "John Summit, Chris Lake, Fisher",
    },
    "deep-house": {
        "bpm_range": (120, 124),
        "default_bpm": 122,
        "kick_character": "warm",
        "hat_pattern": "shuffled",
        "bass_style": "melodic",
        "structure": [
            ("intro", 32, "atmospheric pads, kick enters"),
            ("verse", 32, "bass + chords, vocal phrase"),
            ("build", 16, "filter sweep, tension"),
            ("drop", 32, "full groove, less aggressive than tech"),
            ("break", 16, "melodic moment, vocal"),
            ("drop2", 32, "full groove with variation"),
            ("outro", 32, "elements strip away"),
        ],
        "key_suggestion": "A minor or C minor",
        "reference_artists": "Disclosure, Lane 8, Ben Bohmer",
    },
    "bass-house": {
        "bpm_range": (126, 130),
        "default_bpm": 128,
        "kick_character": "hard",
        "hat_pattern": "driving",
        "bass_style": "wobble",
        "structure": [
            ("intro", 16, "kick pattern, tension"),
            ("build", 16, "riser, snare fill"),
            ("drop", 32, "heavy bass, distorted"),
            ("break", 8, "vocal sample, silence"),
            ("build2", 16, "faster build"),
            ("drop2", 32, "heavier variation"),
            ("break2", 16, "strip down"),
            ("drop3", 32, "final drop, all elements"),
            ("outro", 16, "kick out"),
        ],
        "key_suggestion": "E minor or D minor",
        "reference_artists": "Habstrakt, Joyryde, Skrillex (house era)",
    },
}


def generate_kick(bpm: float, duration_sec: float, sr: int = 44100, character: str = "punchy") -> np.ndarray:
    """Generate a 4-on-the-floor kick pattern."""
    total_samples = int(sr * duration_sec)
    output = np.zeros(total_samples)

    beat_interval = 60.0 / bpm
    samples_per_beat = int(sr * beat_interval)

    # Kick synthesis parameters based on character
    params = {
        "punchy": {"freq_start": 160, "freq_end": 40, "decay": 0.15, "click": 0.8},
        "warm": {"freq_start": 140, "freq_end": 35, "decay": 0.2, "click": 0.4},
        "hard": {"freq_start": 200, "freq_end": 45, "decay": 0.12, "click": 1.0},
    }
    p = params.get(character, params["punchy"])

    kick_len = int(sr * 0.25)  # 250ms kick

    # Synthesize single kick
    t = np.linspace(0, 0.25, kick_len, endpoint=False)
    # Pitch envelope: starts high, drops to sub
    freq_env = p["freq_end"] + (p["freq_start"] - p["freq_end"]) * np.exp(-t * 30)
    # Phase is integral of frequency
    phase = 2 * np.pi * np.cumsum(freq_env) / sr
    kick_wave = np.sin(phase)
    # Amplitude envelope
    amp_env = np.exp(-t / p["decay"])
    # Transient click
    click_env = np.exp(-t * 200) * p["click"]
    click_noise = np.random.randn(kick_len) * click_env

    kick_single = (kick_wave * amp_env + click_noise * 0.3) * 0.8

    # Place kicks on every beat
    pos = 0
    while pos + kick_len < total_samples:
        end = min(pos + kick_len, total_samples)
        output[pos:end] += kick_single[:end - pos]
        pos += samples_per_beat

    # Normalize
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * 0.9

    return output


def generate_hihat(bpm: float, duration_sec: float, sr: int = 44100, pattern: str = "offbeat") -> np.ndarray:
    """Generate hi-hat pattern."""
    total_samples = int(sr * duration_sec)
    output = np.zeros(total_samples)

    beat_interval = 60.0 / bpm
    eighth_interval = beat_interval / 2
    samples_per_eighth = int(sr * eighth_interval)

    hat_len = int(sr * 0.05)  # 50ms hat
    t = np.linspace(0, 0.05, hat_len, endpoint=False)

    # Hi-hat is filtered noise with fast decay
    hat_single = np.random.randn(hat_len) * np.exp(-t * 80) * 0.3

    if pattern == "offbeat":
        # Classic house: hats on every offbeat (the 'and')
        pos = samples_per_eighth  # start on first offbeat
        while pos + hat_len < total_samples:
            end = min(pos + hat_len, total_samples)
            output[pos:end] += hat_single[:end - pos]
            pos += samples_per_eighth * 2
    elif pattern == "driving":
        # Every eighth note
        pos = 0
        i = 0
        while pos + hat_len < total_samples:
            velocity = 0.7 if i % 2 == 0 else 1.0  # accent offbeats
            end = min(pos + hat_len, total_samples)
            output[pos:end] += hat_single[:end - pos] * velocity
            pos += samples_per_eighth
            i += 1
    elif pattern == "shuffled":
        # Swung eighths — late offbeats for groove
        pos = 0
        on_beat = True
        while pos + hat_len < total_samples:
            end = min(pos + hat_len, total_samples)
            velocity = 0.6 if on_beat else 1.0
            output[pos:end] += hat_single[:end - pos] * velocity
            if on_beat:
                pos += int(samples_per_eighth * 1.33)  # swing ratio
            else:
                pos += int(samples_per_eighth * 0.67)
            on_beat = not on_beat

    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * 0.3

    return output


def chop_vocal(input_file: str, num_slices: int = 8, output_dir: str = None) -> list:
    """Chop a vocal recording into rhythmic slices for house production."""
    path = Path(input_file)
    if not path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mono

    total_samples = len(audio)
    slice_len = total_samples // num_slices

    if output_dir is None:
        output_dir = f"{path.stem}_chops"
    Path(output_dir).mkdir(exist_ok=True)

    chop_files = []
    print(f"Chopping {path.name} into {num_slices} slices...")

    for i in range(num_slices):
        start = i * slice_len
        end = min(start + slice_len, total_samples)
        chunk = audio[start:end]

        # Apply short fade in/out to prevent clicks
        fade_len = min(int(sr * 0.005), len(chunk) // 4)
        if fade_len > 0:
            chunk[:fade_len] *= np.linspace(0, 1, fade_len)
            chunk[-fade_len:] *= np.linspace(1, 0, fade_len)

        out_path = str(Path(output_dir) / f"chop_{i+1:02d}.wav")
        sf.write(out_path, chunk, sr)
        chop_files.append(out_path)
        duration_ms = len(chunk) / sr * 1000
        print(f"  chop_{i+1:02d}.wav  ({duration_ms:.0f}ms)")

    print(f"{num_slices} chops saved to: {output_dir}/")
    return chop_files


def generate_beat(bpm: float, duration_sec: float, style: str, output_file: str = None, sr: int = 44100):
    """Generate a basic house beat (kick + hats)."""
    style_info = STYLES.get(style)
    if not style_info:
        print(f"Unknown style: {style}. Options: {', '.join(STYLES.keys())}")
        sys.exit(1)

    print(f"Generating {style} beat at {bpm} BPM ({duration_sec}s)...")

    kick = generate_kick(bpm, duration_sec, sr, style_info["kick_character"])
    hats = generate_hihat(bpm, duration_sec, sr, style_info["hat_pattern"])

    # Mix
    mixed = kick + hats
    mixed = mixed / np.max(np.abs(mixed)) * 0.9

    if output_file is None:
        output_file = f"beat_{style}_{int(bpm)}bpm.wav"

    sf.write(output_file, mixed, sr)
    print(f"Saved: {output_file} ({style}, {int(bpm)} BPM)")
    return output_file


def print_structure(style: str, bpm: float = None):
    """Print the arrangement structure for a house sub-genre."""
    style_info = STYLES.get(style)
    if not style_info:
        print(f"Unknown style: {style}. Options: {', '.join(STYLES.keys())}")
        sys.exit(1)

    if bpm is None:
        bpm = style_info["default_bpm"]

    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * 4  # 4 beats per bar

    print(f"\n{'=' * 65}")
    print(f"  {style.upper()} ARRANGEMENT @ {bpm} BPM")
    print(f"  Reference: {style_info['reference_artists']}")
    print(f"  Key: {style_info['key_suggestion']}")
    print(f"{'=' * 65}")

    total_bars = 0
    total_sec = 0

    print(f"\n  {'Section':<12} {'Bars':>5} {'Time':>8}  Description")
    print(f"  {'-'*12} {'-'*5} {'-'*8}  {'-'*30}")

    for section_name, bars, description in style_info["structure"]:
        sec = bars * bar_sec
        timestamp = f"{int(total_sec // 60)}:{int(total_sec % 60):02d}"
        print(f"  {section_name.upper():<12} {bars:>5} {timestamp:>8}  {description}")
        total_bars += bars
        total_sec += sec

    print(f"\n  TOTAL: {total_bars} bars = {int(total_sec // 60)}:{int(total_sec % 60):02d}")
    print(f"{'=' * 65}")


def cmd_full(bpm: float, style: str, vocal_file: str = None, sr: int = 44100):
    """Generate a full demo: beat + vocal chops if provided."""
    style_info = STYLES.get(style)
    if not style_info:
        print(f"Unknown style: {style}. Options: {', '.join(STYLES.keys())}")
        sys.exit(1)

    # Print structure
    print_structure(style, bpm)

    # Generate beat (30 seconds for demo)
    print()
    beat_file = generate_beat(bpm, 30, style, sr=sr)

    # Chop vocal if provided
    if vocal_file:
        print()
        chop_vocal(vocal_file, num_slices=8)

    print(f"\nDone. Beat: {beat_file}")


def main():
    parser = argparse.ArgumentParser(
        description="House music production toolkit"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Structure
    sub_struct = subparsers.add_parser("structure", help="Print arrangement structure")
    sub_struct.add_argument("--style", "-s", default="tech-house", choices=list(STYLES.keys()))
    sub_struct.add_argument("--bpm", "-b", type=float, default=None)

    # Kick/beat generation
    sub_kick = subparsers.add_parser("kick", help="Generate kick + hat pattern")
    sub_kick.add_argument("--bpm", "-b", type=float, default=126)
    sub_kick.add_argument("--duration", "-d", type=float, default=30, help="Duration in seconds")
    sub_kick.add_argument("--style", "-s", default="tech-house", choices=list(STYLES.keys()))
    sub_kick.add_argument("--output", "-o", default=None)

    # Vocal chop
    sub_chop = subparsers.add_parser("chop", help="Chop vocals into rhythmic slices")
    sub_chop.add_argument("input", help="Vocal file to chop")
    sub_chop.add_argument("--slices", "-n", type=int, default=8)
    sub_chop.add_argument("--output", "-o", default=None, help="Output directory")

    # Full demo
    sub_full = subparsers.add_parser("full", help="Full demo: structure + beat + chops")
    sub_full.add_argument("--bpm", "-b", type=float, default=126)
    sub_full.add_argument("--style", "-s", default="tech-house", choices=list(STYLES.keys()))
    sub_full.add_argument("--vocal", "-v", default=None, help="Vocal file to chop")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "structure":
        print_structure(args.style, args.bpm)
    elif args.command == "kick":
        generate_beat(args.bpm, args.duration, args.style, args.output)
    elif args.command == "chop":
        chop_vocal(args.input, args.slices, args.output)
    elif args.command == "full":
        cmd_full(args.bpm, args.style, args.vocal)


if __name__ == "__main__":
    main()
