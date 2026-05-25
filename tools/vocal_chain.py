"""Vocal processing chain with artist-style presets (Drake, Weeknd, Nav, Travis, Don Toliver)."""

import argparse
import sys
from pathlib import Path

try:
    from pedalboard import (
        Pedalboard,
        Compressor,
        Gain,
        HighpassFilter,
        LowpassFilter,
        LowShelfFilter,
        HighShelfFilter,
        PeakFilter,
        Reverb,
        Delay,
        Limiter,
        Chorus,
    )
    from pedalboard.io import AudioFile
except ImportError:
    print("pedalboard not installed. Run: pip install pedalboard")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy not installed. Run: pip install numpy")
    sys.exit(1)


# === VOCAL PRESETS ===
# Each preset is a function that returns a Pedalboard with the right effects chain

def preset_drake() -> Pedalboard:
    """40's Drake vocal chain: warm, intimate, slightly compressed.
    Subtle pitch correction feel, low-end warmth, controlled dynamics."""
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),      # Remove rumble
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),  # Warmth
        PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5, q=1.0),  # Presence
        PeakFilter(cutoff_frequency_hz=6000, gain_db=-1.5, q=1.0),  # De-harsh
        Compressor(
            threshold_db=-18,
            ratio=4.0,
            attack_ms=10,
            release_ms=100,
        ),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.5),  # Air
        Reverb(
            room_size=0.3,
            damping=0.7,
            wet_level=0.12,
            dry_level=1.0,
        ),
        Delay(
            delay_seconds=0.25,
            feedback=0.15,
            mix=0.08,
        ),
        Gain(gain_db=1.0),
        Limiter(threshold_db=-1.0),
    ])


def preset_weeknd() -> Pedalboard:
    """The Weeknd's dark, ethereal vocal chain.
    Heavy reverb, dark character, wide stereo feel."""
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=90),
        LowShelfFilter(cutoff_frequency_hz=250, gain_db=1.5),
        PeakFilter(cutoff_frequency_hz=2500, gain_db=3.0, q=0.8),  # Forward vocal
        PeakFilter(cutoff_frequency_hz=5000, gain_db=-2.0, q=1.0),  # Dark character
        Compressor(
            threshold_db=-20,
            ratio=3.5,
            attack_ms=15,
            release_ms=150,
        ),
        Reverb(
            room_size=0.65,       # Large dark room
            damping=0.8,          # Dark reverb tail
            wet_level=0.25,       # More wet than Drake
            dry_level=1.0,
        ),
        Delay(
            delay_seconds=0.35,
            feedback=0.25,
            mix=0.12,
        ),
        LowpassFilter(cutoff_frequency_hz=14000),  # Roll off highs for darkness
        Gain(gain_db=1.5),
        Limiter(threshold_db=-1.0),
    ])


def preset_nav() -> Pedalboard:
    """Nav's signature muddy-clean balance.
    Simple chain, not too processed, monotone-friendly."""
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=100),
        PeakFilter(cutoff_frequency_hz=2000, gain_db=2.0, q=1.0),
        PeakFilter(cutoff_frequency_hz=4000, gain_db=1.5, q=1.0),
        Compressor(
            threshold_db=-16,
            ratio=3.0,
            attack_ms=12,
            release_ms=120,
        ),
        Reverb(
            room_size=0.35,
            damping=0.6,
            wet_level=0.15,
            dry_level=1.0,
        ),
        Delay(
            delay_seconds=0.2,
            feedback=0.1,
            mix=0.06,
        ),
        Gain(gain_db=2.0),
        Limiter(threshold_db=-1.0),
    ])


def preset_travis() -> Pedalboard:
    """Travis Scott's psychedelic trap vocal.
    Heavy effects, aggressive compression, lots of reverb and delay."""
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        PeakFilter(cutoff_frequency_hz=1500, gain_db=3.0, q=0.7),   # Midrange aggression
        PeakFilter(cutoff_frequency_hz=4000, gain_db=2.0, q=1.0),   # Cut through
        Compressor(
            threshold_db=-22,
            ratio=6.0,          # Heavy compression
            attack_ms=5,
            release_ms=80,
        ),
        Reverb(
            room_size=0.75,      # Large space
            damping=0.5,         # Bright reverb
            wet_level=0.30,      # Very wet
            dry_level=1.0,
        ),
        Delay(
            delay_seconds=0.4,
            feedback=0.35,       # Long feedback
            mix=0.18,
        ),
        Chorus(
            rate_hz=0.5,
            depth=0.3,
            mix=0.1,
        ),
        Gain(gain_db=2.0),
        Limiter(threshold_db=-0.5),
    ])


def preset_don_toliver() -> Pedalboard:
    """Don Toliver's psychedelic melodic style.
    Lush reverb, pitch-shifted feel, dreamy character."""
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=85),
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.5),  # Warm low end
        PeakFilter(cutoff_frequency_hz=2800, gain_db=3.5, q=0.8),  # Vocal presence
        PeakFilter(cutoff_frequency_hz=7000, gain_db=-1.0, q=1.0),
        Compressor(
            threshold_db=-20,
            ratio=4.5,
            attack_ms=8,
            release_ms=100,
        ),
        Reverb(
            room_size=0.55,
            damping=0.6,
            wet_level=0.22,
            dry_level=1.0,
        ),
        Delay(
            delay_seconds=0.3,
            feedback=0.3,
            mix=0.15,
        ),
        Chorus(
            rate_hz=0.3,
            depth=0.2,
            mix=0.08,
        ),
        HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.0),
        Gain(gain_db=1.5),
        Limiter(threshold_db=-1.0),
    ])


def preset_raw() -> Pedalboard:
    """Minimal processing — just cleanup."""
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(
            threshold_db=-15,
            ratio=2.5,
            attack_ms=15,
            release_ms=150,
        ),
        Gain(gain_db=1.0),
        Limiter(threshold_db=-1.0),
    ])


PRESETS = {
    "drake": ("Drake (40's warm intimate chain)", preset_drake),
    "weeknd": ("The Weeknd (dark ethereal XO chain)", preset_weeknd),
    "nav": ("Nav (clean monotone-friendly chain)", preset_nav),
    "travis": ("Travis Scott (psychedelic trap chain)", preset_travis),
    "don-toliver": ("Don Toliver (dreamy psychedelic chain)", preset_don_toliver),
    "raw": ("Raw (minimal cleanup only)", preset_raw),
}


def process_vocals(input_file: str, preset_name: str, output_file: str = None):
    path = Path(input_file)
    if not path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    if preset_name not in PRESETS:
        print(f"Error: Unknown preset '{preset_name}'")
        print(f"Available presets: {', '.join(PRESETS.keys())}")
        sys.exit(1)

    preset_desc, preset_fn = PRESETS[preset_name]
    board = preset_fn()

    if output_file is None:
        output_file = f"{path.stem}_{preset_name}{path.suffix}"

    print(f"Processing: {path.name}")
    print(f"Preset: {preset_desc}")
    print(f"Chain: {' > '.join(type(effect).__name__ for effect in board)}")
    print()

    with AudioFile(str(path)) as f:
        sr = f.samplerate
        channels = f.num_channels
        audio = f.read(f.frames)

    print(f"  Input: {channels}ch, {sr}Hz, {audio.shape[1]/sr:.1f}s")

    # Process
    processed = board(audio, sr)

    # Write output
    with AudioFile(output_file, "w", sr, channels) as f:
        f.write(processed)

    print(f"  Output: {output_file}")


def process_all_presets(input_file: str):
    """Process the same vocal through all presets for comparison."""
    path = Path(input_file)
    if not path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    print(f"Processing {path.name} through ALL presets for comparison...")
    print()

    for name in PRESETS:
        output = f"{path.stem}_{name}{path.suffix}"
        process_vocals(input_file, name, output)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Process vocals with artist-style presets (Drake, Weeknd, Nav, Travis, Don Toliver)"
    )
    parser.add_argument("input", help="Vocal audio file (.wav)")
    parser.add_argument("--preset", "-p", default="drake",
                        choices=list(PRESETS.keys()),
                        help="Vocal style preset")
    parser.add_argument("--output", "-o", default=None, help="Output filename")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Process through ALL presets for comparison")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available presets")

    args = parser.parse_args()

    if args.list:
        print("Available vocal presets:")
        for name, (desc, _) in PRESETS.items():
            print(f"  {name:<15} {desc}")
        return

    if args.all:
        process_all_presets(args.input)
    else:
        process_vocals(args.input, args.preset, args.output)


if __name__ == "__main__":
    main()
