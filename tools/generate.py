"""
OS Music Pipeline -- AI Music Generation
Generate tracks via Suno API, Google Lyria 3, or local synthesis.
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Required: pip install numpy soundfile")
    sys.exit(1)

# Import shared synthesis from house.py (single source of truth for kick/hat)
from tools.house import generate_kick, generate_hihat


def synth_clap(sr, bpm, duration_sec):
    total = int(sr * duration_sec)
    output = np.zeros(total)
    beat = int(sr * 60.0 / bpm)
    clap_len = int(sr * 0.08)
    t = np.linspace(0, 0.08, clap_len, endpoint=False)

    # Clap = noise burst with bandpass character
    noise = np.random.randn(clap_len)
    env = np.exp(-t * 40) * 0.4
    # Double hit (characteristic clap flutter)
    flutter = np.exp(-t * 80) * 0.2
    clap = noise * (env + flutter)

    # Place on beats 2 and 4
    pos = beat  # start on beat 2
    while pos + clap_len < total:
        output[pos:pos + clap_len] += clap
        pos += beat * 2
    return output


def synth_bass(sr, bpm, duration_sec, root_hz=55.0, pattern="rolling"):
    """Synthesize a bass line. Root frequency in Hz (e.g., 55 = A1)."""
    total = int(sr * duration_sec)
    output = np.zeros(total)
    beat = int(sr * 60.0 / bpm)
    sixteenth = beat // 4

    # Bass note duration
    note_len = int(sr * 0.15)
    t_note = np.linspace(0, 0.15, note_len, endpoint=False)

    # Simple note intervals relative to root (minor scale pattern)
    if pattern == "rolling":
        # Classic tech house rolling bass: root, root, 5th, root pattern per bar
        intervals = [1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.5, 1.0,
                     1.0, 1.189, 1.0, 1.0, 1.5, 1.0, 1.189, 1.0]
    else:
        intervals = [1.0] * 16

    bar_len = beat * 4
    bar_pos = 0

    while bar_pos < total:
        for i, interval in enumerate(intervals):
            pos = bar_pos + i * sixteenth
            if pos + note_len >= total:
                break

            freq = root_hz * interval
            # Square-ish wave (fundamental + odd harmonics) for bass character
            note = (np.sin(2 * np.pi * freq * t_note) * 0.7 +
                    np.sin(2 * np.pi * freq * 3 * t_note) * 0.15 +
                    np.sin(2 * np.pi * freq * 5 * t_note) * 0.05)

            # Envelope
            env = np.exp(-t_note * 8)
            note = note * env * 0.5

            end = min(pos + note_len, total)
            output[pos:end] += note[:end - pos]

        bar_pos += bar_len

    return output


def synth_pad(sr, bpm, duration_sec, root_hz=220.0):
    """Synthesize a warm pad/chord. Very simple: root + minor 3rd + 5th."""
    total = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, total, endpoint=False)

    # Minor chord: root, minor 3rd (1.189), 5th (1.498)
    chord = (np.sin(2 * np.pi * root_hz * t) * 0.15 +
             np.sin(2 * np.pi * root_hz * 1.189 * t) * 0.12 +
             np.sin(2 * np.pi * root_hz * 1.498 * t) * 0.10)

    # Slow filter sweep (simulates low-pass opening)
    sweep = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t))
    chord = chord * sweep

    # Gentle amplitude modulation
    tremolo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
    chord = chord * tremolo

    return chord


def generate_local_track(style: str, bpm: int, duration_sec: int, output: str = None):
    """Generate a complete house track locally using synthesis."""
    sr = 44100

    print(f"Generating {style} track at {bpm} BPM ({duration_sec}s)...")

    # Frequency mapping for common house keys
    key_freqs = {
        "tech-house": (55.0, 220.0),      # A (bass A1, pad A3)
        "deep-house": (58.27, 233.08),     # Bb
        "bass-house": (49.0, 196.0),       # G
        "melodic-house": (55.0, 220.0),    # A
    }
    bass_root, pad_root = key_freqs.get(style, (55.0, 220.0))

    kick_char = {"tech-house": "punchy", "deep-house": "warm", "bass-house": "hard"}.get(style, "punchy")
    hat_pattern = {"tech-house": "offbeat", "deep-house": "shuffled", "bass-house": "driving"}.get(style, "offbeat")

    # Generate each element (kick/hat from house.py, bass/clap/pad local)
    kick = generate_kick(bpm, duration_sec, sr, kick_char)
    hats = generate_hihat(bpm, duration_sec, sr, hat_pattern)
    clap = synth_clap(sr, bpm, duration_sec)
    bass = synth_bass(sr, bpm, duration_sec, bass_root, "rolling")
    pad = synth_pad(sr, bpm, duration_sec, pad_root)
    hats16 = generate_hihat(bpm, duration_sec, sr, "driving")

    # === ARRANGEMENT ===
    # Create energy envelope: intro builds, drop full, break strips, drop2 full, outro fades
    bar_sec = 60.0 / bpm * 4
    total_bars = int(duration_sec / bar_sec)

    arrangement = np.ones(int(sr * duration_sec))
    bass_arr = np.ones(int(sr * duration_sec))
    pad_arr = np.ones(int(sr * duration_sec))

    # Intro: first 25% — no bass, pad fades in
    intro_end = int(sr * duration_sec * 0.2)
    bass_arr[:intro_end] = 0
    pad_arr[:intro_end] = np.linspace(0, 1, intro_end)

    # Build: 20-30% — bass fades in
    build_start = intro_end
    build_end = int(sr * duration_sec * 0.3)
    bass_arr[build_start:build_end] = np.linspace(0, 1, build_end - build_start)

    # Break: 55-65% — strip to kick + pad
    break_start = int(sr * duration_sec * 0.55)
    break_end = int(sr * duration_sec * 0.65)
    bass_arr[break_start:break_end] = np.linspace(1, 0.2, break_end - break_start)
    arrangement[break_start:break_end] = 0.6  # reduce overall energy

    # Outro: last 15% — fade out
    outro_start = int(sr * duration_sec * 0.85)
    fade = np.linspace(1, 0, int(sr * duration_sec) - outro_start)
    arrangement[outro_start:] = fade
    bass_arr[outro_start:] *= fade

    # Mix with arrangement
    mixed = (kick * 0.9 +
             hats * 0.4 * arrangement +
             hats16 * 0.15 * arrangement +
             clap * 0.5 * arrangement +
             bass * bass_arr +
             pad * pad_arr * arrangement)

    # Normalize
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.9

    if output is None:
        output = f"generated_{style}_{bpm}bpm.wav"

    sf.write(output, mixed, sr)

    print(f"  Output: {output}")
    return output


# ============================================================
# SUNO API INTEGRATION
# ============================================================

def generate_suno(prompt: str, count: int = 2, instrumental: bool = True,
                  output_dir: str = "generated"):
    """Generate tracks using Suno API (requires SUNO_COOKIE)."""
    cookie = os.environ.get("SUNO_COOKIE")
    if not cookie:
        print("Error: SUNO_COOKIE not set. Export it from your Suno session.")
        sys.exit(1)

    try:
        from suno import Suno
    except ImportError:
        print("SunoAI not installed. Run: pip install SunoAI")
        sys.exit(1)

    Path(output_dir).mkdir(exist_ok=True)

    print(f"Generating {count} tracks with Suno...")

    client = Suno(cookie=cookie)

    songs = client.generate(
        prompt=prompt,
        is_custom=False,
        wait_audio=True,
    )

    downloaded = []
    for i, song in enumerate(songs):
        filepath = client.download(song=song, path=output_dir)
        print(f"  [{i+1}/{len(songs)}] Downloaded: {filepath}")
        downloaded.append(str(filepath))

    print(f"{len(downloaded)} tracks saved to: {output_dir}/")
    return downloaded


# ============================================================
# GOOGLE LYRIA 3 INTEGRATION
# ============================================================

def generate_lyria(prompt: str, output: str = None):
    """Generate a track using Google Lyria 3 (requires GOOGLE_API_KEY)."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set. Get one at https://aistudio.google.com/apikey")
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        print("google-genai not installed. Run: pip install google-genai")
        sys.exit(1)

    if output is None:
        safe_prompt = prompt.replace(" ", "_")[:30]
        output = f"lyria_{safe_prompt}.mp3"

    print(f"Generating with Google Lyria 3...")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="lyria-3-clip-preview",
        contents=prompt,
    )

    for part in response.candidates[0].content.parts:
        if hasattr(part, 'inline_data') and part.inline_data:
            with open(output, "wb") as f:
                f.write(part.inline_data.data)
            print(f"  Saved: {output}")
            return output

    print("Error: No audio data in response")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate music via Suno, Google Lyria 3, or local synthesis"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--suno", type=str, help="Generate with Suno (prompt)")
    group.add_argument("--lyria", type=str, help="Generate with Google Lyria 3 (prompt)")
    group.add_argument("--local", action="store_true", help="Generate locally (no API needed)")

    parser.add_argument("--style", "-s", default="tech-house",
                        choices=["tech-house", "deep-house", "bass-house", "melodic-house"])
    parser.add_argument("--bpm", "-b", type=int, default=128)
    parser.add_argument("--duration", "-d", type=int, default=30, help="Duration in seconds (local only)")
    parser.add_argument("--count", "-n", type=int, default=2, help="Number of tracks (Suno only)")
    parser.add_argument("--output", "-o", default=None, help="Output filename")
    parser.add_argument("--pipeline", action="store_true",
                        help="Auto-process through club-ready chain after generating")
    parser.add_argument("--ref", type=str, default=None,
                        help="Reference track for mastering (used with --pipeline)")

    args = parser.parse_args()

    generated_files = []

    if args.suno:
        generated_files = generate_suno(args.suno, args.count)
    elif args.lyria:
        result = generate_lyria(args.lyria, args.output)
        if result:
            generated_files = [result]
    elif args.local:
        result = generate_local_track(args.style, args.bpm, args.duration, args.output)
        generated_files = [result]

    # Auto-pipeline
    if args.pipeline and generated_files:
        from tools.club_ready import process_club_ready
        print("\n--- Running club-ready processing ---\n")
        for f in generated_files:
            if f.endswith(".wav"):
                process_club_ready(f, "club", args.bpm, True, None)

        if args.ref:
            from tools.master import master_track
            print("\n--- Reference mastering ---\n")
            for f in generated_files:
                club_file = f.replace(".wav", "_club_ready.wav")
                if Path(club_file).exists():
                    master_track(club_file, args.ref)


if __name__ == "__main__":
    main()
