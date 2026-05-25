"""
OS Music Pipeline — STUDIO
The future of music production: one conversation, one track.

This is what music production looks like in 2031, built today.
No DAW menus. No plugin chains. No manual processing. You describe, it produces.

Usage:
    python studio.py "dark tech house like John Summit, 128 BPM"
    python studio.py "afrobeats banger, Rema style, 108 BPM, party anthem"
    python studio.py "moody R&B, Weeknd vibes, about driving at 2AM"
    python studio.py --interactive    # Conversational mode

What happens:
    1. Parses your intent (genre, style, mood, BPM, key)
    2. Generates optimized prompts for Suno/Udio
    3. Generates track (Suno API / local synthesis fallback)
    4. Analyzes the output
    5. Applies style fingerprint if available
    6. Processes through club-ready chain
    7. Masters against reference profile
    8. Outputs a release-ready file

One command. Raw idea to finished track.
"""

import argparse
import os
import re
import sys
import json
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Required: pip install numpy")
    sys.exit(1)

# Import our tools
sys.path.insert(0, str(Path(__file__).parent))
from tools.prompts import GENRES, get_prompt
from tools.analyze import analyze_track
from tools.club_ready import process_club_ready
from tools.generate import generate_local_track


def parse_intent(description: str) -> dict:
    """Parse a natural language description into production parameters."""
    desc_lower = description.lower()

    # Detect genre
    genre = "tech-house"  # default
    genre_keywords = {
        "tech house": "tech-house",
        "tech-house": "tech-house",
        "deep house": "deep-house",
        "melodic house": "melodic-house",
        "progressive house": "melodic-house",
        "bass house": "bass-house",
        "afrobeats": "afrobeats",
        "afro beats": "afrobeats",
        "afro-beats": "afrobeats",
        "afropop": "afrobeats",
        "amapiano": "amapiano",
        "r&b": "deep-house",  # closest processing match
        "rnb": "deep-house",
    }
    for keyword, g in genre_keywords.items():
        if keyword in desc_lower:
            genre = g
            break

    # Detect BPM
    bpm = None
    bpm_match = re.search(r'(\d{2,3})\s*bpm', desc_lower)
    if bpm_match:
        bpm = int(bpm_match.group(1))
        if bpm < 60 or bpm > 200:
            bpm = None

    # Detect key
    key = None
    key_match = re.search(r'\b([A-G][#b]?)\s*(minor|major|min|maj)?\b', description)
    if key_match:
        key = key_match.group(1)
        if key_match.group(2):
            mode = "minor" if key_match.group(2).startswith("min") else "major"
            key = f"{key} {mode}"

    # Detect mood/style references
    style_refs = []
    artist_keywords = {
        "summit": "john-summit",
        "john summit": "john-summit",
        "fisher": "fisher",
        "chris lake": "chris-lake",
        "dom dolla": "dom-dolla",
        "meduza": "meduza",
        "disclosure": "disclosure",
        "rema": "rema",
        "burna": "burna-boy",
        "wizkid": "wizkid",
        "weeknd": "weeknd",
        "drake": "drake",
        "nav": "nav",
        "travis": "travis-scott",
        "don toliver": "don-toliver",
    }
    for keyword, ref in artist_keywords.items():
        if keyword in desc_lower:
            style_refs.append(ref)

    # Detect vocal preference
    vocal = False
    if any(w in desc_lower for w in ["vocal", "singing", "lyrics", "words", "hook"]):
        vocal = True

    # Detect mood
    mood = "energetic"
    mood_keywords = {
        "dark": "dark",
        "moody": "dark",
        "aggressive": "aggressive",
        "chill": "chill",
        "uplifting": "uplifting",
        "euphoric": "uplifting",
        "party": "energetic",
        "banger": "aggressive",
        "dreamy": "chill",
        "melancholic": "dark",
        "driving": "energetic",
        "peak time": "aggressive",
    }
    for keyword, m in mood_keywords.items():
        if keyword in desc_lower:
            mood = m
            break

    # Detect target
    target = "club"
    if any(w in desc_lower for w in ["spotify", "streaming", "playlist"]):
        target = "streaming"

    return {
        "genre": genre,
        "bpm": bpm,
        "key": key,
        "vocal": vocal,
        "mood": mood,
        "style_refs": style_refs,
        "target": target,
        "raw_description": description,
    }


def run_studio(description: str, output_dir: str = "output"):
    """Full studio session: intent > generate > process > master."""

    print()
    print("=" * 70)
    print("  OS STUDIO")
    print("  One description. One finished track.")
    print("=" * 70)
    print()

    # Step 1: Parse intent
    print("[1/6] PARSING INTENT...")
    intent = parse_intent(description)
    print(f"  Genre:      {intent['genre']}")
    print(f"  BPM:        {intent['bpm'] or 'auto'}")
    print(f"  Key:        {intent['key'] or 'auto'}")
    print(f"  Mood:       {intent['mood']}")
    print(f"  Vocal:      {'yes' if intent['vocal'] else 'no'}")
    print(f"  References: {', '.join(intent['style_refs']) or 'none'}")
    print(f"  Target:     {intent['target']}")
    print()

    # Step 2: Generate prompt
    print("[2/6] GENERATING PROMPT...")
    prompt_result = get_prompt(
        intent["genre"],
        "suno",
        intent["vocal"],
        intent["bpm"],
    )
    print(f"  Suno prompt: {prompt_result['style_prompt'][:80]}...")
    print(f"  BPM: {prompt_result['bpm']} | Key: {prompt_result['key']}")
    print()

    # Step 3: Generate track
    print("[3/6] GENERATING TRACK...")
    Path(output_dir).mkdir(exist_ok=True)

    suno_cookie = os.environ.get("SUNO_COOKIE")
    google_key = os.environ.get("GOOGLE_API_KEY")

    generated_file = None

    if suno_cookie:
        print("  Using Suno API...")
        try:
            from tools.generate import generate_suno
            files = generate_suno(prompt_result["style_prompt"], count=2, output_dir=output_dir)
            if files:
                generated_file = files[0]
        except Exception as e:
            print(f"  Suno failed: {e}")
            print("  Falling back to local synthesis...")

    if not generated_file and google_key:
        print("  Using Google Lyria 3...")
        try:
            from tools.generate import generate_lyria
            generated_file = generate_lyria(
                prompt_result["style_prompt"],
                output=str(Path(output_dir) / "lyria_output.mp3")
            )
        except Exception as e:
            print(f"  Lyria failed: {e}")
            print("  Falling back to local synthesis...")

    if not generated_file:
        print("  Using local synthesis (no API keys found)...")
        generated_file = generate_local_track(
            intent["genre"],
            prompt_result["bpm"],
            60,  # 60 seconds
            str(Path(output_dir) / f"studio_{intent['genre']}_{prompt_result['bpm']}bpm.wav")
        )

    print()

    # Step 4: Analyze
    print("[4/6] ANALYZING...")
    analysis = analyze_track(generated_file, plot=False)
    print()

    # Step 5: Apply fingerprint if available
    fingerprint_dir = Path(__file__).parent / "fingerprints"
    applied_file = generated_file
    if intent["style_refs"]:
        for ref in intent["style_refs"]:
            fp_path = fingerprint_dir / f"{ref}.json"
            if fp_path.exists():
                print(f"[5/6] APPLYING FINGERPRINT: {ref}...")
                from tools.fingerprint import apply_fingerprint
                applied_file = apply_fingerprint(
                    generated_file, ref,
                    str(Path(output_dir) / f"studio_fingerprinted.wav")
                )
                print()
                break
        else:
            print(f"[5/6] No fingerprint found for {intent['style_refs']}. Skipping.")
            print(f"       Build one: python tools/fingerprint.py build tracks_folder/ --name {intent['style_refs'][0]}")
            print()
    else:
        print("[5/6] No style reference. Skipping fingerprint.")
        print()

    # Step 6: Club-ready processing + mastering
    print("[6/6] CLUB-READY PROCESSING...")

    # Only process .wav files
    if applied_file.endswith(".wav"):
        final_file = str(Path(output_dir) / "studio_final.wav")
        process_club_ready(
            applied_file,
            target=intent["target"],
            bpm=float(prompt_result["bpm"]),
            sidechain=True,
            output_file=final_file,
        )
    else:
        final_file = applied_file
        print(f"  Non-WAV file ({applied_file}), skipping processing.")
        print(f"  Convert to WAV first for full processing.")

    # Final analysis
    print()
    if Path(final_file).exists() and final_file.endswith(".wav"):
        print("FINAL ANALYSIS:")
        final_analysis = analyze_track(final_file, plot=False)

    print()
    print("=" * 70)
    print("  SESSION COMPLETE")
    print("=" * 70)
    print(f"  Input:  \"{description}\"")
    print(f"  Output: {final_file}")
    print(f"  Genre:  {intent['genre']} | BPM: {prompt_result['bpm']} | Key: {prompt_result['key']}")
    print(f"  Target: {intent['target']}")
    print()
    print("  To iterate:")
    print(f"    python studio.py \"{description}, but with more bass\"")
    print(f"    python studio.py \"{description}, darker, less reverb\"")
    print()
    print("  To release:")
    print(f"    Upload {final_file} to DistroKid ($22/year)")
    print("=" * 70)

    return final_file


def main():
    parser = argparse.ArgumentParser(
        description="OS Studio: one description, one finished track"
    )
    parser.add_argument("description", nargs="?", default=None,
                        help="Describe the track you want")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory")

    args = parser.parse_args()

    if args.description is None:
        print()
        print("OS STUDIO - The future of music production")
        print()
        print("Usage:")
        print('  python studio.py "dark tech house like John Summit, 128 BPM"')
        print('  python studio.py "afrobeats banger, Rema style, party anthem"')
        print('  python studio.py "moody R&B, Weeknd vibes, about driving at 2AM"')
        print('  python studio.py "deep house, chill, 122 BPM, A minor"')
        print('  python studio.py "bass house, aggressive, festival drop, 130 BPM"')
        print()
        print("What happens: parse intent > generate > analyze > fingerprint > process > master")
        print("One command. Raw idea to finished track.")
        return

    run_studio(args.description, args.output)


if __name__ == "__main__":
    main()
