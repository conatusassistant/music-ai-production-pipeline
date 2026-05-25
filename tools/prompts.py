"""AI generation prompt library for Suno/Udio, tuned from Beatport Top 100 data."""

import argparse
import random
import sys


# Beatport Top 100 Tech House analysis (May 2026):
# BPM range: 127-132, clustering at 128-130
# Common keys: B minor, D minor, Ab major, A major, E major
# 55% contain obvious samples
# Structure: ABAB or BCB most common
# Nearly all tracks are collabs (2-4 artists)

GENRES = {
    "tech-house": {
        "name": "Tech House",
        "reference": "John Summit, Fisher, Chris Lake, Dom Dolla, Meduza",
        "beatport_data": "Currently #1 genre on Beatport. BPM 127-132.",
        "bpm_range": (127, 132),
        "keys": ["B minor", "D minor", "F# minor", "Ab major", "A minor", "E major"],
        "suno_base": "tech house, punchy four-on-the-floor kick, rolling bassline, tight sidechain compression, percussive groove, peak time energy, modern production",
        "suno_instrumental": "tech house, punchy kicks, rolling bassline, percussive groove, peak time energy, tight production, dancefloor focused, {bpm} BPM, no vocals, no orchestral, no guitar",
        "suno_vocal": "tech house, punchy kicks, rolling bassline, catchy vocal hook, peak time energy, tight production, dancefloor focused, {bpm} BPM, no orchestral, no guitar",
        "suno_exclude": "guitar, orchestral, autotune, acoustic",
        "udio": "Driving tech house track with a punchy four-on-the-floor kick, rolling percussive groove, filtered synth stabs, tight sidechain compression, peak-time club energy, {bpm} BPM, {key}",
        "structure_tags": "[Intro]\n(8 bars, kick + hats building)\n\n[Build]\n(filter opening, bass enters)\n\n[Drop]\n(full energy, main hook)\n\n[Break]\n(stripped back, tension)\n\n[Build]\n(riser, snare roll)\n\n[Drop]\n(full energy, variation)\n\n[Outro]\n(kick + hats fading)",
        "pro_tips": [
            "Use 'tight low end' and 'punchy drums' in prompt — Suno v5.5 responds to production descriptors",
            "Add 'wide stereo field' for spatial depth",
            "For John Summit grit: add 'saturated', 'gritty bass', 'analog warmth'",
            "Fisher style: add 'vocal sample chops', 'festival energy', 'singalong hook'",
            "Generate 10-20 variations, pick the best 2-3",
            "Export stems from Suno Studio, rebuild arrangement in FL Studio",
        ],
    },
    "deep-house": {
        "name": "Deep House",
        "reference": "Disclosure, Lane 8, Ben Bohmer, Rufus Du Sol",
        "beatport_data": "Declining in chart relevance but loyal niche audience.",
        "bpm_range": (120, 124),
        "keys": ["A minor", "C minor", "D minor", "F minor"],
        "suno_base": "deep house, warm analog pads, four-on-the-floor, smooth bassline, late night atmosphere",
        "suno_instrumental": "deep house, four-on-the-floor, warm bassline, subtle vocal chops, late night club, groovy and hypnotic, smooth and rolling, {bpm} BPM, no guitar, no orchestral",
        "suno_vocal": "deep house, warm pads, groovy bassline, smooth male vocals, late night atmosphere, intimate and hypnotic, {bpm} BPM, no guitar, no orchestral",
        "suno_exclude": "guitar, orchestral, aggressive, distorted",
        "udio": "Deep atmospheric house, warm analog pads, subtle vocal chops, hypnotic bassline, late-night underground club vibe, minimal and clean mix, {bpm} BPM, {key}",
        "structure_tags": "[Intro]\n(atmospheric pads, kick enters at bar 8)\n\n[Verse]\n(bass + chords, vocal phrase)\n\n[Build]\n(filter sweep, rising tension)\n\n[Drop]\n(full groove, controlled energy)\n\n[Break]\n(melodic moment, stripped)\n\n[Drop]\n(groove with variation)\n\n[Outro]\n(elements fade)",
        "pro_tips": [
            "Use 'warm', 'analog', 'organic' — deep house is about texture not aggression",
            "Add 'jazzy chords' or 'soulful' for Disclosure vibes",
            "Lane 8 style: add 'melodic', 'progressive', 'building energy'",
            "Keep it minimal — 'spacious mix', 'less is more', 'room to breathe'",
        ],
    },
    "melodic-house": {
        "name": "Melodic House / Progressive",
        "reference": "Meduza, Vintage Culture, Anyma, Artbat",
        "beatport_data": "Meduza's pop-house model generates highest streams (1.5B per track).",
        "bpm_range": (122, 126),
        "keys": ["Bb minor", "A minor", "C minor", "G minor"],
        "suno_base": "melodic house, euphoric energy, building tension, lush pads, emotional progression",
        "suno_instrumental": "progressive house, building energy, tension and release, euphoric drop, anthemic, risers, building percussion, lush pads, {bpm} BPM, no vocals",
        "suno_vocal": "melodic house, uplifting sunrise vibe, ethereal female vocals, {bpm} BPM, rolling bassline, lush pads, euphoric festival energy, warm production",
        "suno_exclude": "guitar, acoustic, dark, aggressive",
        "udio": "Melodic progressive house with building energy, euphoric breakdown, emotional synth leads, lush pads, festival-ready drop, {bpm} BPM, {key}",
        "structure_tags": "[Intro]\n(ambient textures, slow build)\n\n[Verse]\n(vocal enters, chords build)\n\n[Pre-Chorus]\n(rising energy, filter opens)\n\n[Chorus]\n(euphoric drop, full energy)\n\n[Break]\n(stripped to vocal + pad, emotional)\n\n[Build]\n(maximum tension)\n\n[Chorus]\n(biggest drop)\n\n[Outro]\n(fade to ambient)",
        "pro_tips": [
            "Meduza formula: catchy vocal topline + house production = streaming gold",
            "The drop should feel 'euphoric' and 'uplifting' — use those words",
            "Add 'radio-ready mix' for commercial sound",
            "'Piece of Your Heart' is 124 BPM, Bb minor — use as reference",
        ],
    },
    "afrobeats": {
        "name": "Afrobeats",
        "reference": "Burna Boy, Wizkid, Rema, Ayra Starr, Tems",
        "beatport_data": "Rema 'Calm Down' = 2B+ streams. Wizkid = 11B+ total streams.",
        "bpm_range": (100, 120),
        "keys": ["B major", "C major", "D minor", "Eb minor"],
        "suno_base": "afrobeats, layered percussion, infectious groove, melodic vocals, West African rhythm",
        "suno_instrumental": "afrobeats, layered percussion, shakers, talking drum, syncopated bass, infectious groove, warm production, {bpm} BPM, no orchestral",
        "suno_vocal": "afrobeats, infectious groove, smooth male vocals, call-and-response chorus, layered percussion, afropop melody, warm and bright, {bpm} BPM",
        "suno_exclude": "orchestral, EDM drop, distorted, metal",
        "udio": "Infectious Afrobeats track with layered percussion, talking drums, syncopated bass groove, melodic vocal hook, call-and-response chorus, warm bright production, {bpm} BPM, {key}",
        "structure_tags": "[Intro]\n(percussion intro, guitar lick)\n\n[Verse 1]\n(melodic vocals, groove establishes)\n\n[Pre-Chorus]\n(energy builds)\n\n[Chorus]\n(catchy hook, call-and-response)\n\n[Verse 2]\n(new melody, same groove)\n\n[Chorus]\n(hook repeats)\n\n[Bridge]\n(breakdown, vocal ad-libs)\n\n[Chorus]\n(final hook, biggest energy)\n\n[Outro]\n(percussion fade)",
        "pro_tips": [
            "Afrobeats is vocal-forward — the melody carries the track, not the production",
            "Use 'polyrhythmic' and 'syncopated' for authentic drum patterns",
            "Rema style: add 'rave-influenced', 'energetic', 'party anthem'",
            "Wizkid style: add 'smooth', 'laid-back', 'romantic', 'sensual'",
            "Burna Boy style: add 'dancehall influence', 'reggae fusion', 'powerful vocals'",
            "'Calm Down' is 107 BPM, B major — use as reference for Rema style",
        ],
    },
    "amapiano": {
        "name": "Amapiano",
        "reference": "Kabza De Small, DJ Maphorisa, Uncle Waffles, Tyler ICU",
        "beatport_data": "Fastest-growing African electronic sub-genre. Crossing over globally.",
        "bpm_range": (108, 115),
        "keys": ["A minor", "C major", "D minor", "F major"],
        "suno_base": "amapiano, log drum, deep piano chords, shakers, South African groove",
        "suno_instrumental": "Amapiano, log drum, deep piano chords, shakers, {bpm} BPM, South African groove, syncopated bass, warm production, jazzy progression",
        "suno_vocal": "Afro-pop Amapiano hybrid, {bpm} BPM, log drum grooves, shaker patterns, bright guitar licks, call-and-response chorus, smooth male vocals",
        "suno_exclude": "EDM, distorted, heavy bass drop, metal",
        "udio": "Amapiano track with signature log drum bass, deep piano chords, shaker percussion, South African groove feel, jazzy harmonies, warm spacious production, {bpm} BPM, {key}",
        "structure_tags": "[Intro]\n(log drum + shakers)\n\n[Verse]\n(vocals enter, piano chords)\n\n[Chorus]\n(melodic hook, full groove)\n\n[Instrumental]\n(log drum solo, percussion builds)\n\n[Verse 2]\n\n[Chorus]\n\n[Bridge]\n(stripped, vocal moment)\n\n[Chorus]\n(final hook)\n\n[Outro]\n(log drum fade)",
        "pro_tips": [
            "The log drum IS the genre — make sure prompt emphasizes it",
            "Add 'jazzy' for harmonic sophistication",
            "Add 'shaker patterns' — the shaker groove is as important as the kick",
            "Keep it warm and spacious — amapiano breathes, it's not compressed like tech house",
        ],
    },
    "bass-house": {
        "name": "Bass House",
        "reference": "Habstrakt, Joyryde, Skrillex (house era), Matroda",
        "beatport_data": "Growing sub-genre. More aggressive than tech house.",
        "bpm_range": (126, 132),
        "keys": ["E minor", "D minor", "G minor", "A minor"],
        "suno_base": "bass house, heavy wobble bass, distorted, aggressive energy, festival ready",
        "suno_instrumental": "bass house, heavy wobble bass, distorted synths, aggressive four-on-the-floor, festival energy, {bpm} BPM, no vocals, no guitar",
        "suno_vocal": "bass house, heavy bass drops, aggressive energy, shouted vocal hook, festival anthem, {bpm} BPM, no guitar",
        "suno_exclude": "acoustic, jazz, soft, ambient",
        "udio": "Aggressive bass house track with heavy wobble bassline, distorted synths, punchy four-on-the-floor kick, festival-ready energy, massive drop, {bpm} BPM, {key}",
        "structure_tags": "[Intro]\n(tension building, kick pattern)\n\n[Build]\n(riser, snare fill, energy climbing)\n\n[Drop]\n(HEAVY bass, distorted, maximum energy)\n\n[Break]\n(vocal sample, silence for impact)\n\n[Build]\n(faster build, more intensity)\n\n[Drop]\n(even heavier variation)\n\n[Outro]\n(kick out, clean)",
        "pro_tips": [
            "Bass house is about the DROP — make the contrast between build and drop extreme",
            "Use 'wobble', 'distorted', 'heavy' — be aggressive with descriptors",
            "Add 'festival energy' and 'mosh pit' for Skrillex-adjacent vibes",
        ],
    },
}


def get_prompt(genre: str, platform: str = "suno", vocal: bool = False,
               bpm: int = None, key: str = None) -> dict:
    """Generate an optimized prompt for the given genre and platform."""
    if genre not in GENRES:
        print(f"Unknown genre: {genre}")
        print(f"Available: {', '.join(GENRES.keys())}")
        sys.exit(1)

    g = GENRES[genre]

    # Pick BPM
    if bpm is None:
        bpm = random.randint(g["bpm_range"][0], g["bpm_range"][1])

    # Pick key
    if key is None:
        key = random.choice(g["keys"])

    result = {
        "genre": g["name"],
        "platform": platform,
        "bpm": bpm,
        "key": key,
        "reference_artists": g["reference"],
    }

    if platform == "suno":
        if vocal:
            result["style_prompt"] = g["suno_vocal"].format(bpm=bpm, key=key)
        else:
            result["style_prompt"] = g["suno_instrumental"].format(bpm=bpm, key=key)
        result["exclude"] = g["suno_exclude"]
        result["structure"] = g["structure_tags"]
    elif platform == "udio":
        result["prompt"] = g["udio"].format(bpm=bpm, key=key)
        result["settings"] = "Prompt Strength: 100%, Lyric Strength: 0% (instrumental)"

    result["tips"] = g["pro_tips"]
    result["beatport_data"] = g["beatport_data"]

    return result


def print_prompt(result: dict):
    """Print a formatted prompt ready to copy-paste."""
    print()
    print("=" * 70)
    print(f"  {result['genre'].upper()} PROMPT ({result['platform'].upper()})")
    print(f"  BPM: {result['bpm']} | Key: {result['key']}")
    print(f"  Reference: {result['reference_artists']}")
    print(f"  Market: {result['beatport_data']}")
    print("=" * 70)

    if result["platform"] == "suno":
        print()
        print("  STYLE OF MUSIC (paste into Suno):")
        print(f"  {result['style_prompt']}")
        print()
        print(f"  EXCLUDE: {result['exclude']}")
        print()
        print("  STRUCTURE (paste into lyrics/structure box):")
        for line in result["structure"].split("\n"):
            print(f"  {line}")
    elif result["platform"] == "udio":
        print()
        print("  PROMPT (paste into Udio):")
        print(f"  {result['prompt']}")
        print()
        print(f"  SETTINGS: {result['settings']}")

    print()
    print("  TIPS:")
    for tip in result["tips"]:
        print(f"    - {tip}")
    print()
    print("=" * 70)


def print_variations(genre: str, platform: str, vocal: bool, count: int):
    """Generate multiple prompt variations."""
    for i in range(count):
        print(f"\n--- Variation {i+1}/{count} ---")
        result = get_prompt(genre, platform, vocal)
        print_prompt(result)


def main():
    parser = argparse.ArgumentParser(
        description="AI music generation prompt library (Suno v5.5 / Udio)"
    )
    parser.add_argument("genre", nargs="?", default=None,
                        choices=list(GENRES.keys()) + ["list"],
                        help="Genre to generate prompt for")
    parser.add_argument("--platform", "-p", default="suno",
                        choices=["suno", "udio"],
                        help="Target platform")
    parser.add_argument("--vocal", "-v", action="store_true",
                        help="Include vocals (default: instrumental)")
    parser.add_argument("--bpm", "-b", type=int, default=None,
                        help="Specific BPM (default: random in genre range)")
    parser.add_argument("--key", "-k", default=None,
                        help="Specific key (default: random from genre)")
    parser.add_argument("--variations", "-n", type=int, default=1,
                        help="Number of prompt variations to generate")

    args = parser.parse_args()

    if args.genre is None or args.genre == "list":
        print("\nAvailable genres:")
        print(f"  {'Genre':<18} {'BPM Range':>10}  Reference Artists")
        print(f"  {'-'*18} {'-'*10}  {'-'*40}")
        for key, g in GENRES.items():
            print(f"  {key:<18} {g['bpm_range'][0]}-{g['bpm_range'][1]:>3}  {g['reference']}")
        print()
        print("Usage: python tools/prompts.py tech-house")
        print("       python tools/prompts.py afrobeats --vocal")
        print("       python tools/prompts.py tech-house --platform udio")
        print("       python tools/prompts.py tech-house --variations 5")
        return

    if args.variations > 1:
        print_variations(args.genre, args.platform, args.vocal, args.variations)
    else:
        result = get_prompt(args.genre, args.platform, args.vocal, args.bpm, args.key)
        print_prompt(result)


if __name__ == "__main__":
    main()
