"""
OS Music Pipeline — Tool #4: Lyrics Generator
Generate lyrics in the style of Drake, Weeknd, Nav, Don Toliver, Travis Scott, Kanye.
Uses Claude API for generation with artist-specific style prompts.

Usage:
    python tools/lyrics.py --artist drake --mood introspective --topic "late night drive"
    python tools/lyrics.py --artist weeknd --mood dark --topic "toxic love"
    python tools/lyrics.py --artist nav --mood flexing --topic "came from nothing"
    python tools/lyrics.py --artist travis --mood psychedelic --topic "rage"
    python tools/lyrics.py --artist don-toliver --mood dreamy --topic "summer nights"
    python tools/lyrics.py --artist kanye --mood confident --topic "self-made"
    python tools/lyrics.py --artist om --mood all --topic "your story"  # YOUR style

Set ANTHROPIC_API_KEY environment variable before running.
"""

import argparse
import os
import sys

try:
    import anthropic
except ImportError:
    print("anthropic not installed. Run: pip install anthropic")
    sys.exit(1)


# === ARTIST STYLE PROFILES ===

ARTIST_STYLES = {
    "drake": {
        "name": "Drake",
        "description": "Drake's writing style",
        "traits": [
            "Switches between rapping and singing mid-verse",
            "References specific Toronto locations, women's names, luxury brands",
            "Introspective bars about fame's loneliness mixed with flex bars",
            "Uses 'I' heavily — everything is personal narrative",
            "Short punchy sentences, conversational tone",
            "Frequent rhetorical questions",
            "References past relationships with specific details",
            "Double entendres and wordplay that reveal on second listen",
            "Vulnerable one line, braggadocious the next",
            "Uses 'yeah' and 'woo' as ad-libs between lines",
        ],
        "structure": "Verse (16 bars) → Pre-chorus (4 bars) → Chorus (8 bars, melodic singing) → Verse 2 → Chorus → Bridge (emotional peak) → Chorus",
        "rhyme_scheme": "AABB with internal rhymes, occasional ABAB",
        "syllable_density": "Medium — leaves space for melody",
        "reference_songs": ["Marvin's Room", "Passionfruit", "Race My Mind", "Texts Go Green", "8am in Charlotte"],
    },
    "weeknd": {
        "name": "The Weeknd",
        "description": "The Weeknd's writing style",
        "traits": [
            "Dark, hedonistic imagery — drugs, sex, heartbreak intertwined",
            "Cinematic scene-setting (specific times, places, atmospheres)",
            "Falsetto-friendly phrasing — words chosen for how they sound sung high",
            "Repetitive hooks that build hypnotically",
            "References to being numb, escaping, losing himself",
            "Second person 'you' addressing a lover/enabler",
            "Short melodic phrases, lots of open vowels (oh, ah, ee)",
            "Dark metaphors: night, stars, blood, pills, mirrors",
            "Contrast between beauty and destruction",
            "Michael Jackson influence in rhythmic phrasing",
        ],
        "structure": "Intro (atmospheric) → Verse (12-16 bars) → Pre-chorus (building tension) → Chorus (falsetto hook, 4-8 bars) → Verse 2 → Chorus → Bridge (breakdown) → Final Chorus",
        "rhyme_scheme": "Loose rhyming, prioritizes melody over strict rhyme",
        "syllable_density": "Low to medium — space for vocal runs",
        "reference_songs": ["The Hills", "Blinding Lights", "Call Out My Name", "Wicked Games", "Die For You"],
    },
    "nav": {
        "name": "Nav",
        "description": "Nav's writing style",
        "traits": [
            "Monotone delivery — lyrics are simple and repetitive by design",
            "Flexing: money, cars, jewelry, women — but matter-of-fact, not aggressive",
            "References being South Asian ('brown boy') as identity",
            "Self-produced references, studio life",
            "Short sentences, 4-6 syllables per bar typical",
            "Heavy use of brand names (Vlone, Bape, Lamborghini)",
            "Melodic hooks that are catchy BECAUSE they're simple",
            "Topics: came from nothing, trust issues, isolation despite success",
            "Doesn't try to be lyrical — hooks over bars",
            "Uses 'yeah yeah yeah' and humming as filler that becomes the hook",
        ],
        "structure": "Chorus (immediate hook) → Verse (12-16 bars) → Chorus → Verse 2 → Chorus → Outro",
        "rhyme_scheme": "AABB, very simple end rhymes",
        "syllable_density": "Low — maximally simple",
        "reference_songs": ["Myself", "Some Way", "Wanted You", "Good For It", "Champion"],
    },
    "travis": {
        "name": "Travis Scott",
        "description": "Travis Scott's writing style",
        "traits": [
            "Ad-lib heavy — 'it's lit!', 'straight up!', 'la flame!'",
            "Psychedelic imagery — lights, colors, space, rage",
            "Energy-focused: words chosen for how they sound screamed/chanted",
            "Festival/concert-ready hooks — call-and-response",
            "References Houston ('H-Town'), Cactus Jack",
            "Dark luxury — Lambos but also darkness, paranoia",
            "Triplet flow mixed with melodic singing",
            "Stacking vocals — same line delivered multiple ways simultaneously",
            "Short phrases repeated and varied slightly each time",
            "References to staying up, partying, losing control",
        ],
        "structure": "Intro (beat drop) → Chorus (rage hook) → Verse (aggressive 16) → Chorus → Verse 2 → Bridge (atmospheric breakdown) → Final Chorus (biggest energy)",
        "rhyme_scheme": "Loose, triplet-based, internal rhymes",
        "syllable_density": "Medium-high in verses, low in hooks",
        "reference_songs": ["SICKO MODE", "goosebumps", "HIGHEST IN THE ROOM", "BUTTERFLY EFFECT", "Antidote"],
    },
    "don-toliver": {
        "name": "Don Toliver",
        "description": "Don Toliver's writing style",
        "traits": [
            "Melodic singing over rapping — every line is a melody",
            "Psychedelic love songs — dreamy, floaty atmosphere",
            "Pitch-shifted vocal feel in the writing (words that bend)",
            "Houston slang mixed with universal themes",
            "Repetitive melodic hooks that stick immediately",
            "Simple vocabulary, complex melodies",
            "References: late nights, driving, women, being in his own world",
            "Open vowel sounds for maximum autotune effect",
            "Layered harmonies in the writing (implies stacked vocals)",
            "Chill energy — never rushed, spacious phrasing",
        ],
        "structure": "Chorus (melodic hook first) → Verse (singing, 12 bars) → Chorus → Verse 2 → Bridge (stripped down) → Chorus",
        "rhyme_scheme": "Melody-first, loose end rhymes",
        "syllable_density": "Low — maximum melodic space",
        "reference_songs": ["No Idea", "After Party", "Lemonade", "Cardigan", "BANDIT"],
    },
    "kanye": {
        "name": "Kanye West",
        "description": "Kanye West's writing style",
        "traits": [
            "Stream of consciousness — jumps between topics mid-verse",
            "Self-referential, grandiose, contradictory",
            "Cultural commentary mixed with personal flex",
            "Soul samples referenced lyrically (chipmunk soul era callbacks)",
            "Humor — punchlines that are funny and deep simultaneously",
            "References God, family, industry, race, fashion",
            "Shifts between vulnerability and megalomania",
            "Uses conversational phrasing that sounds improvised",
            "Bold declarations: 'I am...' statements",
            "Political and social commentary woven through personal narrative",
        ],
        "structure": "Variable — Kanye breaks structure. Sometimes no chorus. Sometimes 3 different beats in one song. The unexpected IS the structure.",
        "rhyme_scheme": "Complex multisyllabic when rapping, simple when singing",
        "syllable_density": "High in rap verses, low in sung sections",
        "reference_songs": ["Runaway", "Stronger", "Ultralight Beam", "Power", "Ghost Town"],
    },
    "om": {
        "name": "Om Sharma (YOUR voice)",
        "description": "Your unique perspective",
        "traits": [
            "Berkeley-educated, tech world perspective",
            "South Asian identity (like Nav, but with your own angle)",
            "Real estate / business world references nobody else has",
            "Actor's storytelling ability — cinematic scenes",
            "Switches between Silicon Valley flex and genuine vulnerability",
            "References hustle culture, late nights coding, deal-making",
            "Immigrant family narrative, proving doubters wrong",
            "Drake's introspection + Nav's brown boy pride + Weeknd's darkness",
            "Real stories — not fabricated street cred",
            "The gap between ambition and where you are right now",
        ],
        "structure": "Verse → Melodic Chorus → Verse 2 → Chorus → Bridge (raw moment) → Chorus",
        "rhyme_scheme": "AABB with internal rhymes, conversational",
        "syllable_density": "Medium — room for melody and autotune",
        "reference_songs": ["Your life is the reference"],
    },
}

MOOD_MODIFIERS = {
    "introspective": "Reflective, late-night thoughts, questioning decisions, looking back",
    "dark": "Shadowy, moody, nocturnal, haunted by something, numb",
    "flexing": "Confident, celebrating wins, luxury, proving people wrong",
    "heartbreak": "Loss, missing someone, regret, memories of better times",
    "psychedelic": "Trippy, colorful, otherworldly, altered states, floating",
    "confident": "Unstoppable energy, manifesting success, declarations of greatness",
    "dreamy": "Floating, ethereal, surreal beauty, half-asleep thoughts",
    "rage": "Aggressive energy, anger, intensity, festival mosh pit energy",
    "vulnerable": "Raw honesty, admitting weakness, fear, uncertainty",
    "romantic": "Love, desire, connection, sensual, intimate",
}


def generate_lyrics(artist: str, mood: str, topic: str, bars: int = 32):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    if artist not in ARTIST_STYLES:
        print(f"Unknown artist: {artist}")
        print(f"Available: {', '.join(ARTIST_STYLES.keys())}")
        sys.exit(1)

    style = ARTIST_STYLES[artist]
    mood_desc = MOOD_MODIFIERS.get(mood, mood)

    prompt = f"""Write original song lyrics in the style of {style['name']}.

STYLE TRAITS:
{chr(10).join(f'- {t}' for t in style['traits'])}

SONG STRUCTURE:
{style['structure']}

RHYME SCHEME: {style['rhyme_scheme']}
SYLLABLE DENSITY: {style['syllable_density']}

MOOD: {mood} — {mood_desc}
TOPIC: {topic}

TARGET LENGTH: approximately {bars} bars total across all sections

REFERENCE SONGS FOR FEEL: {', '.join(style['reference_songs'])}

RULES:
1. Write ORIGINAL lyrics — do not copy existing songs
2. Follow the structure format above
3. Include section labels [Verse 1], [Chorus], [Pre-Chorus], [Bridge], [Outro]
4. Include (ad-lib) annotations in parentheses where they'd naturally go
5. Write lyrics that sound natural when SUNG with autotune, not just read
6. Use open vowel sounds at line endings for autotune resonance
7. Keep it real — no corny lines, no forced rhymes
8. The hook should be immediately memorable on first listen
9. Include notes on delivery style in brackets where helpful [whispered], [falsetto], [aggressive]

Write the complete song now:"""

    print(f"Generating {style['name']}-style lyrics...")
    print(f"  Mood: {mood}")
    print(f"  Topic: {topic}")
    print(f"  Target: ~{bars} bars")
    print()

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    lyrics = message.content[0].text

    print("=" * 60)
    print(f"  {style['name'].upper()} x OM — \"{topic.upper()}\"")
    print(f"  Mood: {mood} | Style: {style['description']}")
    print("=" * 60)
    print()
    print(lyrics)
    print()
    print("=" * 60)

    # Save to file
    safe_topic = topic.replace(" ", "_").replace("/", "-")[:30]
    filename = f"lyrics_{artist}_{safe_topic}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Artist Style: {style['name']}\n")
        f.write(f"Mood: {mood}\n")
        f.write(f"Topic: {topic}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(lyrics)

    print(f"Saved to: {filename}")

    return lyrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate lyrics in the style of Drake, Weeknd, Nav, Travis, Don Toliver, Kanye, or YOU"
    )
    parser.add_argument("--artist", "-a", default="drake",
                        choices=list(ARTIST_STYLES.keys()),
                        help="Artist style to write in")
    parser.add_argument("--mood", "-m", default="introspective",
                        choices=list(MOOD_MODIFIERS.keys()),
                        help="Mood/vibe of the song")
    parser.add_argument("--topic", "-t", required=True,
                        help="What the song is about")
    parser.add_argument("--bars", "-b", type=int, default=32,
                        help="Approximate number of bars (default: 32)")
    parser.add_argument("--list-artists", action="store_true",
                        help="List available artist styles")
    parser.add_argument("--list-moods", action="store_true",
                        help="List available moods")

    args = parser.parse_args()

    if args.list_artists:
        print("Available artist styles:")
        for key, style in ARTIST_STYLES.items():
            print(f"  {key:<15} {style['description']}")
        return

    if args.list_moods:
        print("Available moods:")
        for key, desc in MOOD_MODIFIERS.items():
            print(f"  {key:<15} {desc}")
        return

    generate_lyrics(args.artist, args.mood, args.topic, args.bars)


if __name__ == "__main__":
    main()
