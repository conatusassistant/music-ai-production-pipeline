"""
OS Music Pipeline — Main CLI
Your AI producer. One command to master, separate, analyze, process, or write.

Usage:
    python os_pipeline.py master track.wav --ref drake_ref.wav
    python os_pipeline.py separate song.mp3
    python os_pipeline.py analyze song.wav
    python os_pipeline.py vocal track.wav --preset weeknd
    python os_pipeline.py lyrics --artist drake --mood introspective --topic "late night"
    python os_pipeline.py setup                              # Install all dependencies
"""

import argparse
import subprocess
import sys


def cmd_setup(args):
    """Install all dependencies."""
    print("Installing OS Music Pipeline dependencies...")
    print()
    packages = [
        ("matchering", "Reference mastering"),
        ("demucs", "Stem separation"),
        ("librosa", "Audio analysis"),
        ("numpy", "Numerical computing"),
        ("matplotlib", "Charts and visualization"),
        ("soundfile", "Audio file I/O"),
        ("pedalboard", "Vocal processing chain"),
        ("anthropic", "Lyrics generation (Claude API)"),
        ("lyricsgenius", "Genius lyrics API"),
    ]

    for pkg, desc in packages:
        print(f"Installing {pkg} ({desc})...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], capture_output=True)
        print(f"  Done.")

    print()
    print("All dependencies installed!")
    print()
    print("Quick start:")
    print("  python os_pipeline.py master my_mix.wav --ref drake_song.wav")
    print("  python os_pipeline.py separate any_song.mp3")
    print("  python os_pipeline.py analyze any_song.wav")
    print("  python os_pipeline.py vocal my_vocals.wav --preset weeknd")
    print("  python os_pipeline.py lyrics --artist drake --topic 'late night drive'")


def cmd_master(args):
    from tools.master import master_track
    master_track(args.target, args.ref, args.output, args.format)


def cmd_separate(args):
    from tools.separate import separate_track, check_demucs
    if not check_demucs():
        print("Demucs not installed. Run: python os_pipeline.py setup")
        sys.exit(1)
    separate_track(args.input, args.output, args.vocals_only, args.model)


def cmd_analyze(args):
    from tools.analyze import analyze_track, compare_tracks, batch_analyze
    if args.batch:
        batch_analyze(args.input, plot=not args.no_plot)
    elif args.compare:
        compare_tracks(args.input, args.compare, plot=not args.no_plot)
    else:
        analyze_track(args.input, plot=not args.no_plot)


def cmd_vocal(args):
    from tools.vocal_chain import process_vocals, process_all_presets
    if args.all:
        process_all_presets(args.input)
    else:
        process_vocals(args.input, args.preset, args.output)


def cmd_lyrics(args):
    from tools.lyrics import generate_lyrics
    generate_lyrics(args.artist, args.mood, args.topic, args.bars)


def cmd_house(args):
    from tools.house import print_structure, generate_beat, chop_vocal, cmd_full
    if args.house_command == "structure":
        print_structure(args.style, args.bpm)
    elif args.house_command == "kick":
        generate_beat(args.bpm, args.duration, args.style, args.output)
    elif args.house_command == "chop":
        chop_vocal(args.input, args.slices, args.output)
    elif args.house_command == "full":
        cmd_full(args.bpm, args.style, args.vocal)


def main():
    parser = argparse.ArgumentParser(
        description="OS Music Pipeline — Your AI Producer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python os_pipeline.py setup                                    Install everything
  python os_pipeline.py master mix.wav --ref drake_ref.wav       Master against Drake
  python os_pipeline.py separate song.mp3                        Split into stems
  python os_pipeline.py analyze song.wav                         Get BPM, key, spectrum
  python os_pipeline.py vocal vocals.wav --preset weeknd         Weeknd vocal chain
  python os_pipeline.py lyrics -a drake -t "late night drive"    Write Drake-style lyrics
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline tool to run")

    # Setup
    sub_setup = subparsers.add_parser("setup", help="Install all dependencies")

    # Master
    sub_master = subparsers.add_parser("master", help="Reference mastering")
    sub_master.add_argument("target", help="Your raw mix (.wav)")
    sub_master.add_argument("--ref", "-r", required=True, help="Reference track")
    sub_master.add_argument("--output", "-o", default=None, help="Output filename")
    sub_master.add_argument("--format", "-f", default="wav", choices=["wav", "mp3"])

    # Separate
    sub_sep = subparsers.add_parser("separate", help="Stem separation (Demucs)")
    sub_sep.add_argument("input", help="Audio file or folder")
    sub_sep.add_argument("--output", "-o", default=None, help="Output directory")
    sub_sep.add_argument("--vocals-only", "-v", action="store_true")
    sub_sep.add_argument("--model", "-m", default="htdemucs",
                         choices=["htdemucs", "htdemucs_ft", "mdx_extra"])

    # Analyze
    sub_analyze = subparsers.add_parser("analyze", help="Audio analysis")
    sub_analyze.add_argument("input", help="Audio file or folder")
    sub_analyze.add_argument("--no-plot", action="store_true")
    sub_analyze.add_argument("--compare", "-c", default=None, help="Compare with another file")
    sub_analyze.add_argument("--batch", "-b", action="store_true")

    # Vocal
    sub_vocal = subparsers.add_parser("vocal", help="Vocal processing chain")
    sub_vocal.add_argument("input", help="Vocal audio file (.wav)")
    sub_vocal.add_argument("--preset", "-p", default="drake",
                           choices=["drake", "weeknd", "nav", "travis", "don-toliver", "raw"])
    sub_vocal.add_argument("--output", "-o", default=None)
    sub_vocal.add_argument("--all", "-a", action="store_true",
                           help="Process through ALL presets")

    # Lyrics
    sub_lyrics = subparsers.add_parser("lyrics", help="AI lyrics generator")
    sub_lyrics.add_argument("--artist", "-a", default="drake",
                            choices=["drake", "weeknd", "nav", "travis", "don-toliver", "kanye", "om"])
    sub_lyrics.add_argument("--mood", "-m", default="introspective",
                            choices=["introspective", "dark", "flexing", "heartbreak",
                                     "psychedelic", "confident", "dreamy", "rage",
                                     "vulnerable", "romantic"])
    sub_lyrics.add_argument("--topic", "-t", required=True, help="Song topic")
    sub_lyrics.add_argument("--bars", "-b", type=int, default=32)

    # House
    sub_house = subparsers.add_parser("house", help="House music production toolkit")
    house_sub = sub_house.add_subparsers(dest="house_command")

    hs_struct = house_sub.add_parser("structure", help="Print arrangement structure")
    hs_struct.add_argument("--style", "-s", default="tech-house",
                           choices=["tech-house", "deep-house", "bass-house"])
    hs_struct.add_argument("--bpm", "-b", type=float, default=None)

    hs_kick = house_sub.add_parser("kick", help="Generate kick + hat pattern")
    hs_kick.add_argument("--bpm", "-b", type=float, default=126)
    hs_kick.add_argument("--duration", "-d", type=float, default=30)
    hs_kick.add_argument("--style", "-s", default="tech-house",
                         choices=["tech-house", "deep-house", "bass-house"])
    hs_kick.add_argument("--output", "-o", default=None)

    hs_chop = house_sub.add_parser("chop", help="Chop vocals into rhythmic slices")
    hs_chop.add_argument("input", help="Vocal file to chop")
    hs_chop.add_argument("--slices", "-n", type=int, default=8)
    hs_chop.add_argument("--output", "-o", default=None)

    hs_full = house_sub.add_parser("full", help="Full demo: structure + beat + chops")
    hs_full.add_argument("--bpm", "-b", type=float, default=126)
    hs_full.add_argument("--style", "-s", default="tech-house",
                         choices=["tech-house", "deep-house", "bass-house"])
    hs_full.add_argument("--vocal", "-v", default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "setup": cmd_setup,
        "master": cmd_master,
        "separate": cmd_separate,
        "analyze": cmd_analyze,
        "vocal": cmd_vocal,
        "lyrics": cmd_lyrics,
        "house": cmd_house,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
