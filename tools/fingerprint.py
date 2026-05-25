"""
OS Music Pipeline — Tool #10: Artist Style Fingerprinting
Analyze multiple tracks from any artist and extract their production DNA.
Then apply that fingerprint as a processing target to YOUR tracks.

This is the future: instead of guessing EQ settings, you feed it 10 John Summit
tracks and it extracts his exact spectral profile, compression character,
loudness target, and arrangement patterns. Then it makes YOUR track sound like his.

Usage:
    # Build a fingerprint from reference tracks:
    python tools/fingerprint.py build summit_tracks/ --name "john-summit"

    # Apply a fingerprint to your track:
    python tools/fingerprint.py apply my_track.wav --fingerprint john-summit

    # Compare your track against a fingerprint:
    python tools/fingerprint.py compare my_track.wav --fingerprint john-summit

    # List saved fingerprints:
    python tools/fingerprint.py list
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import librosa
    import soundfile as sf
except ImportError:
    print("Required: pip install librosa numpy soundfile")
    sys.exit(1)

try:
    from pedalboard import (
        Pedalboard, Compressor, Gain, HighpassFilter, LowpassFilter,
        LowShelfFilter, HighShelfFilter, PeakFilter, Limiter, Reverb,
    )
except ImportError:
    print("Required: pip install pedalboard")
    sys.exit(1)


FINGERPRINT_DIR = Path(__file__).parent.parent / "fingerprints"


def analyze_single(filepath: str) -> dict:
    """Extract production DNA from a single track."""
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # Key
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    key_idx = int(np.argmax(chroma_avg))
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_corr = np.corrcoef(np.roll(major_profile, key_idx), chroma_avg)[0, 1]
    minor_corr = np.corrcoef(np.roll(minor_profile, key_idx), chroma_avg)[0, 1]
    mode = "major" if major_corr > minor_corr else "minor"

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # RMS / dynamics
    rms = librosa.feature.rms(y=y)[0]

    # Spectral contrast (frequency band energy distribution)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_means = np.mean(contrast, axis=1).tolist()

    # MFCCs (timbre fingerprint)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1).tolist()

    # Spectral flatness (tonality vs noise)
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    # Onset strength (transient density)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Full spectrum profile (64 frequency bands)
    S = np.abs(librosa.stft(y))
    freq_bands = 64
    band_size = S.shape[0] // freq_bands
    spectrum_profile = []
    for i in range(freq_bands):
        start = i * band_size
        end = start + band_size
        band_energy = float(np.mean(S[start:end, :]))
        spectrum_profile.append(band_energy)

    # Normalize spectrum profile
    sp_max = max(spectrum_profile) if max(spectrum_profile) > 0 else 1
    spectrum_profile = [x / sp_max for x in spectrum_profile]

    return {
        "file": Path(filepath).name,
        "duration_sec": round(duration, 1),
        "bpm": round(tempo, 1),
        "key": key_names[key_idx],
        "mode": mode,
        "spectral_centroid_mean": round(float(np.mean(centroid)), 1),
        "spectral_centroid_std": round(float(np.std(centroid)), 1),
        "spectral_rolloff_mean": round(float(np.mean(rolloff)), 1),
        "spectral_bandwidth_mean": round(float(np.mean(bandwidth)), 1),
        "rms_mean": round(float(np.mean(rms)), 5),
        "rms_std": round(float(np.std(rms)), 5),
        "rms_max": round(float(np.max(rms)), 5),
        "dynamic_range": round(float(np.max(rms) - np.min(rms)), 5),
        "crest_factor": round(float(np.max(np.abs(y)) / (np.sqrt(np.mean(y ** 2)) + 1e-10)), 2),
        "spectral_flatness_mean": round(float(np.mean(flatness)), 5),
        "onset_density": round(float(np.mean(onset_env)), 4),
        "spectral_contrast": [round(x, 2) for x in contrast_means],
        "mfcc_means": [round(x, 2) for x in mfcc_means],
        "spectrum_profile": [round(x, 4) for x in spectrum_profile],
    }


def build_fingerprint(folder: str, name: str):
    """Analyze all tracks in a folder and create an artist fingerprint."""
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    files = (list(folder_path.glob("*.wav")) +
             list(folder_path.glob("*.mp3")) +
             list(folder_path.glob("*.flac")))

    if not files:
        print(f"No audio files found in {folder}")
        sys.exit(1)

    print(f"Building fingerprint '{name}' from {len(files)} tracks...")
    print()

    analyses = []
    for f in files:
        try:
            print(f"  Analyzing: {f.name}...")
            result = analyze_single(str(f))
            analyses.append(result)
        except Exception as e:
            print(f"  Error on {f.name}: {e}")

    if not analyses:
        print("No tracks analyzed successfully.")
        sys.exit(1)

    # Aggregate into fingerprint
    fingerprint = {
        "name": name,
        "track_count": len(analyses),
        "tracks": [a["file"] for a in analyses],
        "bpm": {
            "mean": round(np.mean([a["bpm"] for a in analyses]), 1),
            "min": round(min(a["bpm"] for a in analyses), 1),
            "max": round(max(a["bpm"] for a in analyses), 1),
            "std": round(float(np.std([a["bpm"] for a in analyses])), 1),
        },
        "keys": {a["key"] + " " + a["mode"]: 0 for a in analyses},
        "spectral_centroid": {
            "mean": round(np.mean([a["spectral_centroid_mean"] for a in analyses]), 1),
            "std": round(float(np.std([a["spectral_centroid_mean"] for a in analyses])), 1),
        },
        "spectral_rolloff_mean": round(np.mean([a["spectral_rolloff_mean"] for a in analyses]), 1),
        "spectral_bandwidth_mean": round(np.mean([a["spectral_bandwidth_mean"] for a in analyses]), 1),
        "rms": {
            "mean": round(np.mean([a["rms_mean"] for a in analyses]), 5),
            "std": round(float(np.std([a["rms_mean"] for a in analyses])), 5),
        },
        "dynamic_range_mean": round(np.mean([a["dynamic_range"] for a in analyses]), 5),
        "crest_factor_mean": round(np.mean([a["crest_factor"] for a in analyses]), 2),
        "spectral_flatness_mean": round(np.mean([a["spectral_flatness_mean"] for a in analyses]), 5),
        "onset_density_mean": round(np.mean([a["onset_density"] for a in analyses]), 4),
        "spectral_contrast_mean": [
            round(np.mean([a["spectral_contrast"][i] for a in analyses]), 2)
            for i in range(len(analyses[0]["spectral_contrast"]))
        ],
        "mfcc_means": [
            round(np.mean([a["mfcc_means"][i] for a in analyses]), 2)
            for i in range(len(analyses[0]["mfcc_means"]))
        ],
        "spectrum_profile": [
            round(np.mean([a["spectrum_profile"][i] for a in analyses]), 4)
            for i in range(len(analyses[0]["spectrum_profile"]))
        ],
    }

    # Count keys
    for a in analyses:
        k = a["key"] + " " + a["mode"]
        fingerprint["keys"][k] = fingerprint["keys"].get(k, 0) + 1

    # Save fingerprint
    FINGERPRINT_DIR.mkdir(exist_ok=True)
    fp_path = FINGERPRINT_DIR / f"{name}.json"
    with open(fp_path, "w") as f:
        json.dump(fingerprint, f, indent=2)

    print()
    print("=" * 60)
    print(f"  FINGERPRINT: {name}")
    print(f"  Tracks analyzed: {len(analyses)}")
    print("=" * 60)
    print(f"  BPM: {fingerprint['bpm']['mean']} (range: {fingerprint['bpm']['min']}-{fingerprint['bpm']['max']})")
    print(f"  Most common key: {max(fingerprint['keys'], key=fingerprint['keys'].get)}")
    print(f"  Brightness: {fingerprint['spectral_centroid']['mean']} Hz")
    print(f"  Rolloff: {fingerprint['spectral_rolloff_mean']} Hz")
    print(f"  Energy: {fingerprint['rms']['mean']}")
    print(f"  Dynamic range: {fingerprint['dynamic_range_mean']}")
    print(f"  Crest factor: {fingerprint['crest_factor_mean']}")
    print(f"  Onset density: {fingerprint['onset_density_mean']}")
    print(f"  Spectral flatness: {fingerprint['spectral_flatness_mean']}")
    print("=" * 60)
    print(f"  Saved to: {fp_path}")
    print()
    print(f"  Apply to a track:")
    print(f"    python tools/fingerprint.py apply your_track.wav --fingerprint {name}")

    return fingerprint


def load_fingerprint(name: str) -> dict:
    fp_path = FINGERPRINT_DIR / f"{name}.json"
    if not fp_path.exists():
        print(f"Fingerprint '{name}' not found.")
        print(f"Available: {list_fingerprints()}")
        sys.exit(1)
    with open(fp_path) as f:
        return json.load(f)


def list_fingerprints():
    FINGERPRINT_DIR.mkdir(exist_ok=True)
    fps = list(FINGERPRINT_DIR.glob("*.json"))
    if not fps:
        print("No fingerprints saved yet.")
        print("Create one: python tools/fingerprint.py build tracks_folder/ --name artist-name")
        return []
    names = []
    for fp_path in fps:
        with open(fp_path) as f:
            fp = json.load(f)
        name = fp_path.stem
        names.append(name)
        print(f"  {name:<20} {fp['track_count']} tracks, BPM {fp['bpm']['mean']}, brightness {fp['spectral_centroid']['mean']} Hz")
    return names


def compare_to_fingerprint(filepath: str, fp_name: str):
    """Compare a track against a saved fingerprint. Shows the gaps."""
    fp = load_fingerprint(fp_name)
    track = analyze_single(filepath)

    print()
    print("=" * 70)
    print(f"  COMPARISON: {track['file']} vs {fp_name} fingerprint")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Your Track':>12} {'Target':>12} {'Gap':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")

    metrics = [
        ("BPM", track["bpm"], fp["bpm"]["mean"], ""),
        ("Brightness (Hz)", track["spectral_centroid_mean"], fp["spectral_centroid"]["mean"], "Hz"),
        ("Rolloff (Hz)", track["spectral_rolloff_mean"], fp["spectral_rolloff_mean"], "Hz"),
        ("Bandwidth (Hz)", track["spectral_bandwidth_mean"], fp["spectral_bandwidth_mean"], "Hz"),
        ("RMS Energy", track["rms_mean"], fp["rms"]["mean"], ""),
        ("Dynamic Range", track["dynamic_range"], fp["dynamic_range_mean"], ""),
        ("Crest Factor", track["crest_factor"], fp["crest_factor_mean"], ""),
        ("Spectral Flatness", track["spectral_flatness_mean"], fp["spectral_flatness_mean"], ""),
        ("Onset Density", track["onset_density"], fp["onset_density_mean"], ""),
    ]

    issues = []
    for name, yours, target, unit in metrics:
        gap = yours - target
        gap_pct = abs(gap / (target + 1e-10)) * 100
        flag = " !!" if gap_pct > 30 else ""
        print(f"  {name:<30} {yours:>12.2f} {target:>12.2f} {gap:>+10.2f}{flag}")
        if gap_pct > 30:
            if "Brightness" in name and gap > 0:
                issues.append("Track is too bright - needs low-pass or high cut")
            elif "Brightness" in name and gap < 0:
                issues.append("Track is too dark - needs high shelf boost")
            elif "RMS" in name and gap < 0:
                issues.append("Track is too quiet - needs more gain/compression")
            elif "RMS" in name and gap > 0:
                issues.append("Track is too loud - over-compressed")
            elif "Dynamic Range" in name and gap > 0:
                issues.append("Too much dynamic range - needs more compression")
            elif "Dynamic Range" in name and gap < 0:
                issues.append("Too compressed - needs more dynamics")
            elif "Crest Factor" in name:
                issues.append("Transient balance is off - check compression attack")

    print("=" * 70)
    if issues:
        print("\n  ISSUES TO FIX:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  Track matches fingerprint well!")

    return track, fp


def apply_fingerprint(filepath: str, fp_name: str, output: str = None):
    """Apply a fingerprint's spectral profile to a track.
    Adjusts brightness, energy, and dynamics to match the target artist."""
    fp = load_fingerprint(fp_name)
    track = analyze_single(filepath)

    audio, sr = sf.read(filepath, dtype='float32')
    if audio.ndim > 1:
        audio = audio.T
    else:
        audio = audio[np.newaxis, :]

    print(f"Applying '{fp_name}' fingerprint to {Path(filepath).name}...")

    # Calculate corrections needed
    brightness_gap = fp["spectral_centroid"]["mean"] - track["spectral_centroid_mean"]
    energy_gap = fp["rms"]["mean"] - track["rms_mean"]

    effects = []

    # Brightness correction via EQ
    if brightness_gap > 300:
        # Track is too dark, boost highs
        boost = min(brightness_gap / 500, 4.0)
        effects.append(HighShelfFilter(cutoff_frequency_hz=8000, gain_db=boost))
        print(f"  Boosting highs: +{boost:.1f} dB shelf at 8kHz (track too dark)")
    elif brightness_gap < -300:
        # Track is too bright, cut highs
        cut = min(abs(brightness_gap) / 500, 4.0)
        effects.append(HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-cut))
        effects.append(LowpassFilter(cutoff_frequency_hz=16000))
        print(f"  Cutting highs: -{cut:.1f} dB shelf at 8kHz (track too bright)")

    # Always apply corrective EQ
    effects.append(HighpassFilter(cutoff_frequency_hz=30))
    effects.append(PeakFilter(cutoff_frequency_hz=300, gain_db=-1.5, q=1.5))

    # Compression to match dynamics
    target_crest = fp["crest_factor_mean"]
    current_crest = track["crest_factor"]
    if current_crest > target_crest * 1.3:
        # Too dynamic, compress more
        ratio = min(2.0 + (current_crest - target_crest) * 0.5, 6.0)
        effects.append(Compressor(threshold_db=-16, ratio=ratio, attack_ms=15, release_ms=150))
        print(f"  Compressing: {ratio:.1f}:1 ratio (too dynamic)")
    elif current_crest < target_crest * 0.7:
        # Over-compressed, lighter compression
        effects.append(Compressor(threshold_db=-10, ratio=1.5, attack_ms=30, release_ms=200))
        print(f"  Light compression: 1.5:1 (preserving dynamics)")

    # Energy matching via gain + limiting
    if energy_gap > 0.05:
        gain = min(energy_gap * 40, 6.0)
        effects.append(Gain(gain_db=gain))
        print(f"  Boosting gain: +{gain:.1f} dB (track too quiet)")
    elif energy_gap < -0.05:
        gain = max(energy_gap * 40, -6.0)
        effects.append(Gain(gain_db=gain))
        print(f"  Reducing gain: {gain:.1f} dB (track too loud)")

    effects.append(Limiter(threshold_db=-1.0))

    # Apply
    board = Pedalboard(effects)
    processed = board(audio, sr)

    if output is None:
        output = f"{Path(filepath).stem}_{fp_name}.wav"

    if processed.shape[0] == 1:
        sf.write(output, processed[0], sr)
    else:
        sf.write(output, processed.T, sr)

    # Verify
    result = analyze_single(output)
    print()
    print(f"  Output: {output}")
    print(f"  Brightness: {track['spectral_centroid_mean']:.0f} > {result['spectral_centroid_mean']:.0f} Hz (target: {fp['spectral_centroid']['mean']:.0f})")
    print(f"  Energy: {track['rms_mean']:.5f} > {result['rms_mean']:.5f} (target: {fp['rms']['mean']:.5f})")
    print(f"  Crest: {track['crest_factor']:.2f} > {result['crest_factor']:.2f} (target: {fp['crest_factor_mean']:.2f})")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Artist style fingerprinting: analyze, compare, and apply production DNA"
    )
    subparsers = parser.add_subparsers(dest="command")

    sub_build = subparsers.add_parser("build", help="Build fingerprint from tracks")
    sub_build.add_argument("folder", help="Folder of audio files")
    sub_build.add_argument("--name", "-n", required=True, help="Fingerprint name")

    sub_apply = subparsers.add_parser("apply", help="Apply fingerprint to a track")
    sub_apply.add_argument("input", help="Audio file to process")
    sub_apply.add_argument("--fingerprint", "-f", required=True, help="Fingerprint name")
    sub_apply.add_argument("--output", "-o", default=None)

    sub_compare = subparsers.add_parser("compare", help="Compare track vs fingerprint")
    sub_compare.add_argument("input", help="Audio file")
    sub_compare.add_argument("--fingerprint", "-f", required=True, help="Fingerprint name")

    sub_list = subparsers.add_parser("list", help="List saved fingerprints")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "build":
        build_fingerprint(args.folder, args.name)
    elif args.command == "apply":
        apply_fingerprint(args.input, args.fingerprint, args.output)
    elif args.command == "compare":
        compare_to_fingerprint(args.input, args.fingerprint)
    elif args.command == "list":
        list_fingerprints()


if __name__ == "__main__":
    main()
