"""Audio analysis: BPM, key, frequency spectrum, energy, spectral features."""

import argparse
import json
import sys
from pathlib import Path

try:
    import librosa
    import numpy as np
except ImportError:
    print("Required packages not installed. Run: pip install librosa numpy matplotlib")
    sys.exit(1)


# Key name mapping
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def analyze_track(filepath: str, plot: bool = True) -> dict:
    """Analyze a single audio track and return its DNA."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"Analyzing: {path.name}")

    y, sr = librosa.load(str(path), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # BPM
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # Key Detection
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    key_idx = int(np.argmax(chroma_avg))
    key_name = KEY_NAMES[key_idx]

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_corr = np.corrcoef(np.roll(major_profile, key_idx), chroma_avg)[0, 1]
    minor_corr = np.corrcoef(np.roll(minor_profile, key_idx), chroma_avg)[0, 1]

    mode = "major" if major_corr > minor_corr else "minor"
    key_full = f"{key_name} {mode}"

    # Spectral Analysis
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    avg_centroid = float(np.mean(spectral_centroid))
    avg_rolloff = float(np.mean(spectral_rolloff))
    avg_bandwidth = float(np.mean(spectral_bandwidth))

    # RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    dynamic_range = float(np.max(rms) - np.min(rms))

    # === Zero Crossing Rate (texture) ===
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    avg_zcr = float(np.mean(zcr))

    # === MFCC (timbre fingerprint) ===
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = [float(x) for x in np.mean(mfccs, axis=1)]

    # === Results ===
    results = {
        "file": path.name,
        "duration_sec": round(duration, 1),
        "sample_rate": sr,
        "bpm": round(tempo, 1),
        "key": key_full,
        "key_note": key_name,
        "key_mode": mode,
        "spectral_centroid_hz": round(avg_centroid, 1),
        "spectral_rolloff_hz": round(avg_rolloff, 1),
        "spectral_bandwidth_hz": round(avg_bandwidth, 1),
        "rms_energy_mean": round(rms_mean, 5),
        "rms_energy_std": round(rms_std, 5),
        "dynamic_range": round(dynamic_range, 5),
        "zero_crossing_rate": round(avg_zcr, 5),
        "mfcc_means": [round(x, 2) for x in mfcc_means],
        "underwater_score": "high" if avg_centroid < 2000 else "medium" if avg_centroid < 3000 else "low",
    }

    print()
    print("=" * 50)
    print(f"ANALYSIS COMPLETE: {path.name}")
    print("=" * 50)
    print(f"  BPM:        {results['bpm']}")
    print(f"  Key:        {results['key']}")
    print(f"  Brightness: {results['spectral_centroid_hz']} Hz (underwater: {results['underwater_score']})")
    print(f"  Energy:     {results['rms_energy_mean']}")
    print(f"  Dynamics:   {results['dynamic_range']}")
    print("=" * 50)

    # === Plotting ===
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1, figsize=(14, 10))
            fig.suptitle(f"Audio Analysis: {path.name}", fontsize=14, fontweight="bold")

            times = librosa.times_like(spectral_centroid, sr=sr)

            # 1. Waveform
            librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=0.7)
            axes[0].set_title("Waveform")
            axes[0].set_ylabel("Amplitude")

            # 2. Spectral centroid over time
            axes[1].plot(times, spectral_centroid, color="orange", linewidth=0.8)
            axes[1].axhline(y=2000, color="red", linestyle="--", alpha=0.5, label="Underwater threshold")
            axes[1].set_title("Spectral Centroid (Brightness Over Time)")
            axes[1].set_ylabel("Hz")
            axes[1].legend()

            # 3. RMS Energy
            rms_times = librosa.times_like(rms, sr=sr)
            axes[2].plot(rms_times, rms, color="green", linewidth=0.8)
            axes[2].set_title("Energy (Verse vs Chorus Dynamics)")
            axes[2].set_ylabel("RMS")

            # 4. Chromagram
            librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", ax=axes[3], sr=sr)
            axes[3].set_title("Chromagram (Key/Harmony)")

            plt.tight_layout()
            plot_path = str(path.stem) + "_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nChart saved: {plot_path}")
        except ImportError:
            print("\nmatplotlib not installed — skipping charts. Install with: pip install matplotlib")

    return results


def compare_tracks(file1: str, file2: str, plot: bool = True):
    """Compare two tracks side by side."""
    print("=" * 60)
    print("COMPARISON MODE")
    print("=" * 60)
    print()

    r1 = analyze_track(file1, plot=False)
    print()
    r2 = analyze_track(file2, plot=False)

    print()
    print("=" * 60)
    print(f"COMPARISON: {r1['file']} vs {r2['file']}")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Track 1':>12} {'Track 2':>12} {'Diff':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'BPM':<25} {r1['bpm']:>12} {r2['bpm']:>12} {abs(r1['bpm']-r2['bpm']):>10.1f}")
    print(f"  {'Key':<25} {r1['key']:>12} {r2['key']:>12} {'':>10}")
    print(f"  {'Brightness (centroid)':<25} {r1['spectral_centroid_hz']:>12.0f} {r2['spectral_centroid_hz']:>12.0f} {abs(r1['spectral_centroid_hz']-r2['spectral_centroid_hz']):>10.0f}")
    print(f"  {'Underwater score':<25} {r1['underwater_score']:>12} {r2['underwater_score']:>12} {'':>10}")
    print(f"  {'Energy':<25} {r1['rms_energy_mean']:>12.5f} {r2['rms_energy_mean']:>12.5f} {abs(r1['rms_energy_mean']-r2['rms_energy_mean']):>10.5f}")
    print(f"  {'Dynamic range':<25} {r1['dynamic_range']:>12.5f} {r2['dynamic_range']:>12.5f} {abs(r1['dynamic_range']-r2['dynamic_range']):>10.5f}")
    print("=" * 60)

    return r1, r2


def batch_analyze(folder: str, plot: bool = False):
    """Analyze all audio files in a folder."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Error: Not a directory: {folder}")
        sys.exit(1)

    files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.mp3")) + list(folder_path.glob("*.flac"))
    if not files:
        print(f"No audio files found in {folder}")
        sys.exit(1)

    print(f"Found {len(files)} audio files. Analyzing...\n")

    all_results = []
    for f in files:
        try:
            result = analyze_track(str(f), plot=plot)
            all_results.append(result)
            print()
        except Exception as e:
            print(f"Error analyzing {f.name}: {e}")
            print()

    # Summary table
    if all_results:
        print("\n" + "=" * 80)
        print("BATCH SUMMARY")
        print("=" * 80)
        print(f"  {'File':<30} {'BPM':>6} {'Key':>10} {'Brightness':>12} {'Underwater':>12}")
        print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*12} {'-'*12}")
        for r in all_results:
            name = r['file'][:28]
            print(f"  {name:<30} {r['bpm']:>6.1f} {r['key']:>10} {r['spectral_centroid_hz']:>12.0f} {r['underwater_score']:>12}")

        # Averages
        avg_bpm = np.mean([r['bpm'] for r in all_results])
        avg_bright = np.mean([r['spectral_centroid_hz'] for r in all_results])
        print(f"\n  Average BPM: {avg_bpm:.1f}")
        print(f"  Average Brightness: {avg_bright:.0f} Hz")

        # Save JSON
        json_path = str(folder_path / "analysis_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to: {json_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze audio: BPM, key, frequency spectrum, energy, spectral features"
    )
    parser.add_argument("input", help="Audio file or folder (with --batch)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating charts")
    parser.add_argument("--compare", "-c", default=None, help="Second file to compare against")
    parser.add_argument("--batch", "-b", action="store_true", help="Analyze all files in a folder")
    parser.add_argument("--json", "-j", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    if args.batch:
        results = batch_analyze(args.input, plot=not args.no_plot)
    elif args.compare:
        results = compare_tracks(args.input, args.compare, plot=not args.no_plot)
    else:
        results = analyze_track(args.input, plot=not args.no_plot)

    if args.json:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
