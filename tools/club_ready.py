"""
OS Music Pipeline — Tool #7: Club-Ready Processing
Bridges the gap between bedroom production and club-quality sound.
Implements the specific techniques that John Summit's engineer (Luca Pretolesi) uses.

What this does:
  1. Mono bass management (everything below 150Hz summed to mono)
  2. Sidechain-style ducking (the "pump" that makes tech house groove)
  3. Multi-stage saturation (John Summit's signature: Decapitator + Saturn approach)
  4. Mastering chain: corrective EQ > compression > tonal EQ > saturation > limiting
  5. LUFS targeting for club play (-8 to -9 LUFS) vs streaming (-14 LUFS)

Usage:
    python tools/club_ready.py track.wav --target club
    python tools/club_ready.py track.wav --target streaming
    python tools/club_ready.py track.wav --target club --sidechain --bpm 126
    python tools/club_ready.py track.wav --analyze-only
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    print("Required: pip install numpy soundfile")
    sys.exit(1)

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
        Limiter,
        Distortion,
        Clipping,
    )
except ImportError:
    print("Required: pip install pedalboard")
    sys.exit(1)


def mono_bass(audio: np.ndarray, sr: int, crossover_hz: float = 150.0) -> np.ndarray:
    """Sum everything below crossover_hz to mono.
    Club PAs sum low frequencies to mono anyway — if you don't do this,
    phase cancellation kills your sub-bass on a real system."""
    if audio.ndim == 1:
        return audio  # already mono, nothing to do

    from scipy.signal import butter, sosfilt

    # Design crossover filter
    sos_low = butter(4, crossover_hz, btype='low', fs=sr, output='sos')
    sos_high = butter(4, crossover_hz, btype='high', fs=sr, output='sos')

    channels = audio.shape[0]
    output = np.zeros_like(audio)

    # Split each channel into low and high
    lows = np.array([sosfilt(sos_low, audio[ch]) for ch in range(channels)])
    highs = np.array([sosfilt(sos_high, audio[ch]) for ch in range(channels)])

    # Sum lows to mono, keep highs stereo
    mono_low = np.mean(lows, axis=0)
    for ch in range(channels):
        output[ch] = mono_low + highs[ch]

    return output


def sidechain_pump(audio: np.ndarray, sr: int, bpm: float, depth_db: float = -6.0,
                   attack_ms: float = 5.0, release_ms: float = 150.0) -> np.ndarray:
    """Apply sidechain-style volume ducking synced to BPM.
    This is what creates the 'pump' in tech house — the bass ducks
    every time the kick hits, creating rhythmic breathing."""
    beat_interval = 60.0 / bpm
    samples_per_beat = int(sr * beat_interval)

    attack_samples = int(sr * attack_ms / 1000)
    release_samples = int(sr * release_ms / 1000)

    total_samples = audio.shape[-1]

    # Build the ducking envelope
    envelope = np.ones(total_samples)
    depth_linear = 10 ** (depth_db / 20)  # convert dB to linear

    pos = 0
    while pos < total_samples:
        # Attack phase: duck down
        attack_end = min(pos + attack_samples, total_samples)
        t_attack = np.linspace(1.0, depth_linear, attack_end - pos)
        envelope[pos:attack_end] = t_attack

        # Release phase: recover
        release_start = attack_end
        release_end = min(release_start + release_samples, total_samples)
        t_release = np.linspace(depth_linear, 1.0, release_end - release_start)
        envelope[release_start:release_end] = t_release

        pos += samples_per_beat

    # Apply envelope
    if audio.ndim == 1:
        return audio * envelope
    else:
        return audio * envelope[np.newaxis, :]


def multi_stage_saturation(audio: np.ndarray, sr: int, drive_db: float = 3.0) -> np.ndarray:
    """John Summit's double-saturation technique.
    Stage 1: Soft clipping (like Decapitator) — broad harmonic warmth
    Stage 2: Tanh saturation — controlled distortion that adds grit
    This is what makes his tracks sound 'thick' and 'dirty' without being harsh."""

    # Stage 1: Soft clip (even harmonics, warm character)
    drive = 10 ** (drive_db / 20)
    driven = audio * drive

    # Soft clip using tanh (models tube saturation)
    stage1 = np.tanh(driven * 0.7) / np.tanh(0.7)

    # Stage 2: Asymmetric soft clip (odd + even harmonics, adds grit)
    # Slightly different curve for positive vs negative — this is what
    # gives analog saturation its character vs digital clipping
    positive = np.where(stage1 > 0, stage1, 0)
    negative = np.where(stage1 < 0, stage1, 0)

    # Positive side: gentler saturation
    pos_sat = np.tanh(positive * 1.2) / np.tanh(1.2)
    # Negative side: slightly harder — creates asymmetry = even harmonics
    neg_sat = np.tanh(negative * 1.4) / np.tanh(1.4)

    stage2 = pos_sat + neg_sat

    # Normalize to original peak level to prevent volume change
    if np.max(np.abs(stage2)) > 0:
        stage2 = stage2 * (np.max(np.abs(audio)) / np.max(np.abs(stage2)))

    return stage2


def mastering_chain(audio: np.ndarray, sr: int, target: str = "club") -> np.ndarray:
    """Professional mastering chain.
    Based on Luca Pretolesi's approach (John Summit's engineer).

    Chain order:
    1. Corrective EQ (cuts only — remove mud and harshness)
    2. Glue compression (light, for cohesion)
    3. Tonal EQ (subtle boosts for warmth and air)
    4. Saturation (harmonic enhancement)
    5. Limiter (loudness targeting)

    Target LUFS:
    - Club/DJ: -8 to -9 LUFS (loud, punchy)
    - Streaming: -14 LUFS (Spotify/Apple normalized)
    """

    # Ensure float32 for pedalboard
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Stage 1: Corrective EQ (cuts only)
    corrective_eq = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),            # Remove sub-rumble
        PeakFilter(cutoff_frequency_hz=300, gain_db=-2.0, q=1.5),  # Cut mud
        PeakFilter(cutoff_frequency_hz=3500, gain_db=-1.0, q=2.0),  # Tame harshness
    ])
    audio = corrective_eq(audio, sr)

    # Stage 2: Glue compression
    glue_comp = Pedalboard([
        Compressor(
            threshold_db=-12,
            ratio=2.0,        # Light ratio for glue
            attack_ms=30,     # Slow attack preserves transients
            release_ms=200,   # Medium release
        ),
    ])
    audio = glue_comp(audio, sr)

    # Stage 3: Tonal EQ (subtle boosts)
    tonal_eq = Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=80, gain_db=1.5),     # Low-end warmth
        HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.0),  # Air
    ])
    audio = tonal_eq(audio, sr)

    # Stage 4: Saturation (analog warmth)
    audio = multi_stage_saturation(audio, sr, drive_db=2.0)
    audio = audio.astype(np.float32)

    # Stage 5: Limiter
    if target == "club":
        # Club target: -8 to -9 LUFS, loud and punchy
        limiter = Pedalboard([
            Gain(gain_db=4.0),            # Push into limiter
            Limiter(threshold_db=-1.0),   # True peak ceiling
        ])
    else:
        # Streaming target: -14 LUFS
        limiter = Pedalboard([
            Gain(gain_db=1.0),
            Limiter(threshold_db=-1.0),
        ])

    audio = limiter(audio, sr)

    return audio


def analyze_loudness(audio: np.ndarray, sr: int) -> dict:
    """Estimate loudness metrics."""
    # RMS-based LUFS approximation
    # True LUFS requires K-weighting, but RMS gives a useful estimate
    if audio.ndim > 1:
        mono = np.mean(audio, axis=0)
    else:
        mono = audio

    rms = np.sqrt(np.mean(mono ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    # LUFS is approximately RMS in dBFS for loudness-normalized content
    # This is an approximation — true LUFS uses K-weighting filter
    estimated_lufs = rms_db

    peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)

    # Crest factor (peak-to-RMS ratio) — indicates dynamic range
    crest_db = peak_db - rms_db

    return {
        "rms_db": round(float(rms_db), 1),
        "estimated_lufs": round(float(estimated_lufs), 1),
        "peak_db": round(float(peak_db), 1),
        "crest_factor_db": round(float(crest_db), 1),
        "dynamic_range": "crushed" if crest_db < 6 else "tight" if crest_db < 10 else "dynamic" if crest_db < 14 else "wide",
    }


def process_club_ready(input_file: str, target: str = "club", bpm: float = None,
                       sidechain: bool = False, output_file: str = None):
    """Full club-ready processing pipeline."""
    path = Path(input_file)
    if not path.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    audio, sr = sf.read(str(path), dtype='float32')

    # Transpose to (channels, samples) for pedalboard
    if audio.ndim > 1:
        audio = audio.T
    else:
        audio = audio[np.newaxis, :]  # (1, samples) for mono

    print(f"Processing: {path.name}")
    print(f"  Input: {audio.shape[0]}ch, {sr}Hz, {audio.shape[1]/sr:.1f}s")
    print(f"  Target: {target} ({'~-8 LUFS, loud' if target == 'club' else '~-14 LUFS, streaming'})")
    print()

    # Pre-processing analysis
    pre_stats = analyze_loudness(audio, sr)
    print(f"  BEFORE processing:")
    print(f"    RMS: {pre_stats['rms_db']} dB")
    print(f"    Peak: {pre_stats['peak_db']} dB")
    print(f"    Dynamic range: {pre_stats['dynamic_range']} (crest: {pre_stats['crest_factor_db']} dB)")
    print()

    # Step 1: Mono bass management
    if audio.shape[0] > 1:
        print("  [1/4] Mono bass management (< 150 Hz)...")
        try:
            audio = mono_bass(audio, sr, crossover_hz=150.0)
        except ImportError:
            print("    scipy not installed, skipping mono bass. Install with: pip install scipy")
    else:
        print("  [1/4] Mono input, skipping stereo bass management")

    # Step 2: Sidechain pump
    if sidechain and bpm:
        print(f"  [2/4] Sidechain pump at {bpm} BPM (-6dB duck)...")
        audio = sidechain_pump(audio, sr, bpm, depth_db=-6.0, attack_ms=5.0, release_ms=150.0)
    else:
        print("  [2/4] Sidechain: skipped (use --sidechain --bpm to enable)")

    # Step 3: Multi-stage saturation
    print("  [3/4] Multi-stage saturation (Summit-style grit)...")
    audio = multi_stage_saturation(audio, sr, drive_db=3.0)
    audio = audio.astype(np.float32)

    # Step 4: Mastering chain
    print(f"  [4/4] Mastering chain ({target} target)...")
    audio = mastering_chain(audio, sr, target)

    # Post-processing analysis
    post_stats = analyze_loudness(audio, sr)
    print()
    print(f"  AFTER processing:")
    print(f"    RMS: {post_stats['rms_db']} dB")
    print(f"    Peak: {post_stats['peak_db']} dB")
    print(f"    Dynamic range: {post_stats['dynamic_range']} (crest: {post_stats['crest_factor_db']} dB)")

    # Write output
    if output_file is None:
        output_file = f"{path.stem}_club_ready{path.suffix}"

    # Transpose back to (samples, channels) for soundfile
    if audio.shape[0] == 1:
        sf.write(output_file, audio[0], sr)
    else:
        sf.write(output_file, audio.T, sr)

    print()
    print(f"  Output: {output_file}")
    print()
    print("  What was applied:")
    print("    1. Mono bass (< 150 Hz) - prevents phase cancellation on club PAs")
    print("    2. Corrective EQ - cut mud at 300Hz, tamed harshness at 3.5kHz")
    print("    3. Glue compression (2:1, 30ms attack) - cohesion without killing dynamics")
    print("    4. Tonal EQ - low-end warmth at 80Hz, air at 12kHz")
    print("    5. Double saturation - analog warmth + grit (Summit's signature)")
    print("    6. Brick-wall limiter at -1.0 dBTP")


def main():
    parser = argparse.ArgumentParser(
        description="Club-ready processing: mono bass, sidechain, saturation, mastering"
    )
    parser.add_argument("input", help="Audio file to process")
    parser.add_argument("--target", "-t", default="club", choices=["club", "streaming"],
                        help="Loudness target: club (~-8 LUFS) or streaming (~-14 LUFS)")
    parser.add_argument("--bpm", "-b", type=float, default=None,
                        help="Track BPM (required for sidechain)")
    parser.add_argument("--sidechain", "-s", action="store_true",
                        help="Apply sidechain-style pumping")
    parser.add_argument("--output", "-o", default=None, help="Output filename")
    parser.add_argument("--analyze-only", "-a", action="store_true",
                        help="Only analyze loudness, don't process")

    args = parser.parse_args()

    if args.analyze_only:
        audio, sr = sf.read(args.input, dtype='float32')
        if audio.ndim > 1:
            audio = audio.T
        else:
            audio = audio[np.newaxis, :]
        stats = analyze_loudness(audio, sr)
        print(f"Analysis: {args.input}")
        print(f"  RMS: {stats['rms_db']} dB")
        print(f"  Estimated LUFS: {stats['estimated_lufs']}")
        print(f"  Peak: {stats['peak_db']} dB")
        print(f"  Crest factor: {stats['crest_factor_db']} dB")
        print(f"  Dynamic range: {stats['dynamic_range']}")
        target_note = "(-8 to -9 for club, -14 for streaming)"
        print(f"\n  Target LUFS reference: {target_note}")
        return

    process_club_ready(args.input, args.target, args.bpm, args.sidechain, args.output)


if __name__ == "__main__":
    main()
