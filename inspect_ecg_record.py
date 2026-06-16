"""
Inspect ECG record — extract all information from a WFDB record file.
Usage: python inspect_ecg_record.py <record_path_without_extension>
Example: python inspect_ecg_record.py C:/LoyaltyLo/datasets/mit-bih-atrial-fibrillation-database-1.0.0/04043
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import wfdb
import numpy as np

HEADER_FIELDS = [
    'record_name', 'n_sig', 'fs', 'counter_freq', 'base_counter', 'sig_len',
    'base_time', 'base_date', 'comments', 'sig_name', 'units', 'adc_gain',
    'baseline', 'adc_res', 'adc_zero', 'init_value', 'checksum', 'block_size',
]


def print_header(record_path):
    """Read and display record header."""
    print(f"\n{'='*70}")
    print(f"  HEADER: {os.path.basename(record_path)}.hea")
    print(f"{'='*70}")
    try:
        header = wfdb.rdheader(record_path)
        for field in HEADER_FIELDS:
            val = getattr(header, field, None)
            if val is not None:
                print(f"  {field:<18s}: {val}")
    except Exception as e:
        print(f"  [ERR] Cannot read header: {e}")


def print_signal_stats(record_path):
    """Read and display signal statistics."""
    print(f"\n{'='*70}")
    print(f"  SIGNALS: {os.path.basename(record_path)}.dat")
    print(f"{'='*70}")
    try:
        signals, fields = wfdb.rdsamp(record_path)
        print(f"  Shape:        {signals.shape}")
        print(f"  Sample rate:  {fields['fs']} Hz")
        print(f"  Duration:     {signals.shape[0] / fields['fs'] / 3600:.2f} hours")
        print(f"  Signal names: {fields.get('sig_name', 'unknown')}")
        print(f"  Units:        {fields.get('units', 'unknown')}")

        for i in range(signals.shape[1]):
            sig = signals[:, i]
            name = fields.get('sig_name', [f'ch{i}'])[i] if isinstance(fields.get('sig_name'), list) else f'ch{i}'
            print(f"\n  --- Channel {i} ({name}) ---")
            print(f"    Min:     {np.min(sig):.2f}")
            print(f"    Max:     {np.max(sig):.2f}")
            print(f"    Mean:    {np.mean(sig):.2f}")
            print(f"    Std:     {np.std(sig):.2f}")
            print(f"    Median:  {np.median(sig):.2f}")
            print(f"    IQR:     {np.percentile(sig, 75) - np.percentile(sig, 25):.2f}")
            print(f"    NaN%:    {np.isnan(sig).mean() * 100:.4f}%")
    except Exception as e:
        print(f"  [ERR] Cannot read signal: {e}")


def print_annotation_stats(record_path, ext, label=""):
    """Read and display annotation statistics."""
    print(f"\n{'='*70}")
    print(f"  ANNOTATIONS: {os.path.basename(record_path)}.{ext} {label}")
    print(f"{'='*70}")
    try:
        ann = wfdb.rdann(record_path, ext)
        print(f"  Annotation extension: {ext}")
        print(f"  Full path:  {ann.ann_len} annotations")
        print(f"  Sample rate: {ann.fs} Hz")
        print(f"  Annotation fields: {[a for a in dir(ann) if not a.startswith('_') and a != 'blob']}")

        # Sample positions and labels
        symbols = ann.symbol if hasattr(ann, 'symbol') else [''] * ann.ann_len
        samples = ann.sample

        print(f"\n  --- Symbol distribution (first 1000 and last 1000) ---")
        total = len(symbols)
        if total > 2000:
            # For performance, sample first 1000 + last 1000
            sym_subset = list(symbols[:1000]) + list(symbols[-1000:])
            print(f"  (sampled {len(sym_subset)} of {total} total)")
        else:
            sym_subset = symbols

        from collections import Counter
        counter = Counter(str(s) for s in sym_subset)
        for sym, count in counter.most_common(30):
            pct = count / len(sym_subset) * 100
            print(f"    '{sym}': {count:>6d} ({pct:5.1f}%)")

        # Sample positions
        if len(samples) > 1:
            intervals = np.diff(samples) / ann.fs * 1000.0
            print(f"\n  --- Inter-annotation intervals (ms) ---")
            print(f"    Count:   {len(intervals)}")
            print(f"    Mean:    {np.mean(intervals):.1f} ms")
            print(f"    Median:  {np.median(intervals):.1f} ms")
            print(f"    Std:     {np.std(intervals):.1f} ms")
            print(f"    Min:     {np.min(intervals):.1f} ms")
            print(f"    Max:     {np.max(intervals):.1f} ms")
            print(f"    % < 400: {np.mean(intervals < 400) * 100:.2f}%")
            print(f"    % > 1500:{np.mean(intervals > 1500) * 100:.2f}%")

        # Aux note analysis
        if hasattr(ann, 'aux_note') and ann.aux_note:
            notes = [str(n) for n in ann.aux_note if n]
            note_counter = Counter(notes)
            print(f"\n  --- Aux note distribution ---")
            for note, count in note_counter.most_common(20):
                print(f"    '{note}': {count}")

        # Beat-level rhythm annotation mapping
        if hasattr(ann, 'aux_note') and ann.aux_note:
            rhythm_changes = [(ann.sample[i], ann.aux_note[i])
                              for i in range(len(ann.aux_note))
                              if ann.aux_note[i]]
            print(f"\n  --- Rhythm changes (first 30) ---")
            for sample, note in rhythm_changes[:30]:
                time_min = sample / ann.fs / 60.0
                print(f"    @ {time_min:.1f} min (sample {sample}): {note}")
            if len(rhythm_changes) > 30:
                print(f"    ... and {len(rhythm_changes) - 30} more")

    except Exception as e:
        print(f"  [WARN] No .{ext} annotation: {e}")


def print_rr_interval_stats(record_path, fs=128):
    """Derive RR intervals from beat annotations and print statistics."""
    print(f"\n{'='*70}")
    print(f"  RR INTERVAL ANALYSIS (derived from beat annotations)")
    print(f"{'='*70}")

    # Try different beat annotation extensions
    beat_exts = ['qrs', 'atr', 'ecg']
    beat_ann = None
    ann_ext = None
    for ext in beat_exts:
        try:
            beat_ann = wfdb.rdann(record_path, ext)
            ann_ext = ext
            break
        except Exception:
            continue

    if beat_ann is None:
        print("  [WARN] No beat annotation found (tried .qrs, .atr, .ecg)")
        return

    print(f"  Source: .{ann_ext} ({beat_ann.ann_len} beats)")

    samples = beat_ann.sample
    rr = np.diff(samples) / beat_ann.fs * 1000.0

    if len(rr) < 2:
        print("  [WARN] Too few beats for RR analysis")
        return

    # Basic stats
    print(f"\n  --- RR Interval Statistics (raw) ---")
    print(f"    Total beats:   {len(samples)}")
    print(f"    RR count:      {len(rr)}")
    print(f"    Mean RR:       {np.mean(rr):.1f} ms ({60 / np.mean(rr) * 1000:.1f} bpm)")
    print(f"    Median RR:     {np.median(rr):.1f} ms")
    print(f"    Std RR:        {np.std(rr):.1f} ms")
    print(f"    RMSSD:         {np.sqrt(np.mean(np.diff(rr) ** 2)):.1f} ms")
    print(f"    Min RR:        {np.min(rr):.1f} ms")
    print(f"    Max RR:        {np.max(rr):.1f} ms")
    print(f"    pNN50:         {np.mean(np.abs(np.diff(rr)) > 50) * 100:.1f}%")

    # Filtered stats
    rr_filtered = rr[(rr >= 300) & (rr <= 2000)]
    if len(rr_filtered) >= 2:
        print(f"\n  --- RR Interval Statistics (300-2000ms filtered) ---")
        print(f"    Valid beats:   {len(rr_filtered) + 1}")
        print(f"    Outlier %:     {(1 - len(rr_filtered) / len(rr)) * 100:.2f}%")
        print(f"    Mean RR:       {np.mean(rr_filtered):.1f} ms")
        print(f"    Median RR:     {np.median(rr_filtered):.1f} ms")
        print(f"    Std RR:        {np.std(rr_filtered):.1f} ms")
        print(f"    CV:            {np.std(rr_filtered) / np.mean(rr_filtered):.4f}")
        print(f"    RMSSD:         {np.sqrt(np.mean(np.diff(rr_filtered) ** 2)):.1f} ms")
        print(f"    Min RR:        {np.min(rr_filtered):.1f} ms")
        print(f"    Max RR:        {np.max(rr_filtered):.1f} ms")
        print(f"    pNN50:         {np.mean(np.abs(np.diff(rr_filtered)) > 50) * 100:.1f}%")
        print(f"    pNN20:         {np.mean(np.abs(np.diff(rr_filtered)) > 20) * 100:.1f}%")

    # Heart rate
    hr = 60000.0 / rr_filtered if len(rr_filtered) >= 2 else np.array([])
    if len(hr) >= 2:
        print(f"\n  --- Heart Rate Statistics (bpm) ---")
        print(f"    Mean HR:       {np.mean(hr):.1f} bpm")
        print(f"    Median HR:     {np.median(hr):.1f} bpm")
        print(f"    Min HR:        {np.min(hr):.1f} bpm")
        print(f"    Max HR:        {np.max(hr):.1f} bpm")
        print(f"    % HR < 50:     {np.mean(hr < 50) * 100:.1f}%")
        print(f"    % HR > 100:    {np.mean(hr > 100) * 100:.1f}%")
        print(f"    % HR > 120:    {np.mean(hr > 120) * 100:.1f}%")

    # Rhythm analysis (if aux_notes available)
    print_rhythm_summary(beat_ann, rr)


def print_rhythm_summary(beat_ann, rr):
    """Print rhythm/label statistics from aux_note annotations."""
    if not hasattr(beat_ann, 'aux_note') or not beat_ann.aux_note:
        return

    afib_beats = 0
    nsr_beats = 0
    other_beats = 0
    beat_labels = [''] * beat_ann.ann_len
    current_rhythm = ''
    rhythm_samples = beat_ann.sample
    rhythm_notes = beat_ann.aux_note

    for i, bs in enumerate(beat_ann.sample):
        matched = False
        for j, rs in enumerate(rhythm_samples):
            if bs >= rs:
                matched = True
                if rhythm_notes[j]:
                    current_rhythm = rhythm_notes[j]
            else:
                break
        beat_labels[i] = current_rhythm
        if '(AFIB' in str(current_rhythm).upper():
            afib_beats += 1
        elif '(N' in str(current_rhythm).upper() and '(AFIB' not in str(current_rhythm).upper():
            nsr_beats += 1
        elif current_rhythm:
            other_beats += 1

    total = beat_ann.ann_len
    print(f"\n  --- Rhythm Summary ---")
    print(f"    AFib beats:     {afib_beats} ({afib_beats / max(total, 1) * 100:.1f}%)")
    print(f"    NSR beats:      {nsr_beats} ({nsr_beats / max(total, 1) * 100:.1f}%)")
    print(f"    Other/unknown:  {other_beats} ({other_beats / max(total, 1) * 100:.1f}%)")

    # AFib episode detection
    episodes = []
    in_afib = False
    start_idx = None
    for i, label in enumerate(beat_labels):
        note = str(label).upper()
        if '(AFIB' in note and not in_afib:
            in_afib = True
            start_idx = i
        elif '(' in note and '(AFIB' not in note and in_afib:
            in_afib = False
            episodes.append({'start': start_idx, 'end': i - 1})
    if in_afib:
        episodes.append({'start': start_idx, 'end': len(beat_labels) - 1})

    if episodes:
        print(f"\n  --- AFib Episodes ({len(episodes)}) ---")
        durations = []
        for ep in episodes:
            dur = (beat_ann.sample[min(ep['end'], len(beat_ann.sample) - 1)] -
                   beat_ann.sample[ep['start']]) / beat_ann.fs / 60.0
            durations.append(dur)
        print(f"    Count:         {len(episodes)}")
        print(f"    Mean duration: {np.mean(durations):.1f} min")
        print(f"    Median:        {np.median(durations):.1f} min")
        print(f"    Min:           {np.min(durations):.1f} min")
        print(f"    Max:           {np.max(durations):.1f} min")
        print(f"    Total:         {np.sum(durations):.1f} min ({np.sum(durations) / 60:.1f} hrs)")
        print(f"    Burdensome %:  {np.sum(durations) / max(len(rr) * np.mean(rr) / 60000.0, 1) * 100:.1f}%")

        # Show first 10 episodes
        print(f"\n  --- First 10 Episodes ---")
        for i, ep in enumerate(episodes[:10]):
            t_start = beat_ann.sample[ep['start']] / beat_ann.fs / 60.0
            t_end = beat_ann.sample[min(ep['end'], len(beat_ann.sample) - 1)] / beat_ann.fs / 60.0
            dur = t_end - t_start
            print(f"    #{i+1}: {t_start:.1f} → {t_end:.1f} min ({dur:.1f} min)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_ecg_record.py <record_path_without_extension>")
        print("Example: python inspect_ecg_record.py C:/LoyaltyLo/datasets/mit-bih-atrial-fibrillation-database-1.0.0/04043")
        sys.exit(1)

    record_path = sys.argv[1]
    if not os.path.exists(os.path.dirname(record_path)):
        print(f"Error: directory not found: {os.path.dirname(record_path)}")
        sys.exit(1)

    base_name = os.path.basename(record_path)
    print(f"Record: {base_name}")
    print(f"Path:   {os.path.dirname(record_path)}")

    print_header(record_path)

    # Check if signal file exists
    if os.path.exists(record_path + '.dat'):
        print_signal_stats(record_path)

    # List available annotation files
    hea_dir = os.path.dirname(record_path)
    hea_name = os.path.basename(record_path)
    all_files = os.listdir(hea_dir)
    annot_exts = sorted(set(
        f.split('.')[-1] for f in all_files
        if f.startswith(hea_name + '.') and len(f.split('.')) == 2
    ))
    annot_exts = [e for e in annot_exts if e not in ('dat', 'hea')]
    print(f"\n  Available annotations: {annot_exts}")

    for ext in annot_exts:
        print_annotation_stats(record_path, ext)

    print_rr_interval_stats(record_path)

    print(f"\n{'='*70}")
    print(f"  END OF REPORT")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
