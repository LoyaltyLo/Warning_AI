"""
Deep diagnostic script: analyze false positive window feature patterns.
Extracts 15-dim features for FP vs Normal windows to identify root causes.
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import joblib
import numpy as np
import wfdb

from batch_evaluate_cdss import (extract_features, _suppress_bottom_noise, _ewm_smooth,
    _compute_trend, _compute_adaptive_thresholds, _adaptive_alert, extract_ai_alarm_episodes)
from scipy.signal import savgol_filter

device = torch.device("cpu")

FEATURE_NAMES = [
    'cv_suppressed', 'mad', 'rmssd_suppressed', 'pnn50_suppressed',
    'samp_en', 'dfa_alpha1', 'pip', 'sd1', 'poincare_ratio',
    'lf_hf_ratio', 'pip_raw', 'dfa_raw', 'bigeminy_corr', 'bimodality_ratio',
    'qrs_template_corr', 'qrs_outlier_ratio', 'inter_lead_corr_drop'
]

WINDOW_BEATS_VAL = 600
STEP_BEATS_VAL = 120
TIME_STEPS = 6


def load_model_and_scaler(model_path, scaler_path):
    from train import AFibAttentionSeq2Seq
    model = AFibAttentionSeq2Seq(input_dim=18, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler


def analyze_patient(record_path, model, scaler):
    patient_id = os.path.basename(record_path)

    signals, fields = wfdb.rdsamp(record_path)
    ecg_signal = signals[:, 0]
    fs = fields['fs']

    if os.path.exists(record_path + '.atr'):
        ann_atr = wfdb.rdann(record_path, 'atr')
    elif os.path.exists(record_path + '.ecg'):
        ann_atr = wfdb.rdann(record_path, 'ecg')
    else:
        return None

    beat_samples = ann_atr.sample
    beat_labels = ann_atr.symbol if hasattr(ann_atr, 'symbol') else ['N'] * len(beat_samples)
    rr_intervals = np.diff(beat_samples) / fs * 1000.0

    all_features = []
    valid_starts = []

    for start in range(0, len(rr_intervals) - WINDOW_BEATS_VAL + 1, STEP_BEATS_VAL):
        end = start + WINDOW_BEATS_VAL
        window_rr = rr_intervals[start:end]
        window_labels = beat_labels[start+1:end+1] if len(beat_labels) > end else ['N'] * len(window_rr)
        feat = extract_features(window_rr, window_labels)
        if feat is not None:
            all_features.append(feat)
            valid_starts.append(start)

    if len(all_features) < TIME_STEPS:
        return None

    all_features = np.array(all_features)

    X = []
    for i in range(len(all_features) - TIME_STEPS + 1):
        seq = all_features[i:i+TIME_STEPS].copy()
        seq_full = np.zeros((TIME_STEPS, 18))
        seq_full[:, :17] = seq
        seq_full[:, 17] = np.gradient(seq[:, 4])
        X.append(seq_full)

    X = np.array(X)
    if len(X) == 0:
        return None

    N, T, D = X.shape
    X_flat = X.reshape(-1, D)
    X_scaled = scaler.transform(X_flat).reshape(N, T, D)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        probs, _ = model(X_tensor)
    raw_probs = probs[:, -1].cpu().numpy()

    suppressed = _suppress_bottom_noise(raw_probs, threshold=0.20, exponent=1.8)
    ewm = _ewm_smooth(suppressed, span=5)
    if len(ewm) >= 11:
        smoothed = savgol_filter(ewm, window_length=11, polyorder=2)
        smoothed = np.clip(smoothed, 0.0, 1.0)
    else:
        smoothed = ewm

    trend = _compute_trend(smoothed, window=3)
    (p1_enter, p2_enter, p3_enter, p3_trend, exit_thresh, display_thresh,
     p1_sustain, p3_sustain) = _compute_adaptive_thresholds(smoothed, calibration_windows=30)

    active = _adaptive_alert(smoothed, trend, p1_enter, p2_enter, p3_enter,
                             p3_trend, exit_thresh, p1_sustain, p3_sustain)

    cum_time = np.cumsum(np.concatenate([[0], rr_intervals])) / 1000.0 / 60.0
    window_times = []
    for start in valid_starts:
        idx = min(start, len(cum_time) - 1)
        window_times.append(cum_time[idx])
    window_times = np.array(window_times)

    alarm_times = window_times[TIME_STEPS-1:]
    alarms = extract_ai_alarm_episodes(alarm_times, active)

    # GT episodes from annotations
    gt_episodes = []
    if hasattr(ann_atr, 'aux_note'):
        for i, note in enumerate(ann_atr.aux_note):
            if 'AF' in str(note).upper() or 'AFIB' in str(note).upper():
                gt_episodes.append({
                    'start': ann_atr.sample[i] / fs / 60.0,
                    'end': ann_atr.sample[i] / fs / 60.0 + 1.0
                })

    fp_windows = []
    valid_windows = []
    for alarm in alarms:
        trigger_min = alarm['start']
        is_fp = True
        for gt in gt_episodes:
            if trigger_min <= gt['end'] and trigger_min >= (gt['start'] - 120.0):
                is_fp = False
                break

        window_idx = np.argmin(np.abs(alarm_times - trigger_min))
        if window_idx < len(raw_probs):
            info = {
                'time_min': trigger_min,
                'window_idx': window_idx,
                'raw_prob': raw_probs[window_idx],
                'smoothed_prob': smoothed[window_idx],
                'features_last': all_features[min(window_idx + TIME_STEPS - 1, len(all_features) - 1)].copy(),
            }
            if is_fp:
                fp_windows.append(info)
            else:
                valid_windows.append(info)

    # Normal windows (no alarm, sampled evenly)
    normal_windows = []
    alert_set = set()
    for a in alarms:
        for t in np.arange(a['start'], a['end'], 0.2):
            idx = np.argmin(np.abs(alarm_times - t))
            alert_set.add(idx)

    for i in range(0, min(len(raw_probs), 500), max(1, len(raw_probs) // 100)):
        if i not in alert_set and raw_probs[i] < display_thresh * 0.8:
            normal_windows.append({
                'window_idx': i,
                'raw_prob': raw_probs[i],
                'features_last': all_features[min(i + TIME_STEPS - 1, len(all_features) - 1)].copy(),
            })
        if len(normal_windows) >= 80:
            break

    return {
        'patient_id': patient_id,
        'fp_windows': fp_windows,
        'valid_windows': valid_windows,
        'normal_windows': normal_windows,
        'raw_probs': raw_probs,
        'threshold': display_thresh,
        'n_windows': len(raw_probs),
        'n_alarms': len(alarms),
        'n_fp': len(fp_windows),
        'n_valid': len(valid_windows),
    }


def print_feature_analysis(results):
    if not results or results['n_fp'] == 0:
        if results:
            print(f"\n  Patient {results['patient_id']}: [OK] No FP")
        return

    fp_count = results['n_fp']
    total = results['n_windows']
    print(f"\n{'='*70}")
    print(f"  Patient {results['patient_id']}: {fp_count} FP / {results['n_alarms']} alarms / {total} windows")
    print(f"  Threshold: {results['threshold']:.3f}")

    fp_w = results['fp_windows']
    norm_w = results['normal_windows']
    valid_w = results['valid_windows']

    fp_mean_prob = np.mean([w['raw_prob'] for w in fp_w])
    norm_mean_prob = np.mean([w['raw_prob'] for w in norm_w]) if norm_w else 0.01
    valid_mean_prob = np.mean([w['raw_prob'] for w in valid_w]) if valid_w else 0

    print(f"\n  Mean Raw Probabilities:")
    print(f"    FP windows:      {fp_mean_prob:.4f}")
    print(f"    Valid windows:   {valid_mean_prob:.4f}")
    print(f"    Normal windows:  {norm_mean_prob:.4f}")
    print(f"    FP/Normal ratio: {fp_mean_prob / max(norm_mean_prob, 1e-6):.2f}x")

    if len(fp_w) == 0 or len(norm_w) == 0:
        return

    fp_feats = np.array([w['features_last'] for w in fp_w])
    norm_feats = np.array([w['features_last'] for w in norm_w])

    print(f"\n  Feature Deviation Analysis (FP vs Normal):")
    print(f"  {'Feature':<20s} {'FP mean':>10s} {'Norm mean':>10s} {'Diff%':>8s} {'Z-score':>8s}")
    print(f"  {'-'*60}")

    deviations = []
    for i, name in enumerate(FEATURE_NAMES):
        if i >= fp_feats.shape[1]:
            break
        fp_mean = np.mean(fp_feats[:, i])
        norm_mean = np.mean(norm_feats[:, i])
        norm_std = np.std(norm_feats[:, i]) + 1e-10
        pct_diff = (fp_mean - norm_mean) / max(abs(norm_mean), 1e-10) * 100
        z_score = (fp_mean - norm_mean) / norm_std

        marker = ""
        if abs(z_score) > 2.0:
            marker = " ***"
        elif abs(z_score) > 1.0:
            marker = " **"

        print(f"  {name:<20s} {fp_mean:>10.4f} {norm_mean:>10.4f} {pct_diff:>7.1f}% {z_score:>8.2f}{marker}")
        deviations.append((name, abs(z_score), z_score, fp_mean, norm_mean))

    deviations.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top-5 Discriminating Features (by |Z-score|):")
    for i, (name, abs_z, z, fp_m, n_m) in enumerate(deviations[:5]):
        direction = "HIGHER in FP" if z > 0 else "LOWER in FP"
        print(f"    {i+1}. {name}: Z={z:+.2f} ({direction}) | FP={fp_m:.4f} vs Norm={n_m:.4f}")

    # Time clustering
    if len(fp_w) >= 3:
        times = sorted([w['time_min'] for w in fp_w])
        diffs = np.diff(times)
        clusters = sum(1 for d in diffs if d < 10)
        print(f"\n  Time Analysis: {len(times)} FPs in [{times[0]:.0f}-{times[-1]:.0f}]min")
        print(f"    Mean interval: {np.mean(diffs):.1f}min, Median: {np.median(diffs):.1f}min")
        print(f"    Short-interval clusters (<10min): {clusters}/{len(diffs)}")


if __name__ == "__main__":
    model_path = "best_afib_model.pth"
    scaler_path = "feature_scaler.pkl"

    print("Loading model...")
    model, scaler = load_model_and_scaler(model_path, scaler_path)

    targets = [
        ("NSR", r"C:\LoyaltyLo\datasets\mit-bih-normal-sinus-rhythm-database-1.0.0\mit-bih-normal-sinus-rhythm-database-1.0.0\16272"),
        ("NSR", r"C:\LoyaltyLo\datasets\mit-bih-normal-sinus-rhythm-database-1.0.0\mit-bih-normal-sinus-rhythm-database-1.0.0\19093"),
        ("NSR", r"C:\LoyaltyLo\datasets\mit-bih-normal-sinus-rhythm-database-1.0.0\mit-bih-normal-sinus-rhythm-database-1.0.0\16539"),
        ("AFib", r"C:\LoyaltyLo\datasets\mit-bih-atrial-fibrillation-database-1.0.0\06453"),
        ("AFib", r"C:\LoyaltyLo\datasets\mit-bih-atrial-fibrillation-database-1.0.0\07910"),
        ("AFib", r"C:\LoyaltyLo\datasets\mit-bih-atrial-fibrillation-database-1.0.0\04126"),
    ]

    all_results = []
    for dtype, rec in targets:
        if os.path.exists(rec + '.dat'):
            try:
                results = analyze_patient(rec, model, scaler)
                if results:
                    print_feature_analysis(results)
                    all_results.append(results)
            except Exception as e:
                print(f"  [ERR] {os.path.basename(rec)}: {e}")
        else:
            print(f"  [SKIP] {os.path.basename(rec)}: no .dat file")

    # Cross-patient summary
    print(f"\n{'='*70}")
    print(f"  CROSS-PATIENT SUMMARY")
    print(f"{'='*70}")

    for results in all_results:
        if results['n_fp'] == 0:
            continue
        fp_w = results['fp_windows']
        print(f"\n  {results['patient_id']} ({results['n_fp']} FPs):")
        print(f"    Mean raw prob at FP: {np.mean([w['raw_prob'] for w in fp_w]):.4f}")
        print(f"    Raw prob range: [{min(w['raw_prob'] for w in fp_w):.4f} - {max(w['raw_prob'] for w in fp_w):.4f}]")
        print(f"    Threshold: {results['threshold']:.4f}")
