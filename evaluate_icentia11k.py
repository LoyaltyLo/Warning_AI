"""
Evaluate 15D model on Icentia11k (independent NSR dataset).
Icentia11k is completely separate from training data — no data leakage.
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import wfdb
import torch
import numpy as np
import joblib
import multiprocessing as mp
from tqdm import tqdm
from scipy.signal import welch, medfilt, savgol_filter
from scipy.interpolate import interp1d
import antropy as ant
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train import AFibAttentionSeq2Seq
from batch_evaluate_cdss import (
    _interpolate_nans, _ewm_smooth, _suppress_bottom_noise,
    _compute_trend, _compute_adaptive_thresholds, _compute_rolling_thresholds,
    _adaptive_alert, _suppress_flatline_probs,
    extract_ai_alarm_episodes, extract_features
)
from logging_config import setup_logging, get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)

WINDOW_BEATS = 600
STEP_BEATS = 30
TIME_STEPS = 6


def evaluate_icentia11k_patient(args):
    record_path, model_path, scaler_path, plot_out_dir = args
    patient_id = os.path.basename(record_path)

    try:
        ann = wfdb.rdann(record_path, 'atr')
        fs = getattr(ann, 'fs', 250)
        all_samples = ann.sample
        all_symbols = [str(s).strip() for s in ann.symbol]
    except Exception as e:
        return {'status': 'error', 'msg': f"Failed to read: {e}"}

    # Filter '+' markers (rhythm changes, not beats)
    beat_mask = [s != '+' for s in all_symbols]
    beat_samples_raw = np.array(all_samples)[beat_mask]
    beat_labels = np.array(all_symbols)[beat_mask]

    if len(beat_samples_raw) < WINDOW_BEATS * TIME_STEPS:
        return {'status': 'skipped', 'msg': f'Too few beats: {len(beat_samples_raw)}'}

    beat_samples = beat_samples_raw

    # 🎯 重采样到128Hz：Icentia11k原生250Hz，需统一到训练数据的128Hz
    if fs != 128:
        beat_samples = np.round(beat_samples_raw * 128.0 / fs).astype(int)
        fs = 128

    # RR intervals in ms (computed from 128Hz-resampled samples, consistent with training)
    rr_intervals = np.zeros(len(beat_samples)) 
    rr_intervals[1:] = np.diff(beat_samples) / fs * 1000.0
    rr_intervals[0] = np.mean(rr_intervals[1:11]) if len(rr_intervals) > 11 else rr_intervals[1]
    rr_intervals = np.clip(rr_intervals, 300, 3000)

    # Cumulative time
    cum_time_mins = np.zeros(len(rr_intervals))
    if len(rr_intervals) > 1:
        cum_time_mins[1:] = np.cumsum(rr_intervals[1:]) / 60000.0

    total_duration_mins = cum_time_mins[-1]

    # Load model
    device = torch.device("cpu")
    model = AFibAttentionSeq2Seq(input_dim=16, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    scaler = joblib.load(scaler_path)

    history_beats = WINDOW_BEATS * TIME_STEPS
    time_axis_mins = []
    all_raw_seqs = []
    pip_values = []
    current_beat = history_beats
    last_valid_features = [0.0] * 15  # 15 HRV features

    while current_beat < len(rr_intervals):
        sequence_features = []
        for i in range(TIME_STEPS):
            w_end = current_beat - (TIME_STEPS - 1 - i) * WINDOW_BEATS
            w_start = w_end - WINDOW_BEATS
            rr_window = rr_intervals[w_start:w_end]
            notes_window = beat_labels[w_start:w_end]
            feats_hrv = extract_features(rr_window, notes_window)

            if feats_hrv is not None:
                feats = feats_hrv  # 15D HRV features
                sequence_features.append(feats)
                last_valid_features = feats
            else:
                sequence_features.append(last_valid_features)

        final_seq_feats = []
        for j in range(len(sequence_features)):
            curr = sequence_features[j].copy()
            delta_en = 0.0 if j == 0 else curr[4] - sequence_features[j - 1][4]
            curr.append(delta_en)
            final_seq_feats.append(curr)

        all_raw_seqs.append(final_seq_feats)
        pip_values.append(final_seq_feats[-1][6])
        time_axis_mins.append(cum_time_mins[current_beat])
        current_beat += STEP_BEATS

    if len(all_raw_seqs) == 0:
        return {'status': 'skipped', 'msg': 'No valid features'}

    X_raw_array = np.array(all_raw_seqs)
    X_flat = X_raw_array.reshape(-1, 16)
    X_norm_flat = scaler.transform(X_flat)
    X_norm_array = X_norm_flat.reshape(-1, TIME_STEPS, 16)

    X_tensor = torch.tensor(X_norm_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs, _ = model(X_tensor)
        ai_risk_probs = probs[:, -1].cpu().numpy()

    # Post-processing pipeline (与 batch_evaluate_cdss.py v4 一致)
    raw_probs = np.array(ai_risk_probs)
    suppressed = _suppress_bottom_noise(raw_probs, threshold=0.25, exponent=1.8)
    ewm_smoothed = _ewm_smooth(suppressed, span=5)
    if len(ewm_smoothed) >= 11:
        smoothed_probs = savgol_filter(ewm_smoothed, window_length=11, polyorder=2)
        smoothed_probs = np.clip(smoothed_probs, 0.0, 1.0)
    else:
        smoothed_probs = ewm_smoothed

    # 🎯 多尺度趋势一致性门控
    trend_3w = _compute_trend(smoothed_probs, window=3)
    trend_7w = _compute_trend(smoothed_probs, window=7)
    consensus_trend = np.minimum(trend_3w, trend_7w)
    trend_gate = np.clip(0.92 + consensus_trend * 3.0, 0.92, 1.0)
    smoothed_probs = smoothed_probs * trend_gate

    (p1_enter, p2_enter, p3_enter, p3_trend,
     exit_thresh, display_thresh, p1_sustain, p3_sustain) = \
        _compute_adaptive_thresholds(smoothed_probs, calibration_windows=30)

    (rolling_p1, rolling_p2, rolling_p3, rolling_p3t,
     rolling_exit, rolling_disp, rolling_p1s, rolling_p3s) = \
        _compute_rolling_thresholds(smoothed_probs, calibration_windows=30, recalibrate_every=30)

    active_alerts = _adaptive_alert(smoothed_probs, trend_signal=consensus_trend,
                                    p1_enter=rolling_p1, p2_enter=rolling_p2,
                                    p3_enter=rolling_p3, p3_trend=rolling_p3t,
                                    exit_thresh=rolling_exit,
                                    p1_sustain=rolling_p1s, p3_sustain=rolling_p3s,
                                    pip_values=pip_values)
    ai_alarms = extract_ai_alarm_episodes(time_axis_mins, active_alerts)

    # 🛡️ 最小报警持续时间过滤：丢弃短于1.5分钟的孤立报警（仅用于NSR评估）
    ai_alarms = [a for a in ai_alarms if a['end'] - a['start'] >= 1.5]

    # Plot
    if len(time_axis_mins) > 0:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(time_axis_mins, smoothed_probs, 'r-', linewidth=2.5, label='AI Risk (S-G Filtered)')
        ax.fill_between(time_axis_mins, 0, smoothed_probs, where=active_alerts,
                        color='salmon', alpha=0.3, label='Active Alert')
        ax.axhline(y=display_thresh, color='orange', linestyle=':', linewidth=2,
                   label=f'Threshold ({display_thresh:.2f})')
        ax.legend(loc='upper left', fontsize=11)
        ax.set_title(f"Icentia11k: Patient {patient_id} "
                     f"[P1={p1_enter:.2f}x{p1_sustain} P2={p2_enter:.2f} "
                     f"P3={p3_enter:.2f}x{p3_sustain} Exit={exit_thresh:.2f}]",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (Minutes)", fontsize=13)
        ax.set_ylabel("Risk Probability", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(time_axis_mins[0], total_duration_mins)
        plt.tight_layout()

        plot_filename = os.path.join(plot_out_dir, f"icentia11k_{patient_id}_curve.png")
        plt.savefig(plot_filename, dpi=200)
        plt.close(fig)

    # Time in alarm (all alarms are false by definition for NSR)
    total_alarm_mins = sum(max(0, a['end'] - a['start']) for a in ai_alarms)
    alarm_confidences = []
    for a in ai_alarms:
        vals = [smoothed_probs[ti] for ti, t in enumerate(time_axis_mins)
                if a['start'] <= t <= a['end'] and ti < len(smoothed_probs)]
        if vals:
            alarm_confidences.append(float(np.mean(vals)))

    return {
        'status': 'success',
        'patient_id': patient_id,
        'total_hours': total_duration_mins / 60.0,
        'n_alarms': len(ai_alarms),
        'n_windows': len(raw_probs),
        'total_alarm_mins': total_alarm_mins,
        'mean_prob': float(np.mean(raw_probs)),
        'max_prob': float(np.max(raw_probs)) if len(raw_probs) > 0 else 0.0,
        'threshold': float(display_thresh),
        'alarm_confidences': alarm_confidences,
    }


def scan_icentia11k(db_path, max_records=None):
    """Scan Icentia11k directory, return all valid records sorted by name."""
    records = []
    for f in sorted(os.listdir(db_path)):
        if f.endswith('.atr'):
            rec_name = f[:-4]
            rec_path = os.path.join(db_path, rec_name)
            if os.path.exists(rec_path + '.dat') and os.path.exists(rec_path + '.hea'):
                records.append(rec_path)

    if max_records:
        records = records[:max_records]

    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=128)
    parser.add_argument('--max-records', type=int, default=None,
                        help='Max records to evaluate (default: all 396)')
    parser.add_argument('--patients', type=str, default=None,
                        help='Comma-separated patient IDs to filter (e.g. p00000,p00001)')
    args = parser.parse_args()

    setup_logging(log_file="logs/evaluate_icentia11k.log")

    model_path = f"best_afib_model_s{args.seed}.pth"
    scaler_path = f"feature_scaler_s{args.seed}.pkl"

    db_path = r"D:\LoyaltyWorks\datasets\Icentia11k\Icentia11k"
    plot_out_dir = "evaluation_results_icentia11k"
    os.makedirs(plot_out_dir, exist_ok=True)

    records = scan_icentia11k(db_path, max_records=args.max_records)

    # Apply patient filter if specified
    if args.patients:
        target_pids = set(p.strip() for p in args.patients.split(','))
        records = [r for r in records if os.path.basename(r).split('_s')[0] in target_pids]

    logger.info(f"Found {len(records)} records to evaluate")

    args_list = [(rec, model_path, scaler_path, plot_out_dir) for rec in records]

    results = []
    with mp.Pool(2) as pool:  # reduced workers to avoid memory exhaustion
        for res in tqdm(pool.imap_unordered(evaluate_icentia11k_patient, args_list),
                        total=len(args_list)):
            results.append(res)

    # Aggregate
    total_hours = 0.0
    total_alarms = 0
    total_windows = 0
    total_alarm_mins = 0.0
    all_confidences = []
    success_count = 0

    for r in results:
        if r['status'] == 'success':
            success_count += 1
            total_hours += r['total_hours']
            total_alarms += r['n_alarms']
            total_windows += r['n_windows']
            total_alarm_mins += r.get('total_alarm_mins', 0)
            all_confidences.extend(r.get('alarm_confidences', []))

    total_mins = total_hours * 60.0
    far_per_24h = (total_alarms / total_hours * 24.0) if total_hours > 0 else 0.0
    alarm_burden = (total_alarm_mins / total_mins * 100) if total_mins > 0 else 0.0
    mean_conf = np.mean(all_confidences) if all_confidences else 0.0
    median_conf = np.median(all_confidences) if all_confidences else 0.0

    # Per-patient alarm burden distribution
    patient_burdens = []
    for r in results:
        if r['status'] == 'success' and r['total_hours'] > 0:
            burden = r.get('total_alarm_mins', 0) / (r['total_hours'] * 60) * 100
            patient_burdens.append(burden)
    patient_burdens = np.array(patient_burdens) if patient_burdens else np.array([0])

    print(f"\n{'='*72}")
    print(f"  Icentia11k NSR Evaluation v3 (seed={args.seed})")
    print(f"{'='*72}")
    print(f"  Patients: {success_count}  |  Duration: {total_hours:.1f}h  |  Windows: {total_windows}")
    print(f"")
    print(f"  [1] EVENT-LEVEL (all alarms are false by definition)")
    print(f"  ───────────────────────────────────────")
    print(f"  Total alarms:         {total_alarms}")
    print(f"  NSR FAR (event):      {far_per_24h:.1f} / 24h")
    print(f"")
    print(f"  [2] TIME-DOMAIN")
    print(f"  ───────────────────────────────────────")
    print(f"  Alarm burden:         {alarm_burden:.2f}% of total time")
    print(f"  Alarm burden dist:    med={np.median(patient_burdens):.2f}% "
          f"p25={np.percentile(patient_burdens, 25):.2f}% "
          f"p75={np.percentile(patient_burdens, 75):.2f}% "
          f"max={np.max(patient_burdens):.2f}%")
    print(f"")
    print(f"  [3] ALARM QUALITY")
    print(f"  ───────────────────────────────────────")
    print(f"  Confidence:           mean={mean_conf:.3f} median={median_conf:.3f}")
    print(f"")
    print(f"  [4] PER-PATIENT (top 20 by alarms)")
    print(f"  ───────────────────────────────────────")
    print(f"  {'ID':<20s} {'Hours':>7s} {'Alarms':>7s} {'Burden%':>8s} {'MeanP':>7s} {'MaxP':>7s} {'Thresh':>7s}")
    print(f"  {'-'*68}")
    for r in sorted(results, key=lambda x: x.get('n_alarms', 0), reverse=True)[:20]:
        if r['status'] == 'success':
            b = r.get('total_alarm_mins', 0) / (r['total_hours'] * 60) * 100 if r['total_hours'] > 0 else 0
            print(f"  {r['patient_id']:<20s} {r['total_hours']:>6.1f}h {r['n_alarms']:>7d} "
                  f"{b:>7.2f}% {r['mean_prob']:>7.4f} {r['max_prob']:>7.4f} {r['threshold']:>7.3f}")
    print(f"{'='*72}")

    # Save report
    report_path = os.path.join(plot_out_dir, "evaluation_report_icentia11k.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Icentia11k NSR Evaluation v3 (seed={args.seed})\n")
        f.write(f"{'='*60}\n")
        f.write(f"Patients: {success_count}  |  Hours: {total_hours:.1f}  |  Windows: {total_windows}\n")
        f.write(f"Event FAR: {far_per_24h:.1f} / 24h ({total_alarms} alarms)\n")
        f.write(f"Time burden: {alarm_burden:.2f}% (med={np.median(patient_burdens):.2f}%)\n")
        f.write(f"Confidence: mean={mean_conf:.3f} median={median_conf:.3f}\n")
    print(f"  Report saved to {report_path}")
