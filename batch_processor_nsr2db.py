import os
import wfdb
import torch
import numpy as np
from tqdm import tqdm
from scipy.signal import welch, medfilt
from scipy.interpolate import interp1d
import antropy as ant
import warnings


warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(".", "mixed_tensors_train")
TIME_STEPS = 6
WINDOW_BEATS = 600      # ~10 min at 60bpm
STEP_BEATS = 600        # sparse sampling for healthy controls
CONTINUITY_BEATS = 2100 # ~35 min continuity check
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _interpolate_nans(arr):
    """用线性插值替换 NaN，边缘 NaN 用最近有效值填充（等价于 interpolate().bfill().ffill()）"""
    if not np.any(np.isnan(arr)):
        return arr.copy()
    result = arr.copy()
    nans = np.isnan(result)
    valid_idx = np.where(~nans)[0]
    if len(valid_idx) == 0:
        return np.zeros_like(result)
    result[nans] = np.interp(np.where(nans)[0], valid_idx, result[valid_idx])
    return result


def resample_ecg_dataset(ann_samples, original_fs=128, target_fs=128):
    if original_fs == target_fs: return ann_samples, target_fs
    return np.round(ann_samples * (target_fs / original_fs)).astype(int), target_fs


def extract_features(rr_window, aux_notes=None):
    """
    心搏域特征提取 v2.0：增强NSR判别力

    改进点：
    1. 容差死区 40→50ms（过滤呼吸性窦性心律不齐）
    2. Soft Noise Gate 阈值 10→20ms（更严格压制低变异时的非线性特征）
    3. 新增呼吸性窦性心律不齐周期性检测
    4. 代偿中和阈值收紧 0.92/1.08 → 0.94/1.06
    """
    total_beats = len(rr_window)
    if total_beats < 50: return None

    rr = rr_window.copy().astype(float)

    # 异位心搏掩码
    if aux_notes is not None:
        ectopic = {'V', 'A', 'a', 'J', 'S'}
        for i in range(len(aux_notes)):
            if str(aux_notes[i]).strip() in ectopic:
                rr[i] = np.nan

    raw_rr = rr[~np.isnan(rr)]
    if len(raw_rr) < 30: return None

    # 🛡️ 1. SQA 门控
    if np.sum((raw_rr < 400) | (raw_rr > 3000)) / len(raw_rr) > 0.05:
        return None
    if len(raw_rr) > 1 and np.sum(np.abs(np.diff(raw_rr)) > 300) / len(raw_rr) > 0.05:
        return None

    rr_clean = _interpolate_nans(rr)

    # 🛡️ 2. 中值滤波
    rr_filtered = medfilt(rr_clean, kernel_size=3)

    # 🌟 3. 早搏/二联律代偿中和 (0.94/1.06)
    rr_data_clean = rr_filtered.copy()
    local_mean = np.median(rr_data_clean)
    i = 0
    while i < len(rr_data_clean) - 1:
        rr1, rr2 = rr_data_clean[i], rr_data_clean[i + 1]
        if (rr1 < 0.95 * local_mean and rr2 > 1.05 * local_mean) or \
                (rr1 > 1.05 * local_mean and rr2 < 0.95 * local_mean):
            avg_rr = (rr1 + rr2) / 2.0
            rr_data_clean[i] = avg_rr
            rr_data_clean[i + 1] = avg_rr
            i += 2
        else:
            i += 1

    rr_diff = np.diff(rr_data_clean)

    # 🚀 4. 容差死区 (50.0ms)
    rr_diff_clean = np.where(np.abs(rr_diff) < 50.0, 0.0, rr_diff)

    mean_rr = np.mean(rr_data_clean)
    std_rr = np.std(rr_data_clean)
    cv = std_rr / mean_rr if mean_rr > 0 else 0.0
    median_rr = np.median(rr_data_clean)
    mad = np.median(np.abs(rr_data_clean - median_rr))

    rmssd = np.sqrt(np.mean(rr_diff_clean ** 2)) if len(rr_diff_clean) > 0 else 0.0
    pnn50 = np.sum(np.abs(rr_diff_clean) > 50) / len(rr_diff_clean) if len(rr_diff_clean) > 0 else 0.0

    # 🛡️ 5. Soft Noise Gate v2 — 提高下限从10→25ms
    gate_weight = np.clip((rmssd - 25.0) / 25.0, 0.0, 1.0)

    try:
        samp_en_raw = ant.sample_entropy(rr_data_clean)
        samp_en = samp_en_raw * gate_weight
    except:
        samp_en_raw = 0.0
        samp_en = 0.0

    try:
        dfa_raw = ant.detrended_fluctuation(rr_data_clean)
        dfa_alpha1 = dfa_raw * gate_weight
    except:
        dfa_raw = 0.0
        dfa_alpha1 = 0.0

    if len(rr_diff_clean) > 1:
        sign_changes = (rr_diff_clean[:-1] * rr_diff_clean[1:]) < 0
        run_len = 0
        persistent_count = 0
        for sc in sign_changes:
            if sc:
                run_len += 1
            else:
                if run_len >= 3:
                    persistent_count += run_len
                run_len = 0
        if run_len >= 3:
            persistent_count += run_len
        pip_raw = persistent_count / len(rr_diff_clean)
        pip = pip_raw * gate_weight
        sd1 = np.sqrt(0.5 * np.var(rr_diff_clean)) * gate_weight
        sd2 = np.sqrt(0.5 * np.var(rr_data_clean[1:] + rr_data_clean[:-1]))
        poincare_ratio = (sd1 / sd2 if sd2 > 15.0 else 0.0) * gate_weight
        sd2_normalized = sd2 / mean_rr if mean_rr > 0 else 0.0
    else:
        pip_raw, pip, sd1, sd2, poincare_ratio, sd2_normalized = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 🌟 6. 呼吸性窦性心律不齐周期性检测
    respiratory_periodicity = 0.0
    if len(rr_data_clean) > 30 and std_rr > 5.0:
        try:
            time_x_local = np.cumsum(rr_data_clean) / 1000.0
            time_x_local = time_x_local - time_x_local[0]
            if time_x_local[-1] > 30.0:
                f_interp = interp1d(time_x_local, rr_data_clean, kind='cubic', fill_value="extrapolate")
                fs_local = 4.0
                t_uniform = np.arange(0, time_x_local[-1], 1 / fs_local)
                rr_uniform = f_interp(t_uniform)
                f_psd, pxx_psd = welch(rr_uniform, fs_local, nperseg=min(128, len(rr_uniform)))
                resp_power = np.trapezoid(pxx_psd[(f_psd >= 0.15) & (f_psd < 0.40)],
                                          f_psd[(f_psd >= 0.15) & (f_psd < 0.40)])
                total_power = np.trapezoid(pxx_psd[(f_psd >= 0.04) & (f_psd < 0.40)],
                                           f_psd[(f_psd >= 0.04) & (f_psd < 0.40)])
                if total_power > 1e-6:
                    respiratory_periodicity = resp_power / total_power
        except:
            respiratory_periodicity = 0.0

    # 呼吸周期性压制
    resp_suppression = 1.0 - np.clip(respiratory_periodicity - 0.25, 0.0, 0.5)
    cv_suppressed = cv * resp_suppression
    rmssd_suppressed = rmssd * resp_suppression
    pnn50_suppressed = pnn50 * resp_suppression

    time_x = np.cumsum(rr_data_clean) / 1000.0
    time_x = time_x - time_x[0]

    if len(time_x) >= 2 and time_x[-1] > 0:
        f_interp = interp1d(time_x, rr_data_clean, kind='cubic', fill_value="extrapolate")
        fs_interp = 4.0
        time_uniform = np.arange(0, time_x[-1], 1 / fs_interp)
        rr_uniform = f_interp(time_uniform)
        try:
            f, pxx = welch(rr_uniform, fs_interp, nperseg=256)
            lf_power = np.trapezoid(pxx[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
            hf_power = np.trapezoid(pxx[(f >= 0.15) & (f < 0.40)], f[(f >= 0.15) & (f < 0.40)])
            lf_hf_ratio = (lf_power / (hf_power + 1e-6)) * gate_weight
        except:
            lf_hf_ratio = 0.0
    else:
        lf_hf_ratio = 0.0

    # P1a: 二联律/三联律检测 — lag-2 + lag-3自相关 (AFib=无结构, 早搏=有结构)
    rr_norm = (rr_data_clean - np.mean(rr_data_clean)) / (np.std(rr_data_clean) + 1e-6)
    if len(rr_norm) > 6:
        corr_lag2 = np.corrcoef(rr_norm[2:], rr_norm[:-2])[0, 1]
        corr_lag2 = np.clip(corr_lag2, -1.0, 1.0) if not np.isnan(corr_lag2) else 0.0
        corr_lag3 = np.corrcoef(rr_norm[3:], rr_norm[:-3])[0, 1]
        corr_lag3 = np.clip(corr_lag3, -1.0, 1.0) if not np.isnan(corr_lag3) else 0.0
        bigeminy_corr = max(corr_lag2, corr_lag3)
    elif len(rr_norm) > 4:
        bigeminy_corr = np.corrcoef(rr_norm[2:], rr_norm[:-2])[0, 1]
        bigeminy_corr = np.clip(bigeminy_corr, -1.0, 1.0) if not np.isnan(bigeminy_corr) else 0.0
    else:
        bigeminy_corr = 0.0

    # P1b: RR分布双峰检测 (AFib=均匀≈1, 早搏=双峰<<1)
    rr_sorted = np.sort(rr_data_clean)
    mid = len(rr_sorted) // 2
    lower_std = np.std(rr_sorted[:mid])
    upper_std = np.std(rr_sorted[mid:])
    denom = max(lower_std, upper_std)
    bimodality_ratio = min(lower_std, upper_std) / denom if denom > 1e-6 else 1.0

    return [cv_suppressed, mad, rmssd_suppressed, pnn50_suppressed,
            samp_en, dfa_alpha1, pip, sd1, poincare_ratio, lf_hf_ratio,
            sd2_normalized, pip_raw, dfa_raw, bigeminy_corr, bimodality_ratio]


def process_healthy_record(record_name):
    base_name = os.path.basename(record_name)
    save_path = os.path.join(OUTPUT_DIR, f"healthy_mitbih_nsr_{base_name}.pt")
    if os.path.exists(save_path):
        return f"Already processed: {base_name}"

    try:
        ann = wfdb.rdann(record_name, 'atr')
        fs = getattr(ann, 'fs', 128)
        samples = ann.sample
    except Exception:
        return f"Failed: {base_name}"

    if fs != 128:
        samples, fs = resample_ecg_dataset(samples, original_fs=fs, target_fs=128)

    rr_intervals = np.zeros(len(samples))
    if len(samples) > 1:
        rr_intervals[1:] = np.diff(samples) / fs * 1000.0
        rr_intervals[0] = rr_intervals[1]

    total_beats = len(rr_intervals)
    all_slices = []
    start_idx = 0

    while start_idx + WINDOW_BEATS <= total_beats:
        end_idx = start_idx + WINDOW_BEATS
        rr_window = rr_intervals[start_idx:end_idx]
        feats_hrv = extract_features(rr_window)
        if feats_hrv is not None:
            feats = feats_hrv  # 14D HRV features
            all_slices.append({'start_idx': start_idx, 'feats': feats, 'label': 0.0})
        start_idx += STEP_BEATS

    X, Y = [], []
    if len(all_slices) >= TIME_STEPS:
        for i in range(len(all_slices) - TIME_STEPS + 1):
            seq = all_slices[i: i + TIME_STEPS]
            valid_seq = True
            for j in range(1, len(seq)):
                if seq[j]['start_idx'] - seq[j - 1]['start_idx'] > CONTINUITY_BEATS:
                    valid_seq = False
                    break
            if valid_seq:
                seq_feats = []
                for j in range(len(seq)):
                    curr_feats = seq[j]['feats'].copy()
                    delta_en = 0.0 if j == 0 else curr_feats[4] - seq[j - 1]['feats'][4]
                    curr_feats.append(delta_en)  # 15D: 14 HRV + delta_entropy
                    seq_feats.append(curr_feats)
                X.append(seq_feats)
                Y.append([s['label'] for s in seq])

    if len(X) == 0:
        return f"Skipped: {base_name}"

    tensor_data = {
        'record': f"healthy_mitbih_nsr_{base_name}",
        'X': torch.tensor(X, dtype=torch.float32),
        'Y': torch.tensor(Y, dtype=torch.float32)
    }
    torch.save(tensor_data, save_path)
    return f"Success: {base_name}"

def process_nsr2db_record(record_name):
    """Process a single NSR2DB record — RR-only, no waveform."""
    base_name = os.path.basename(record_name)
    save_path = os.path.join(OUTPUT_DIR, f"healthy_nsr2db_{base_name}.pt")
    if os.path.exists(save_path):
        return f"Already processed: {base_name}"

    try:
        ann = wfdb.rdann(record_name, 'ecg')
        fs = getattr(ann, 'fs', 128)
        samples = ann.sample
    except Exception:
        return f"Failed: {base_name}"

    if fs != 128:
        samples, fs = resample_ecg_dataset(samples, original_fs=fs, target_fs=128)

    rr_intervals = np.zeros(len(samples))
    if len(samples) > 1:
        rr_intervals[1:] = np.diff(samples) / fs * 1000.0
        rr_intervals[0] = rr_intervals[1]

    total_beats = len(rr_intervals)
    all_slices = []
    start_idx = 0

    while start_idx + WINDOW_BEATS <= total_beats:
        end_idx = start_idx + WINDOW_BEATS
        rr_window = rr_intervals[start_idx:end_idx]
        feats_hrv = extract_features(rr_window)
        if feats_hrv is not None:
            # NSR2DB has no waveform → HRV features only
            feats = feats_hrv
            all_slices.append({'start_idx': start_idx, 'feats': feats, 'label': 0.0})
        start_idx += STEP_BEATS

    X, Y = [], []
    if len(all_slices) >= TIME_STEPS:
        for i in range(len(all_slices) - TIME_STEPS + 1):
            seq = all_slices[i: i + TIME_STEPS]
            valid_seq = True
            for j in range(1, len(seq)):
                if seq[j]['start_idx'] - seq[j - 1]['start_idx'] > CONTINUITY_BEATS:
                    valid_seq = False
                    break
            if valid_seq:
                seq_feats = []
                for j in range(len(seq)):
                    curr_feats = seq[j]['feats'].copy()
                    delta_en = 0.0 if j == 0 else curr_feats[4] - seq[j - 1]['feats'][4]
                    curr_feats.append(delta_en)
                    seq_feats.append(curr_feats)
                X.append(seq_feats)
                Y.append([s['label'] for s in seq])

    if len(X) == 0:
        return f"Skipped: {base_name}"

    tensor_data = {
        'record': f"healthy_nsr2db_{base_name}",
        'X': torch.tensor(X, dtype=torch.float32),
        'Y': torch.tensor(Y, dtype=torch.float32)
    }
    torch.save(tensor_data, save_path)
    return f"Success: {base_name}"


if __name__ == '__main__':
    # DB1: MIT-BIH NSR (2-lead ECG waveforms)
    db_mitbih = r"D:\LoyaltyWorks\datasets\mit-bih-normal-sinus-rhythm-database-1.0.0\mit-bih-normal-sinus-rhythm-database-1.0.0"
    mitbih_records = sorted(set([
        os.path.join(db_mitbih, f.split(".")[0])
        for f in os.listdir(db_mitbih) if f.endswith(".atr")
    ]))
    print(f"Found {len(mitbih_records)} MIT-BIH NSR records")
    for rec in tqdm(mitbih_records, total=len(mitbih_records), desc="MIT-BIH NSR"):
        result = process_healthy_record(rec)
        if "Failed" in result or "Skipped" in result:
            tqdm.write(f"  {result}")

    # DB2: NSR2DB (RR-only, no waveforms)
    db_nsr2 = r"D:\LoyaltyWorks\datasets\normal-sinus-rhythm-rr-interval-database-1.0.0\normal-sinus-rhythm-rr-interval-database-1.0.0"
    nsr2_records = sorted(set([
        os.path.join(db_nsr2, f.split(".")[0])
        for f in os.listdir(db_nsr2) if f.endswith(".ecg")
    ]))
    print(f"Found {len(nsr2_records)} NSR2DB records")
    for rec in tqdm(nsr2_records, total=len(nsr2_records), desc="NSR2DB"):
        result = process_nsr2db_record(rec)
        if "Failed" in result or "Skipped" in result:
            tqdm.write(f"  {result}")
