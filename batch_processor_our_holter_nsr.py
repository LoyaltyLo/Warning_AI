"""
Our Holter ECG Database — NSR (正常窦性心律) 训练数据处理器
============================================================

处理无 AFib (type=19) 的正常患者，生成全绿区 (label=0.0) 训练张量。
与 batch_processor_nsr2db.py 逻辑一致。

数据格式:
  Types CSV: 第一列 R波位置 (200Hz), 第二列 心搏类型
    5/204 = 正常, 56 = unknown, 41 = 室早, 32 = 房早
    (无 type=19，已预筛选)

输出: mixed_tensors_train/our_holter_nsr_<patient_id>.pt

用法:
  python batch_processor_our_holter_nsr.py
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.signal import welch, medfilt
from scipy.interpolate import interp1d
import antropy as ant
import warnings

warnings.filterwarnings('ignore')

# ─── 常量 ──────────────────────────────────────────────
OUTPUT_DIR = os.path.join(".", "mixed_tensors_train")
TIME_STEPS = 6
WINDOW_BEATS = 600           # ~10 min at 60bpm
STEP_BEATS = 600             # 稀疏采样 (~10 min spacing for healthy controls)
CONTINUITY_BEATS = 2100      # ~35 min 序列连续性校验

SIGNAL_FS = 200              # 原始数据采样率
TARGET_FS = 128              # 目标采样率（与训练数据一致）

# 异位心搏类型（用于 extract_features 掩码）
ECTOPIC_TYPES = {41, 32}     # 41=PVC(室早), 32=PAC(房早)

# ─── 路径 ──────────────────────────────────────────────
TYPES_DIR = r"D:\LoyaltyWorks\datasets\our_holter_ecg_database\ok\Types"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════
# 辅助函数（与 batch_processor_shdb.py 保持一致）
# SQA 阈值使用原始 5% — NSR 心率变异性低，不会被误拒
# ═══════════════════════════════════════════════════════

def _interpolate_nans(arr):
    if not np.any(np.isnan(arr)):
        return arr.copy()
    result = arr.copy()
    nans = np.isnan(result)
    valid_idx = np.where(~nans)[0]
    if len(valid_idx) == 0:
        return np.zeros_like(result)
    result[nans] = np.interp(np.where(nans)[0], valid_idx, result[valid_idx])
    return result


def extract_features(rr_window, aux_notes=None):
    """
    心搏域特征提取 v2.0 — 标准版 (5% SQA 阈值)
    与 batch_processor_shdb.py / batch_processor_nsr2db.py 完全一致。
    返回 14 维 HRV 特征。
    """
    total_beats = len(rr_window)
    if total_beats < 50:
        return None

    rr = rr_window.copy().astype(float)

    # 异位心搏掩码
    if aux_notes is not None:
        ectopic = {'V', 'A', 'a', 'J', 'S'}
        for i in range(len(aux_notes)):
            if str(aux_notes[i]).strip() in ectopic:
                rr[i] = np.nan

    raw_rr = rr[~np.isnan(rr)]
    if len(raw_rr) < 30:
        return None

    # SQA 门控 — NSR 用标准 5% 阈值
    if np.sum((raw_rr < 400) | (raw_rr > 3000)) / len(raw_rr) > 0.05:
        return None
    if len(raw_rr) > 1 and np.sum(np.abs(np.diff(raw_rr)) > 300) / len(raw_rr) > 0.05:
        return None

    rr_clean = _interpolate_nans(rr)
    rr_filtered = medfilt(rr_clean, kernel_size=3)

    # 早搏/二联律代偿中和
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
    rr_diff_clean = np.where(np.abs(rr_diff) < 50.0, 0.0, rr_diff)

    mean_rr = np.mean(rr_data_clean)
    std_rr = np.std(rr_data_clean)
    cv = std_rr / mean_rr if mean_rr > 0 else 0.0
    median_rr = np.median(rr_data_clean)
    mad = np.median(np.abs(rr_data_clean - median_rr))

    rmssd = np.sqrt(np.mean(rr_diff_clean ** 2)) if len(rr_diff_clean) > 0 else 0.0
    pnn50 = np.sum(np.abs(rr_diff_clean) > 50) / len(rr_diff_clean) if len(rr_diff_clean) > 0 else 0.0

    # Soft Noise Gate v2 — 阈值 25ms
    gate_weight = np.clip((rmssd - 25.0) / 25.0, 0.0, 1.0)

    try:
        samp_en_raw = ant.sample_entropy(rr_data_clean)
        samp_en = samp_en_raw * gate_weight
    except Exception:
        samp_en_raw = 0.0
        samp_en = 0.0

    try:
        dfa_raw = ant.detrended_fluctuation(rr_data_clean)
        dfa_alpha1 = dfa_raw * gate_weight
    except Exception:
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

    # 呼吸性窦性心律不齐周期性检测
    respiratory_periodicity = 0.0
    if len(rr_data_clean) > 30 and std_rr > 5.0:
        try:
            time_x_local = np.cumsum(rr_data_clean) / 1000.0
            time_x_local = time_x_local - time_x_local[0]
            if time_x_local[-1] > 30.0:
                f_interp = interp1d(time_x_local, rr_data_clean, kind='cubic',
                                    fill_value="extrapolate")
                fs_local = 4.0
                t_uniform = np.arange(0, time_x_local[-1], 1 / fs_local)
                rr_uniform = f_interp(t_uniform)
                f_psd, pxx_psd = welch(rr_uniform, fs_local,
                                       nperseg=min(128, len(rr_uniform)))
                resp_power = np.trapezoid(
                    pxx_psd[(f_psd >= 0.15) & (f_psd < 0.40)],
                    f_psd[(f_psd >= 0.15) & (f_psd < 0.40)])
                total_power = np.trapezoid(
                    pxx_psd[(f_psd >= 0.04) & (f_psd < 0.40)],
                    f_psd[(f_psd >= 0.04) & (f_psd < 0.40)])
                if total_power > 1e-6:
                    respiratory_periodicity = resp_power / total_power
        except Exception:
            respiratory_periodicity = 0.0

    resp_suppression = 1.0 - np.clip(respiratory_periodicity - 0.25, 0.0, 0.5)
    cv_suppressed = cv * resp_suppression
    rmssd_suppressed = rmssd * resp_suppression
    pnn50_suppressed = pnn50 * resp_suppression

    time_x = np.cumsum(rr_data_clean) / 1000.0
    time_x = time_x - time_x[0]

    if len(time_x) >= 2 and time_x[-1] > 0:
        f_interp = interp1d(time_x, rr_data_clean, kind='cubic',
                            fill_value="extrapolate")
        fs_interp = 4.0
        time_uniform = np.arange(0, time_x[-1], 1 / fs_interp)
        rr_uniform = f_interp(time_uniform)
        try:
            f, pxx = welch(rr_uniform, fs_interp, nperseg=256)
            lf_power = np.trapezoid(pxx[(f >= 0.04) & (f < 0.15)],
                                    f[(f >= 0.04) & (f < 0.15)])
            hf_power = np.trapezoid(pxx[(f >= 0.15) & (f < 0.40)],
                                    f[(f >= 0.15) & (f < 0.40)])
            lf_hf_ratio = (lf_power / (hf_power + 1e-6)) * gate_weight
        except Exception:
            lf_hf_ratio = 0.0
    else:
        lf_hf_ratio = 0.0

    # 二联律/三联律检测
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

    # RR分布双峰检测
    rr_sorted = np.sort(rr_data_clean)
    mid = len(rr_sorted) // 2
    lower_std = np.std(rr_sorted[:mid])
    upper_std = np.std(rr_sorted[mid:])
    denom = max(lower_std, upper_std)
    bimodality_ratio = min(lower_std, upper_std) / denom if denom > 1e-6 else 1.0

    # P1: 多尺度样本熵 (7 scales, vectorized) + Recurrence Plot (6 features, vectorized)
    mse_vals = [0.0] * 7
    rp_det, rp_lam, rp_entr, rp_rr, rp_tt, rp_maxline = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if len(rr_data_clean) >= 100:
        # --- 多尺度样本熵 (scales 2,3,5,7,10,15,20, vectorized) ---
        scales = [2, 3, 5, 7, 10, 15, 20]
        for si, scale in enumerate(scales):
            try:
                n_c = len(rr_data_clean) // scale
                if n_c >= 30:
                    coarse = np.array([np.mean(rr_data_clean[i*scale:(i+1)*scale]) for i in range(n_c)])
                    sd = np.std(coarse)
                    if sd > 1e-8:
                        r = 0.2 * sd; m = 2; N = n_c - m
                        if N > 1:
                            emb = np.lib.stride_tricks.sliding_window_view(coarse, m)
                            from scipy.spatial.distance import cdist
                            dists = cdist(emb, emb, metric='chebyshev')
                            cnt = int(np.sum(np.triu(dists < r, k=1)))
                            den = max(1, N * (N - 1) // 2)
                            mse_vals[si] = np.clip(-np.log(max(1, cnt) / den), 0, 10)
            except Exception:
                pass
        # --- Recurrence Plot (embedded phase space, vectorized) ---
        try:
            rr_z = (rr_data_clean - np.mean(rr_data_clean)) / (np.std(rr_data_clean) + 1e-8)
            n_pts = len(rr_z) - 1
            if n_pts > 20:
                phase = np.column_stack([rr_z[:n_pts], rr_z[1:n_pts+1]])
                th = 0.3 * np.std(rr_z)
                max_diag = 40
                diag_lens, vert_lens = [], []
                n_recur = 0
                for k in range(1, max_diag + 1):
                    dists_k = np.sqrt((phase[:n_pts-k, 0] - phase[k:n_pts, 0])**2 +
                                     (phase[:n_pts-k, 1] - phase[k:n_pts, 1])**2)
                    is_rp = dists_k < th
                    n_recur += int(np.sum(is_rp))
                    changes = np.diff(np.concatenate([[0], is_rp.astype(np.int8), [0]]))
                    starts = np.where(changes == 1)[0]
                    ends = np.where(changes == -1)[0]
                    for s, e in zip(starts, ends):
                        if e - s >= 2:
                            diag_lens.append(int(e - s))
                n_recur *= 2
                total = n_pts * max_diag * 2
                if n_recur > 0:
                    rp_rr = n_recur / total
                    if diag_lens:
                        rp_det = sum(diag_lens) / n_recur
                        rp_maxline = max(diag_lens)
                        rp_tt = np.mean(diag_lens)
                        cnts = np.bincount([d for d in diag_lens if d >= 2])
                        if len(cnts) > 0 and cnts.sum() > 0:
                            p = cnts / cnts.sum()
                            rp_entr = -sum(pp * np.log(max(pp, 1e-10)) for pp in p if pp > 0)
                for j in range(n_pts):
                    i_start = max(0, j - max_diag)
                    i_end = min(n_pts, j + max_diag)
                    if i_end <= i_start:
                        continue
                    dists_v = np.sqrt((phase[i_start:i_end, 0] - phase[j, 0])**2 +
                                     (phase[i_start:i_end, 1] - phase[j, 1])**2)
                    is_rp_v = dists_v < th
                    changes = np.diff(np.concatenate([[0], is_rp_v.astype(np.int8), [0]]))
                    starts = np.where(changes == 1)[0]
                    ends = np.where(changes == -1)[0]
                    for s, e in zip(starts, ends):
                        if e - s >= 2:
                            vert_lens.append(int(e - s))
                if n_recur > 0 and vert_lens:
                    rp_lam = sum(vert_lens) / n_recur
        except Exception:
            pass

    return [cv_suppressed, mad, rmssd_suppressed, pnn50_suppressed,
            samp_en, dfa_alpha1, pip, sd1, poincare_ratio, lf_hf_ratio,
            sd2_normalized,
            mse_vals[0], mse_vals[1], mse_vals[2], mse_vals[3],
            mse_vals[4], mse_vals[5], mse_vals[6],
            rp_det, rp_lam, rp_entr, rp_rr, rp_tt, rp_maxline,
            pip_raw, dfa_raw, bigeminy_corr, bimodality_ratio]


# ═══════════════════════════════════════════════════════
# 数据读取
# ═══════════════════════════════════════════════════════

def read_our_holter_types(types_path):
    """读取 Types CSV，返回 (r_positions, beat_types)"""
    positions = []
    types = []
    with open(types_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    positions.append(int(parts[0].strip()))
                    types.append(int(parts[1].strip()))
                except ValueError:
                    continue
    return np.array(positions, dtype=int), np.array(types, dtype=int)


# ═══════════════════════════════════════════════════════
# 单患者处理
# ═══════════════════════════════════════════════════════

def process_single_nsr(patient_id):
    """处理单个 NSR 患者，生成全绿区训练张量"""
    save_path = os.path.join(OUTPUT_DIR, f"our_holter_nsr_{patient_id}.pt")
    if os.path.exists(save_path):
        return f"Skipped (exists): {patient_id}"

    types_path = os.path.join(TYPES_DIR, f"{patient_id}.csv")
    if not os.path.exists(types_path):
        return f"Failed (no types): {patient_id}"

    # 1. 读取 Types
    r_positions, beat_types = read_our_holter_types(types_path)
    if len(r_positions) < WINDOW_BEATS + 100:
        return f"Skipped (too few beats: {len(r_positions)}): {patient_id}"

    # 2. 重采样: 200Hz → 128Hz
    r_positions_128 = np.round(r_positions * TARGET_FS / SIGNAL_FS).astype(int)

    # 3. 计算 RR 间期 (ms)
    rr_intervals = np.zeros(len(r_positions_128))
    if len(r_positions_128) > 1:
        rr_intervals[1:] = np.diff(r_positions_128) / TARGET_FS * 1000.0
        rr_intervals[0] = rr_intervals[1]

    # 4. 构建 aux_notes（异位心搏掩码用）
    aux_notes = []
    for bt in beat_types:
        if bt in ECTOPIC_TYPES:
            aux_notes.append('V')
        else:
            aux_notes.append('')
    aux_notes = np.array(aux_notes, dtype=object)

    total_beats = len(rr_intervals)

    # 5. 稀疏滑动窗口 — 所有标签 = 0.0
    all_slices = []
    start_idx = 0

    while start_idx + WINDOW_BEATS <= total_beats:
        end_idx = start_idx + WINDOW_BEATS
        rr_window = rr_intervals[start_idx:end_idx]
        notes_window = aux_notes[start_idx:end_idx]
        feats_hrv = extract_features(rr_window, notes_window)
        if feats_hrv is not None:
            all_slices.append({
                'start_idx': start_idx,
                'feats': feats_hrv,
                'label': 0.0
            })
        start_idx += STEP_BEATS

    # 6. 组装 6 步时间序列
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
        return f"Skipped (no valid sequences): {patient_id}"

    # 7. 保存
    tensor_data = {
        'record': f"our_holter_nsr_{patient_id}",
        'X': torch.tensor(X, dtype=torch.float32),
        'Y': torch.tensor(Y, dtype=torch.float32)
    }
    torch.save(tensor_data, save_path)

    return f"OK: {patient_id} | seqs={len(X)} | slices={len(all_slices)}"


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    # 可选参数：限制处理数量（按字母序取前 N 个）
    limit = -1

    types_files = [f.replace('.csv', '') for f in os.listdir(TYPES_DIR)
                   if f.endswith('.csv')]
    all_ids = sorted(types_files)
    print(f"\n{'='*60}")
    print(f"  Our Holter NSR — Training Tensor Generator")
    print(f"  Total patients in Types dir: {len(all_ids)}")
    print(f"{'='*60}")

    # 预筛选：排除有 type=19 (AFib) 的患者
    nsr_patients = []
    skipped_afib = 0
    AFIB_BEAT_TYPE = 19
    for pid in tqdm(all_ids, desc="Filtering NSR (exclude AFib)"):
        types_path = os.path.join(TYPES_DIR, f"{pid}.csv")
        try:
            _, beat_types = read_our_holter_types(types_path)
            if AFIB_BEAT_TYPE not in beat_types:
                nsr_patients.append(pid)
            else:
                skipped_afib += 1
        except Exception:
            skipped_afib += 1

    print(f"  NSR patients (no type 19): {len(nsr_patients)}")
    print(f"  Excluded (has AFib): {skipped_afib}")

    if limit > 0:
        nsr_patients = nsr_patients[:limit]
        print(f"  Limited to {limit} patients (alphabetical order)")

    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Sampling: STEP_BEATS={STEP_BEATS} (~10 min spacing)")
    print(f"  SQA threshold: 5% (standard NSR)")
    print(f"{'='*60}\n")

    # 处理
    results = []
    for pid in tqdm(nsr_patients, desc="Processing NSR patients"):
        result = process_single_nsr(pid)
        results.append(result)

    # 汇总
    success = sum(1 for r in results if r.startswith("OK"))
    skipped_exists = sum(1 for r in results if "exists" in r)
    skipped_other = sum(1 for r in results
                        if r.startswith("Skipped") and "exists" not in r)
    failed = sum(1 for r in results if r.startswith("Failed"))

    total_seqs = 0
    total_slices = 0
    for r in results:
        if r.startswith("OK"):
            parts = r.split("|")
            try:
                total_seqs += int(parts[1].strip().replace("seqs=", ""))
                total_slices += int(parts[2].strip().replace("slices=", ""))
            except (IndexError, ValueError):
                pass

    print(f"\n{'='*60}")
    print(f"  NSR PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Success:          {success}")
    print(f"  Skipped (exists): {skipped_exists}")
    print(f"  Skipped (other):  {skipped_other}")
    print(f"  Failed:           {failed}")
    print(f"  ---")
    print(f"  Total sequences:  {total_seqs}")
    print(f"  Total slices:     {total_slices} (all label=0.0)")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"{'='*60}")
