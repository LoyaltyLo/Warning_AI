"""
Our Holter ECG Database 训练数据处理器
=========================================

数据格式:
  Signal CSV (D:\LoyaltyWorks\datasets\our_holter_ecg_database\ok\Signal):
    3导联 ECG, 200Hz, 24h — 原始ADC值
    转换公式: (signal - 2048) / 4096
  Types CSV  (D:\LoyaltyWorks\datasets\our_holter_ecg_database\ok\Types):
    第一列: R波位置 (200Hz采样率下)
    第二列: 心搏类型
      5/204 = 正常
      19    = 房颤
      41    = 室早 (PVC)
      32    = 房早 (PAC)
      56    = unknown

输出: mixed_tensors_train/our_holter_<patient_id>.pt

用法:
  source .venv/Scripts/activate
  python batch_processor_our_holter.py
"""

import os
import torch
import numpy as np
import pandas as pd
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
STEP_BEATS = 120             # ~2 min at 60bpm (红区密集步进)
GREEN_SPACING_BEATS = 600    # ~10 min 绿区最小间距
AFIB_PROXIMITY_BEATS = 3600  # ~60 min 高危区
SAFE_DISTANCE_BEATS = 7200   # ~120 min 安全区
CONTINUITY_BEATS = 2100      # ~35 min 序列连续性校验

SIGNAL_FS = 200              # 原始数据采样率
TARGET_FS = 128              # 目标采样率（与训练数据一致）

# 心搏类型映射
AFIB_BEAT_TYPE = 19
ECTOPIC_TYPES = {41, 32}     # 41=PVC(室早), 32=PAC(房早)
NORMAL_TYPES = {5, 204}      # 正常心搏

# ─── 路径 ──────────────────────────────────────────────
SIGNAL_DIR = r"D:\LoyaltyWorks\datasets\our_holter_ecg_database\ok\Signal"
TYPES_DIR  = r"D:\LoyaltyWorks\datasets\our_holter_ecg_database\ok\Types"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════
# 辅助函数（与 batch_processor_shdb.py 保持一致）
# ═══════════════════════════════════════════════════════

def _interpolate_nans(arr):
    """用线性插值替换 NaN，边缘 NaN 用最近有效值填充"""
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
    心搏域特征提取 v2.0：增强NSR判别力
    （与 batch_processor_shdb.py / batch_evaluate_cdss.py 完全一致）

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

    # 🛡️ 1. SQA 门控
    # NOTE: diff spike 阈值从 5% 放宽到 20%，因为 AFib 窗口的 RR
    # 间期天然高度不规则（>10% diffs >300ms 是房颤特征），5% 会
    # 错误拒绝几乎所有 AFib 窗口。20% 仍能过滤真正的伪影窗口。
    if np.sum((raw_rr < 400) | (raw_rr > 3000)) / len(raw_rr) > 0.05:
        return None
    if len(raw_rr) > 1 and np.sum(np.abs(np.diff(raw_rr)) > 300) / len(raw_rr) > 0.20:
        return None

    rr_clean = _interpolate_nans(rr)

    # 🛡️ 2. 中值滤波
    rr_filtered = medfilt(rr_clean, kernel_size=3)

    # 🌟 3. 早搏/二联律代偿中和
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

    # 🛡️ 5. Soft Noise Gate v2 — 阈值 25ms
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

    # 🌟 6. 呼吸性窦性心律不齐周期性检测
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

    # 呼吸周期性压制
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

    # P1a: 二联律/三联律检测
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

    # P1b: RR分布双峰检测
    rr_sorted = np.sort(rr_data_clean)
    mid = len(rr_sorted) // 2
    lower_std = np.std(rr_sorted[:mid])
    upper_std = np.std(rr_sorted[mid:])
    denom = max(lower_std, upper_std)
    bimodality_ratio = min(lower_std, upper_std) / denom if denom > 1e-6 else 1.0

    return [cv_suppressed, mad, rmssd_suppressed, pnn50_suppressed,
            samp_en, dfa_alpha1, pip, sd1, poincare_ratio, lf_hf_ratio,
            sd2_normalized, pip_raw, dfa_raw, bigeminy_corr, bimodality_ratio]


# ═══════════════════════════════════════════════════════
# 数据读取
# ═══════════════════════════════════════════════════════

def read_our_holter_types(types_path):
    """
    读取 Types CSV 文件。
    返回: (r_positions: np.ndarray, beat_types: np.ndarray)
      r_positions: R波样本位置 (200Hz采样率下)
      beat_types: 心搏类型 int
    """
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
# AFib 发作检测 & 窗口标签
# ═══════════════════════════════════════════════════════

def get_all_afib_episodes(aux_notes):
    """从心搏标注数组中提取房颤发作区间（心搏索引）"""
    episodes = []
    in_afib = False
    start_idx = None
    for i in range(len(aux_notes)):
        note_str = str(aux_notes[i]).upper()
        if '(AFIB' in note_str and not in_afib:
            in_afib = True
            start_idx = i
        elif '(' in note_str and '(AFIB' not in note_str and in_afib:
            in_afib = False
            episodes.append({'start_idx': start_idx, 'end_idx': i})
    if in_afib:
        episodes.append({'start_idx': start_idx, 'end_idx': len(aux_notes) - 1})
    return episodes


def find_afib_episodes_from_beats(beat_types, min_afib_beats=10, merge_gap=30):
    """
    从逐搏类型中识别房颤发作区间（与 evaluate_syaf.py 一致）。

    Args:
        beat_types: 心搏类型数组
        min_afib_beats: 最少连续AFib心搏数
        merge_gap: 两个AFib片段之间的最大间隔（心搏数）

    Returns:
        [(start_beat_idx, end_beat_idx), ...]  发作区间（心搏索引，左闭右闭）
    """
    afib_mask = (beat_types == AFIB_BEAT_TYPE)
    episodes = []
    in_episode = False
    start = None

    for i, is_afib in enumerate(afib_mask):
        if is_afib and not in_episode:
            in_episode = True
            start = i
        elif not is_afib and in_episode:
            in_episode = False
            if i - start >= min_afib_beats:
                episodes.append((start, i - 1))

    if in_episode:
        end = len(afib_mask) - 1
        if end - start + 1 >= min_afib_beats:
            episodes.append((start, end))

    # 合并间隔小于 merge_gap 的片段
    if merge_gap > 0 and len(episodes) > 1:
        merged = [episodes[0]]
        for ep in episodes[1:]:
            if ep[0] - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], ep[1])
            else:
                merged.append(ep)
        episodes = merged

    return episodes


def check_overlap(w_start_idx, w_end_idx, episodes):
    """检查窗口 [w_start_idx, w_end_idx) 是否与任一房颤发作重叠"""
    for ep in episodes:
        if max(w_start_idx, ep['start_idx']) < min(w_end_idx, ep['end_idx']):
            return True
    return False


def beats_to_next_afib(w_end_idx, episodes):
    """从窗口结束位置到下一个房颤发作的心搏数"""
    future_eps = [ep for ep in episodes if ep['start_idx'] >= w_end_idx]
    return future_eps[0]['start_idx'] - w_end_idx if future_eps else float('inf')


def beats_since_last_afib(w_start_idx, episodes):
    """从上次房颤结束到窗口起始位置的心搏数"""
    past_eps = [ep for ep in episodes if ep['end_idx'] <= w_start_idx]
    return w_start_idx - past_eps[-1]['end_idx'] if past_eps else float('inf')


# ═══════════════════════════════════════════════════════
# 单患者处理
# ═══════════════════════════════════════════════════════

def process_single_patient(patient_id):
    """处理单个患者，生成训练张量并保存"""
    save_path = os.path.join(OUTPUT_DIR, f"our_holter_{patient_id}.pt")
    if os.path.exists(save_path):
        return f"Skipped (exists): {patient_id}"

    types_path = os.path.join(TYPES_DIR, f"{patient_id}.csv")
    if not os.path.exists(types_path):
        return f"Failed (no types): {patient_id}"

    # 1. 读取 Types → R波位置 + 心搏类型
    r_positions, beat_types = read_our_holter_types(types_path)
    if len(r_positions) < WINDOW_BEATS + 100:
        return f"Skipped (too few beats: {len(r_positions)}): {patient_id}"

    # 2. 重采样 R 波位置: 200Hz → 128Hz（与训练数据一致）
    r_positions_128 = np.round(r_positions * TARGET_FS / SIGNAL_FS).astype(int)

    # 3. 计算 RR 间期 (ms) — 基于128Hz重采样后的位置
    rr_intervals = np.zeros(len(r_positions_128))
    if len(r_positions_128) > 1:
        rr_intervals[1:] = np.diff(r_positions_128) / TARGET_FS * 1000.0
        rr_intervals[0] = rr_intervals[1]

    # 4. 构建 aux_notes（用于 extract_features 异位心搏掩码和 AFib 检测）
    #    41(PVC), 32(PAC) → 'V' (ectopic mask)
    #    19(AFib) → '(AFIB'
    #    5, 204(Normal), 56(unknown), 其他 → ''
    aux_notes = []
    for bt in beat_types:
        if bt in ECTOPIC_TYPES:
            aux_notes.append('V')
        elif bt == AFIB_BEAT_TYPE:
            aux_notes.append('(AFIB')
        else:
            aux_notes.append('')
    aux_notes = np.array(aux_notes, dtype=object)

    # 5. 识别 AFib 发作区间（心搏索引）
    afib_episodes_raw = find_afib_episodes_from_beats(beat_types)
    if not afib_episodes_raw:
        return f"Skipped (no AFib episodes): {patient_id}"

    # 转换为与 batch_processor_shdb.py 兼容的 dict 格式
    afib_episodes = [{'start_idx': s, 'end_idx': e} for s, e in afib_episodes_raw]

    total_beats = len(rr_intervals)

    # 6. 滑动窗口采样 + 标签
    all_slices = []
    start_idx = 0
    last_green_idx = None

    while start_idx + WINDOW_BEATS <= total_beats:
        end_idx = start_idx + WINDOW_BEATS

        is_afib = check_overlap(start_idx, end_idx, afib_episodes)
        b_to_next = beats_to_next_afib(end_idx, afib_episodes)
        b_since_last = beats_since_last_afib(start_idx, afib_episodes)

        label = None
        if is_afib:
            label = 1.0
        elif b_to_next <= AFIB_PROXIMITY_BEATS:
            label = 0.3 + 0.7 * (1.0 - (b_to_next / AFIB_PROXIMITY_BEATS))
        elif b_to_next >= SAFE_DISTANCE_BEATS and b_since_last >= SAFE_DISTANCE_BEATS:
            label = 0.0

        if label is not None:
            should_extract = True
            if label == 0.0:
                if last_green_idx is not None and \
                        (start_idx - last_green_idx) < GREEN_SPACING_BEATS:
                    should_extract = False
            if should_extract:
                rr_window = rr_intervals[start_idx:end_idx]
                notes_window = aux_notes[start_idx:end_idx]
                feats_hrv = extract_features(rr_window, notes_window)
                if feats_hrv is not None:
                    all_slices.append({
                        'start_idx': start_idx,
                        'feats': feats_hrv,
                        'label': label
                    })
                    if label == 0.0:
                        last_green_idx = start_idx
        start_idx += STEP_BEATS

    # 7. 组装 6 步时间序列
    X, Y = [], []
    if len(all_slices) >= TIME_STEPS:
        for i in range(len(all_slices) - TIME_STEPS + 1):
            seq = all_slices[i: i + TIME_STEPS]
            # 连续性校验
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

    # 8. 保存张量
    tensor_data = {
        'record': patient_id,
        'X': torch.tensor(X, dtype=torch.float32),
        'Y': torch.tensor(Y, dtype=torch.float32)
    }
    torch.save(tensor_data, save_path)

    n_green = sum(1 for s in all_slices if s['label'] == 0.0)
    n_red = sum(1 for s in all_slices if s['label'] == 1.0)
    n_yellow = len(all_slices) - n_green - n_red
    return (f"OK: {patient_id} | seqs={len(X)} "
            f"| slices={len(all_slices)} (R={n_red} Y={n_yellow} G={n_green}) "
            f"| afib_eps={len(afib_episodes)}")


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    # 获取共有的患者 ID
    signal_files = set(f.replace('.csv', '')
                       for f in os.listdir(SIGNAL_DIR) if f.endswith('.csv'))
    types_files = set(f.replace('.csv', '')
                      for f in os.listdir(TYPES_DIR) if f.endswith('.csv'))
    common_ids = sorted(signal_files & types_files)

    print(f"\n{'='*60}")
    print(f"  Our Holter ECG Database — Training Tensor Generator")
    print(f"  Signal files: {len(signal_files)}")
    print(f"  Types files:  {len(types_files)}")
    print(f"  Common patients: {len(common_ids)}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # 快速预筛选：只处理有 type=19（AFib）的患者
    afib_patients = []
    skipped_no_afib = 0
    for pid in tqdm(common_ids, desc="Scanning for AFib"):
        types_path = os.path.join(TYPES_DIR, f"{pid}.csv")
        try:
            _, beat_types = read_our_holter_types(types_path)
            if AFIB_BEAT_TYPE in beat_types:
                afib_patients.append(pid)
            else:
                skipped_no_afib += 1
        except Exception:
            skipped_no_afib += 1

    print(f"\n  AFib patients: {len(afib_patients)}")
    print(f"  Skipped (no AFib): {skipped_no_afib}")
    print(f"{'='*60}\n")

    # 处理 AFib 患者
    results = []
    for pid in tqdm(afib_patients, desc="Processing AFib patients"):
        result = process_single_patient(pid)
        results.append(result)
        # 每个患者处理后即时打印（方便追踪进度）
        print(f"  {result}")

    # ─── 汇总统计 ───
    success_count = sum(1 for r in results if r.startswith("OK"))
    skipped_exists = sum(1 for r in results if "exists" in r)
    skipped_other = sum(1 for r in results if r.startswith("Skipped") and "exists" not in r)
    failed = sum(1 for r in results if r.startswith("Failed"))

    total_seqs = 0
    total_slices = 0
    total_red = 0
    total_yellow = 0
    total_green = 0

    for r in results:
        if r.startswith("OK"):
            # 解析统计信息
            parts = r.split("|")
            try:
                seqs_part = parts[1].strip()
                total_seqs += int(seqs_part.replace("seqs=", "").strip())

                slices_part = parts[2].strip()
                slices_info = slices_part.replace("slices=", "").strip()
                total_slices += int(slices_info.split(" ")[0])

                import re
                r_match = re.search(r'R=(\d+)', parts[2])
                y_match = re.search(r'Y=(\d+)', parts[2])
                g_match = re.search(r'G=(\d+)', parts[2])
                if r_match:
                    total_red += int(r_match.group(1))
                if y_match:
                    total_yellow += int(y_match.group(1))
                if g_match:
                    total_green += int(g_match.group(1))
            except (IndexError, ValueError):
                pass

    print(f"\n{'='*60}")
    print(f"  PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Success:          {success_count}")
    print(f"  Skipped (exists): {skipped_exists}")
    print(f"  Skipped (other):  {skipped_other}")
    print(f"  Failed:           {failed}")
    print(f"  ---")
    print(f"  Total sequences:  {total_seqs}")
    print(f"  Total slices:     {total_slices}")
    print(f"    Red (AFib):     {total_red}")
    print(f"    Yellow (prox):  {total_yellow}")
    print(f"    Green (safe):   {total_green}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"{'='*60}")
