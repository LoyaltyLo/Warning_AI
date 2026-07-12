"""
SYAF 数据集评估脚本
======================

数据格式:
  Signal CSV (C:\LoyaltyLo\datasets\SYAF\Signal):  3导联 ECG, 200Hz, 24h
    原始ADC值, 需转换为mV: (signal - 2048) / 4096
  Types CSV  (C:\LoyaltyLo\datasets\SYAF\Types):   两列: R波位置, 心搏类型
    类型: 5/204=正常, 19=房颤, 41=室早, 32=房早, 56=unknown, 193=未知

用法:
  source .venv/Scripts/activate
  python evaluate_syaf.py --seed 2048
"""

import os
import torch
import numpy as np
import pandas as pd
import joblib
import multiprocessing as mp
from tqdm import tqdm
from scipy.signal import welch, medfilt, savgol_filter
from scipy.interpolate import interp1d
import antropy as ant
import warnings
import matplotlib
import csv
import re

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train import AFibAttentionSeq2Seq
from logging_config import setup_logging, get_logger

warnings.filterwarnings('ignore')

logger = get_logger(__name__)

# ─── 常量 ──────────────────────────────────────────────
WINDOW_BEATS = 600
STEP_BEATS = 30              # 评估步长（细粒度）
TIME_STEPS = 6
SIGNAL_FS = 200              # SYAF 原始数据采样率
TARGET_FS = 128              # 目标采样率（与训练数据一致）
AFIB_BEAT_TYPE = 19

# 异位心搏类型（用于特征提取中掩码）
ECTOPIC_TYPES = {41, 32}     # 41=PVC(室早), 32=PAC(房早)

# ─── 路径 ──────────────────────────────────────────────
SIGNAL_DIR = r"D:\LoyaltyWorks\datasets\SYAF\Signal"
TYPES_DIR  = r"D:\LoyaltyWorks\datasets\SYAF\Types"
OUTPUT_DIR = "evaluation_results_syaf"


# ═══════════════════════════════════════════════════════
# 辅助函数（与 batch_evaluate_cdss.py 保持一致）
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


def _ewm_smooth(values, span=5):
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _suppress_bottom_noise(probs, threshold=0.25, exponent=1.8):
    result = probs.copy()
    mask = result < threshold
    result[mask] = result[mask] ** exponent
    return result


def _compute_trend(probs, window=3):
    n = len(probs)
    trend = np.zeros(n)
    for i in range(window - 1, n):
        x = np.arange(window)
        y = probs[i - window + 1:i + 1]
        if len(y) == window:
            slope = np.polyfit(x, y, 1)[0]
            trend[i] = slope
    return trend


def _suppress_v_recovery(smoothed_probs, lookback=8, drop_threshold=0.12, max_suppression=0.08):
    n = len(smoothed_probs)
    if n < lookback + 1:
        return smoothed_probs
    suppressed = smoothed_probs.copy()
    for i in range(lookback, n):
        recent_peak = np.max(smoothed_probs[i - lookback:i + 1])
        drop = recent_peak - smoothed_probs[i]
        if drop > drop_threshold:
            v_factor = 1.0 - min(drop * 0.5, max_suppression)
            suppressed[i] *= v_factor
    return suppressed


def _suppress_flatline_probs(smoothed_probs, local_window=20, std_threshold=0.005,
                              max_suppression=0.30, min_mean=0.80):
    """Suppress probabilities that show near-zero local variance (model saturation artifact)."""
    n = len(smoothed_probs)
    if n < local_window:
        return smoothed_probs
    half_w = local_window // 2
    suppressed = smoothed_probs.copy()
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        local_std = np.std(smoothed_probs[lo:hi])
        local_mean = np.mean(smoothed_probs[lo:hi])
        if local_mean > min_mean and local_std < std_threshold:
            flatness = 1.0 - local_std / std_threshold
            factor = 1.0 - max_suppression * flatness
            suppressed[i] *= factor
    return suppressed


def _compute_adaptive_thresholds(smoothed_probs, calibration_windows=30, calib_data=None):
    DEFAULT_P1 = 0.55
    DEFAULT_P2 = 0.85
    DEFAULT_P3 = 0.35
    DEFAULT_P3_TREND = 0.04
    DEFAULT_EXIT = 0.35
    DEFAULT_P1_SUSTAIN = 4    # 6→4, 更快触发提前预警
    DEFAULT_P3_SUSTAIN = 2    # 3→2, 趋势路径更灵敏

    if calib_data is None:
        calib_data = smoothed_probs[:calibration_windows]
    if len(calib_data) < 10:
        return (DEFAULT_P1, DEFAULT_P2, DEFAULT_P3, DEFAULT_P3_TREND,
                DEFAULT_EXIT, DEFAULT_P1, DEFAULT_P1_SUSTAIN, DEFAULT_P3_SUSTAIN)

    baseline_mean = np.mean(calib_data)
    base_shift = baseline_mean - 0.15
    shift = np.clip(base_shift, -0.05, 0.15)

    p1_enter = np.clip(DEFAULT_P1 + shift, 0.45, 0.70)
    p2_enter = np.clip(DEFAULT_P2 + shift * 0.5, 0.75, 0.92)
    p3_enter = np.clip(DEFAULT_P3 + shift * 0.5, 0.25, 0.50)
    p3_trend = DEFAULT_P3_TREND
    exit_thresh = np.clip(DEFAULT_EXIT + shift * 0.5, 0.35, 0.50)
    display_thresh = p1_enter
    p1_sustain = DEFAULT_P1_SUSTAIN
    p3_sustain = DEFAULT_P3_SUSTAIN

    return (p1_enter, p2_enter, p3_enter, p3_trend,
            exit_thresh, display_thresh, p1_sustain, p3_sustain)


def _compute_rolling_thresholds(smoothed_probs, calibration_windows=30, recalibrate_every=30):
    n = len(smoothed_probs)
    p1_arr = np.zeros(n)
    p2_arr = np.zeros(n)
    p3_arr = np.zeros(n)
    p3t_arr = np.zeros(n)
    exit_arr = np.zeros(n)
    disp_arr = np.zeros(n)
    p1s_arr = np.zeros(n, dtype=int)
    p3s_arr = np.zeros(n, dtype=int)

    for seg_start in range(0, n, recalibrate_every):
        seg_end = min(seg_start + recalibrate_every, n)
        calib_start = max(0, seg_start - calibration_windows)
        calib_data = smoothed_probs[calib_start:seg_start]

        if len(calib_data) < 10:
            calib_data = smoothed_probs[:min(calibration_windows, n)]

        (p1, p2, p3, p3t, exit_t, disp_t, p1s, p3s) = \
            _compute_adaptive_thresholds(smoothed_probs, calibration_windows,
                                         calib_data=calib_data)

        p1_arr[seg_start:seg_end] = p1
        p2_arr[seg_start:seg_end] = p2
        p3_arr[seg_start:seg_end] = p3
        p3t_arr[seg_start:seg_end] = p3t
        exit_arr[seg_start:seg_end] = exit_t
        disp_arr[seg_start:seg_end] = disp_t
        p1s_arr[seg_start:seg_end] = p1s
        p3s_arr[seg_start:seg_end] = p3s

    return p1_arr, p2_arr, p3_arr, p3t_arr, exit_arr, disp_arr, p1s_arr, p3s_arr


def _adaptive_alert(smoothed_probs, trend_signal=None,
                    p1_enter=0.50, p2_enter=0.80, p3_enter=0.30,
                    p3_trend=0.05, exit_thresh=0.30,
                    p1_sustain=3, p3_sustain=2,
                    pip_values=None):
    """
    🎯 自适应多路径报警系统 v3.3 — 个体化阈值 + 置信度加权冷却期
    （与 batch_evaluate_cdss.py 完全一致）

    三条独立触发路径，任一路径满足即报警：
    - 路径1 持续中置信度：概率 >= p1_enter 连续 p1_sustain 窗口
    - 路径2 高置信度突发：概率 >= p2_enter 连续 3 窗口
    - 路径3 趋势加速：概率 >= p3_enter 且趋势 >= p3_trend，连续 p3_sustain 窗口

    统一退出：概率 < exit_thresh 连续 3 窗口
    置信度加权冷却期：cd = 8 + (1.0 - alarm_peak) * 20, clamped [8, 20]
      高峰度报警(≥0.85): cd≈8-11 → 快速恢复
      低置信度报警(~0.60): cd≈16 → 延长压制

    P0 异位搏动压制：当 pip > 0.005 时，所有入场阈值 +0.10，
    要求更高置信度才能触发报警。

    Args:
        smoothed_probs: 平滑后的风险概率序列
        trend_signal: 预计算的趋势信号（可选）
        p1_enter: 路径1入场阈值（默认0.50）
        p2_enter: 路径2入场阈值（默认0.80）
        p3_enter: 路径3入场阈值（默认0.35）
        p3_trend: 路径3趋势斜率阈值（默认0.05）
        exit_thresh: 报警退出阈值（默认0.30）
        p1_sustain: 路径1持续性窗口数（默认4，高基线患者为5）
        p3_sustain: 路径3持续性窗口数（默认3，高基线患者为4）
        pip_values: 每窗口原始pip值数组，用于异位搏动检测

    Returns:
        active_alerts: bool数组
    """
    n = len(smoothed_probs)
    active_alerts = np.zeros(n, dtype=bool)

    if trend_signal is None:
        trend_signal = _compute_trend(smoothed_probs, window=3)

    # 支持标量或数组形式的阈值参数（数组用于滚动重校准）
    def _to_arr(val, dtype=float):
        if np.isscalar(val):
            return np.full(n, val, dtype=dtype)
        return np.asarray(val, dtype=dtype)

    _p1 = _to_arr(p1_enter)
    _p2 = _to_arr(p2_enter)
    _p3 = _to_arr(p3_enter)
    _p3t = _to_arr(p3_trend)
    _exit = _to_arr(exit_thresh)
    _p1s = _to_arr(p1_sustain, dtype=int)
    _p3s = _to_arr(p3_sustain, dtype=int)

    p1_count = 0
    p2_count = 0
    p3_count = 0

    state = 'IDLE'
    fall_count = 0
    cooldown_count = 0
    cooldown_limit = 12  # 动态冷却期上限，由报警峰值置信度决定
    alarm_start = 0
    alarm_peak_prob = 0.0

    def _compute_cooldown(peak):
        """置信度加权冷却期：高峰度→短冷却，低置信度→长冷却"""
        return int(np.clip(8 + (1.0 - peak) * 12, 8, 12))

    for i in range(n):
        p = smoothed_probs[i]
        t = trend_signal[i]

        # P0 异位搏动压制：pip高时抬高入场阈值
        ectopic_boost = 0.10 if (pip_values is not None and i < len(pip_values)
                                  and pip_values[i] > 0.005) else 0.0

        # 三路径触发判断（使用当前窗口的个体化阈值 + 异位搏动偏移）
        path1_fire = p >= (_p1[i] + ectopic_boost)
        path2_fire = p >= (_p2[i] + ectopic_boost)
        path3_fire = p >= (_p3[i] + ectopic_boost) and t >= _p3t[i]

        p1_count = p1_count + 1 if path1_fire else 0
        p2_count = p2_count + 1 if path2_fire else 0
        p3_count = p3_count + 1 if path3_fire else 0

        if state == 'IDLE':
            triggered = (p1_count >= _p1s[i]) or (p2_count >= 2) or (p3_count >= _p3s[i])
            if triggered:
                state = 'ALARM'
                alarm_peak_prob = p
                if p2_count >= 2:
                    alarm_start = max(0, i - p2_count + 1)
                elif p1_count >= _p1s[i]:
                    alarm_start = max(0, i - p1_count + 1)
                else:
                    alarm_start = max(0, i - p3_count + 1)
                active_alerts[alarm_start:i + 1] = True
                p1_count = p2_count = p3_count = 0

        elif state == 'ALARM':
            alarm_peak_prob = max(alarm_peak_prob, p)
            if p >= _exit[i]:
                active_alerts[i] = True
                fall_count = 0
            else:
                fall_count += 1
                if fall_count >= 3:
                    state = 'COOLDOWN'
                    cooldown_count = 0
                    cooldown_limit = _compute_cooldown(alarm_peak_prob)
                    p1_count = p2_count = p3_count = 0
                else:
                    active_alerts[i] = True

        elif state == 'COOLDOWN':
            cooldown_count += 1
            triggered = (p1_count >= _p1s[i]) or (p2_count >= 2) or (p3_count >= _p3s[i])
            if triggered:
                state = 'ALARM'
                alarm_peak_prob = p
                if p2_count >= 2:
                    alarm_start = max(0, i - p2_count + 1)
                elif p1_count >= _p1s[i]:
                    alarm_start = max(0, i - p1_count + 1)
                else:
                    alarm_start = max(0, i - p3_count + 1)
                active_alerts[alarm_start:i + 1] = True
                p1_count = p2_count = p3_count = 0
                cooldown_count = 0
                cooldown_limit = _compute_cooldown(alarm_peak_prob)
            if cooldown_count >= cooldown_limit:
                state = 'IDLE'

    return active_alerts


def extract_ai_alarm_episodes(time_axis_mins, active_alerts):
    """从布尔报警序列中提取报警片段"""
    alarm_episodes = []
    in_alarm = False
    alarm_start = None
    for i in range(len(active_alerts)):
        if active_alerts[i] and not in_alarm:
            in_alarm = True
            alarm_start = time_axis_mins[i]
        elif not active_alerts[i] and in_alarm:
            in_alarm = False
            alarm_episodes.append({
                'start': alarm_start,
                'end': time_axis_mins[i - 1]
            })
    if in_alarm:
        alarm_episodes.append({
            'start': alarm_start,
            'end': time_axis_mins[-1]
        })
    return alarm_episodes


# ═══════════════════════════════════════════════════════
# 特征提取（与 batch_processor_shdb.py 保持一致）
# ═══════════════════════════════════════════════════════

def extract_features(rr_window, aux_notes=None):
    """
    心搏域特征提取 — 与 batch_processor_shdb.py 完全一致。
    返回 14 维 HRV 特征。
    """
    total_beats = len(rr_window)
    if total_beats < 50:
        return None

    rr = rr_window.copy().astype(float)

    # 异位心搏掩码
    if aux_notes is not None:
        ectopic = {'V', 'A', 'a', 'J', 'S'}
        for i in range(min(len(aux_notes), len(rr))):
            if str(aux_notes[i]).strip() in ectopic:
                rr[i] = np.nan

    raw_rr = rr[~np.isnan(rr)]
    if len(raw_rr) < 30:
        return None

    # SQA 门控
    if np.sum((raw_rr < 400) | (raw_rr > 3000)) / len(raw_rr) > 0.05:
        return None
    if len(raw_rr) > 1 and np.sum(np.abs(np.diff(raw_rr)) > 300) / len(raw_rr) > 0.05:
        return None

    rr_clean = _interpolate_nans(rr)

    # 中值滤波
    rr_filtered = medfilt(rr_clean, kernel_size=3)

    # PAC 补偿性中和
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

    # 容差死区 (50.0ms)
    rr_diff_clean = np.where(np.abs(rr_diff) < 50.0, 0.0, rr_diff)

    mean_rr = np.mean(rr_data_clean)
    std_rr = np.std(rr_data_clean)
    cv = std_rr / mean_rr if mean_rr > 0 else 0.0
    median_rr = np.median(rr_data_clean)
    mad = np.median(np.abs(rr_data_clean - median_rr))

    rmssd = np.sqrt(np.mean(rr_diff_clean ** 2)) if len(rr_diff_clean) > 0 else 0.0
    pnn50 = np.sum(np.abs(rr_diff_clean) > 50) / len(rr_diff_clean) if len(rr_diff_clean) > 0 else 0.0

    # Soft Noise Gate v2 — threshold 25ms
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

    # 呼吸性窦性心律不齐周期性检测
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
# SYAF 数据读取
# ═══════════════════════════════════════════════════════

def read_syaf_types(types_path):
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


def read_syaf_signal(signal_path, max_samples=None):
    """
    读取 Signal CSV 文件并转换为 mV。
    公式: (signal - 2048) / 4096

    SYAF Signal CSV 格式: 3 列 (3 导联), 200Hz.
    文件每隔一行有空行 (Windows \r\n 双换行), 使用 pandas 自动跳过空行.

    返回: ecg_mv (3, n_samples) 或 None 若文件不存在
    """
    if not os.path.exists(signal_path):
        return None

    # 使用 pandas 读取 (自动处理空行)
    nrows = max_samples if max_samples else None
    df = pd.read_csv(signal_path, header=None,
                     skip_blank_lines=True, nrows=nrows)
    data = df.values.astype(np.float64)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # ADC → mV: (adc - 2048) / 4096
    data_mv = (data - 2048.0) / 4096.0

    # 转置为 (n_leads, n_samples)
    ecg_mv = data_mv.T
    return ecg_mv


def find_afib_episodes_from_beats(beat_types, min_afib_beats=10, merge_gap=300):
    """
    从逐搏类型中识别房颤发作区间。

    Args:
        beat_types: 心搏类型数组
        min_afib_beats: 最少连续AFib心搏数（少于此次数的AFib片段忽略）
        merge_gap: 两个AFib片段之间的最大间隔（心搏数），小于此间隔的合并为一个发作

    Returns:
        [(start_beat_idx, end_beat_idx), ...]  发作区间（心搏索引，左闭右闭）
    """
    # 找到所有 type==19 的搏动位置
    afib_mask = (beat_types == AFIB_BEAT_TYPE)

    # 识别连续片段
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


# ═══════════════════════════════════════════════════════
# 单患者评估
# ═══════════════════════════════════════════════════════

def evaluate_single_syaf(args):
    patient_id, types_path, signal_path, model_path, scaler_path, plot_out_dir = args

    try:
        # 1. 读取 Types → R波位置 + 心搏类型
        r_positions, beat_types = read_syaf_types(types_path)
        if len(r_positions) < WINDOW_BEATS + 100:
            return {'status': 'skipped', 'msg': f'{patient_id}: 心搏数不足 ({len(r_positions)})'}

        # 2. 重采样 R 波位置: 200Hz → 128Hz（与训练数据一致）
        r_positions = np.round(r_positions * TARGET_FS / SIGNAL_FS).astype(int)

        # 3. 计算 RR 间期 (ms) — 基于128Hz重采样后的位置
        rr_intervals = np.zeros(len(r_positions))
        if len(r_positions) > 1:
            rr_intervals[1:] = np.diff(r_positions) / TARGET_FS * 1000.0
            rr_intervals[0] = rr_intervals[1]

        # 3. 构建 aux_notes（用于异位心搏掩码）
        #    将 ECTOPIC_TYPES 映射为 'V' 类型，AFib 映射为 '(AFIB'，其它为空
        aux_notes = []
        for bt in beat_types:
            if bt in ECTOPIC_TYPES:
                aux_notes.append('V')     # PVC/PAC → ectopic mask
            elif bt == AFIB_BEAT_TYPE:
                aux_notes.append('(AFIB')
            else:
                aux_notes.append('')
        aux_notes = np.array(aux_notes, dtype=object)

        # 4. 识别 AFib 发作区间（心搏索引）
        afib_episodes_beat = find_afib_episodes_from_beats(beat_types)
        if not afib_episodes_beat:
            return {'status': 'skipped', 'msg': f'{patient_id}: 无AFib发作'}

        # 5. 累积时间（从RR间期推导）
        cum_time_mins = np.zeros(len(rr_intervals))
        if len(rr_intervals) > 1:
            cum_time_mins[1:] = np.cumsum(rr_intervals[1:]) / 60000.0

        total_duration_mins = cum_time_mins[-1]
        total_beats = len(rr_intervals)

        # 6. 将 AFib 发作转换为分钟
        gt_episodes = []
        for start_beat, end_beat in afib_episodes_beat:
            gt_episodes.append({
                'start': cum_time_mins[start_beat],
                'end': cum_time_mins[min(end_beat, len(cum_time_mins) - 1)]
            })

        # 7. 加载模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AFibAttentionSeq2Seq(input_dim=29, hidden_dim=128).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        scaler = joblib.load(scaler_path)

        # 8. 滑动窗口推理
        history_beats = WINDOW_BEATS * TIME_STEPS  # 3600 beats
        if len(rr_intervals) < history_beats:
            return {'status': 'skipped', 'msg': f'{patient_id}: 数据太短'}

        time_axis_mins = []
        all_raw_seqs = []
        pip_values = []
        current_beat = history_beats
        last_valid_features = [0.0] * 28

        # 尝试读取信号（用于波形可视化）
        ecg_signal = None
        try:
            ecg_signal = read_syaf_signal(signal_path)
        except Exception:
            pass

        # Feature cache: pre-compute once per window position (avoids redundant extraction)
        feature_cache = {}

        def _get_features(beat_pos):
            key = beat_pos // STEP_BEATS
            if key not in feature_cache:
                w_end = beat_pos
                w_start = max(0, w_end - WINDOW_BEATS)
                if w_end - w_start >= 50:
                    feat = extract_features(rr_intervals[w_start:w_end], aux_notes[w_start:w_end])
                    feature_cache[key] = feat if feat is not None else last_valid_features
                else:
                    feature_cache[key] = last_valid_features
            return feature_cache[key]

        while current_beat < total_beats:
            sequence_features = []
            for seq_i in range(TIME_STEPS):
                w_end = current_beat - (TIME_STEPS - 1 - seq_i) * WINDOW_BEATS
                feats = _get_features(w_end)
                sequence_features.append(feats)

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
            return {'status': 'skipped', 'msg': f'{patient_id}: 无有效特征'}

        # 9. 模型推理
        X_raw_array = np.array(all_raw_seqs)
        X_flat = X_raw_array.reshape(-1, 29)
        X_norm_flat = scaler.transform(X_flat)
        X_norm_array = X_norm_flat.reshape(-1, TIME_STEPS, 29)

        X_tensor = torch.tensor(X_norm_array, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs, _ = model(X_tensor)
            ai_risk_probs = probs[:, -1].cpu().numpy().tolist()

        # 10. 后处理管线
        raw_probs = np.array(ai_risk_probs)

        # Layer 1: 底部噪声压制
        suppressed = _suppress_bottom_noise(raw_probs, threshold=0.25, exponent=1.8)

        # Layer 2: EWM平滑
        ewm_smoothed = _ewm_smooth(suppressed, span=5)

        # Layer 3: Savitzky-Golay滤波
        if len(ewm_smoothed) >= 11:
            smoothed_probs = savgol_filter(ewm_smoothed, window_length=11, polyorder=2)
            smoothed_probs = np.clip(smoothed_probs, 0.0, 1.0)
        else:
            smoothed_probs = ewm_smoothed

        # Layer 4: S2 多尺度趋势一致性门控
        trend_3w = _compute_trend(smoothed_probs, window=3)
        trend_7w = _compute_trend(smoothed_probs, window=7)
        consensus_trend = np.minimum(trend_3w, trend_7w)
        trend_gate = np.clip(0.95 + consensus_trend * 3.0, 0.95, 1.0)  # 0.92→0.95, 减少信号衰减
        smoothed_probs = smoothed_probs * trend_gate

        # 🛡️ P1: V型恢复压制 — 压制"上升后快速回落"假阳性
        # 真AFib前驱单调递增，NSR假阳性呈V型尖峰
        smoothed_probs = _suppress_v_recovery(smoothed_probs)

        # 自适应阈值校准
        (p1_enter, p2_enter, p3_enter, p3_trend,
         exit_thresh, display_thresh, p1_sustain, p3_sustain) = \
            _compute_adaptive_thresholds(smoothed_probs, calibration_windows=30)

        # 滚动重校准
        (rolling_p1, rolling_p2, rolling_p3, rolling_p3t,
         rolling_exit, rolling_disp, rolling_p1s, rolling_p3s) = \
            _compute_rolling_thresholds(smoothed_probs, calibration_windows=30, recalibrate_every=30)

        # 自适应多路径报警
        active_alerts = _adaptive_alert(smoothed_probs, trend_signal=consensus_trend,
                                        p1_enter=rolling_p1, p2_enter=rolling_p2,
                                        p3_enter=rolling_p3, p3_trend=rolling_p3t,
                                        exit_thresh=rolling_exit,
                                        p1_sustain=rolling_p1s, p3_sustain=rolling_p3s,
                                        pip_values=pip_values)
        ai_alarms = extract_ai_alarm_episodes(time_axis_mins, active_alerts)
        alert_threshold = display_thresh

        # 11. 事件级评估 v2
        MAX_PREDICT_MINS = 120.0
        # --- 合并 GT: 临床窗口重叠原则 ---
        merged_gt = []
        for gt in sorted(gt_episodes, key=lambda g: g['start']):
            gt_win_start = gt['start'] - MAX_PREDICT_MINS
            if merged_gt and (gt_win_start <= merged_gt[-1]['end']):
                merged_gt[-1]['end'] = max(merged_gt[-1]['end'], gt['end'])
            else:
                merged_gt.append({'start': gt['start'], 'end': gt['end']})

        metrics = {
            'patient_id': patient_id,
            'total_duration_hours': total_duration_mins / 60.0,
            'gt_episodes_count': len(merged_gt),
            'gt_episodes_raw_count': len(gt_episodes),
            'caught_episodes_count': 0,
            'early_warning_times': [],
            'detection_times': [],
            'false_alarms_count': 0,
            'total_alarm_mins': 0.0,
            'total_afib_mins': 0.0,
            'afib_covered_mins': 0.0,
            'alarm_confidences': [],
        }

        total_alarm_mins = sum(max(0, a['end'] - a['start']) for a in ai_alarms)
        metrics['total_alarm_mins'] = total_alarm_mins
        total_afib_mins = sum(gt['end'] - gt['start'] for gt in merged_gt)
        metrics['total_afib_mins'] = total_afib_mins

        # --- 一告警一命中 + 最优EWT ---
        used_alarm_indices = set()
        matched_alarm_indices = set()

        for gt in merged_gt:
            best_alarm_idx = None
            best_ewt = -999
            for idx, alarm in enumerate(ai_alarms):
                if idx in used_alarm_indices:
                    continue
                cw_start = gt['start'] - MAX_PREDICT_MINS
                cw_end = gt['end']
                if (alarm['start'] <= cw_end) and (alarm['end'] >= cw_start):
                    ewt_candidate = gt['start'] - max(alarm['start'], cw_start)
                    if ewt_candidate > best_ewt:
                        best_ewt = ewt_candidate
                        best_alarm_idx = idx

            if best_alarm_idx is not None:
                metrics['caught_episodes_count'] += 1
                used_alarm_indices.add(best_alarm_idx)
                matched_alarm_indices.add(best_alarm_idx)
                ewt = max(0, best_ewt)
                if best_ewt > 0:
                    metrics['early_warning_times'].append(ewt)
                else:
                    metrics['detection_times'].append(0.0)

        # --- AFib 覆盖时长 ---
        afib_covered = 0.0
        for gt in merged_gt:
            for idx in matched_alarm_indices:
                alarm = ai_alarms[idx]
                o_start = max(alarm['start'], gt['start'])
                o_end = min(alarm['end'], gt['end'])
                if o_start < o_end:
                    afib_covered += (o_end - o_start)
        metrics['afib_covered_mins'] = afib_covered

        # --- 告警置信度 ---
        for idx in range(len(ai_alarms)):
            alarm = ai_alarms[idx]
            prob_vals = [smoothed_probs[ti] for ti, t in enumerate(time_axis_mins)
                         if alarm['start'] <= t <= alarm['end'] and ti < len(smoothed_probs)]
            if prob_vals:
                metrics['alarm_confidences'].append(float(np.mean(prob_vals)))

        # --- 确定首次TP时间点（首次命中后不再计FP，节律已被破坏）---
        first_tp_time = None
        if matched_alarm_indices:
            first_matched_idx = min(matched_alarm_indices)
            first_tp_time = ai_alarms[first_matched_idx]['start']

        # --- 误报统计 ---
        for idx, alarm in enumerate(ai_alarms):
            if idx in matched_alarm_indices:
                continue
            # 首次TP之后的告警：节律已被房颤破坏，不计为FP
            if first_tp_time is not None and alarm['start'] >= first_tp_time:
                continue
            is_fa = True
            for gt in merged_gt:
                if (alarm['start'] <= gt['end']) and (alarm['end'] >= gt['start'] - MAX_PREDICT_MINS):
                    is_fa = False
                    break
            if is_fa:
                metrics['false_alarms_count'] += 1

        # 12. 绘图 — 宏观预警曲线
        if plot_out_dir:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(15, 6))

            ax.plot(time_axis_mins, smoothed_probs, 'r-', linewidth=2.5, label='AI Real-time Risk')
            ax.fill_between(time_axis_mins, 0, smoothed_probs, where=active_alerts,
                            color='salmon', alpha=0.3, label='Active Clinical Alert')
            ax.axhline(y=alert_threshold, color='orange', linestyle=':', linewidth=2,
                       label=f'Adaptive Threshold ({alert_threshold:.2f})')

            for idx, gt in enumerate(gt_episodes):
                label = 'Expert Annotations: AFib' if idx == 0 else ""
                ax.axvspan(gt['start'], gt['end'], color='black', alpha=0.15, label=label)

            history_mins = cum_time_mins[history_beats] if history_beats < len(cum_time_mins) else cum_time_mins[-1]
            ax.legend(loc='upper left', fontsize=11)
            ax.set_title(f"Clinical CDSS Dashboard: Patient {patient_id} [SYAF]", fontsize=14, fontweight='bold')
            ax.set_xlabel("Continuous Monitoring Time (Minutes)", fontsize=13)
            ax.set_ylabel("Risk Probability (P_AFib)", fontsize=13)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(history_mins, total_duration_mins)
            plt.tight_layout()

            plot_filename = os.path.join(plot_out_dir, f"patient_{patient_id}_curve.png")
            plt.savefig(plot_filename, dpi=200)
            plt.close(fig)

        return {'status': 'success', 'metrics': metrics}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'msg': f'{patient_id}: {str(e)}'}


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=128, help='Model seed')
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of patients (0=all)')
    args = parser.parse_args()

    setup_logging(log_file="logs/evaluate_syaf.log")

    model_path = f"best_afib_model_s{args.seed}.pth"
    scaler_path = f"feature_scaler_s{args.seed}.pkl"

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print(f"  Available models: {[f for f in os.listdir('.') if f.startswith('best_afib_model')]}")
        exit(1)
    if not os.path.exists(scaler_path):
        print(f"[ERROR] Scaler not found: {scaler_path}")
        exit(1)

    # 获取共有的文件
    signal_files = set(f.replace('.csv', '') for f in os.listdir(SIGNAL_DIR) if f.endswith('.csv'))
    types_files = set(f.replace('.csv', '') for f in os.listdir(TYPES_DIR) if f.endswith('.csv'))
    common_ids = sorted(signal_files & types_files)

    print(f"\n{'='*60}")
    print(f"  SYAF Dataset Evaluation")
    print(f"  Model: {model_path}")
    print(f"  Signal files: {len(signal_files)}  |  Types files: {len(types_files)}")
    print(f"  Common patients: {len(common_ids)}")
    print(f"{'='*60}\n")

    if args.limit > 0:
        common_ids = common_ids[:args.limit]
        print(f"  Limited to {args.limit} patients\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    strip_out_dir = os.path.join(OUTPUT_DIR, "Warning_Strips")
    os.makedirs(strip_out_dir, exist_ok=True)

    # 组装参数
    args_list = []
    for pid in common_ids:
        types_path = os.path.join(TYPES_DIR, f"{pid}.csv")
        signal_path = os.path.join(SIGNAL_DIR, f"{pid}.csv")
        args_list.append((pid, types_path, signal_path, model_path, scaler_path, OUTPUT_DIR))

    # Single-process (mp.Pool breaks CUDA on Windows spawn)
    results = []
    for res in tqdm(map(evaluate_single_syaf, args_list), total=len(args_list),
                    desc="Evaluating SYAF"):
            results.append(res)

    # ─── 汇总统计 v3 ───
    total_hours = 0
    total_gt_episodes = 0
    total_gt_raw = 0
    total_caught_episodes = 0
    all_ewts = []
    all_detections = []
    total_false_alarms = 0
    total_alarm_mins = 0.0
    total_afib_mins = 0.0
    total_afib_covered_mins = 0.0
    all_confidences = []
    successful_patients = 0
    patient_reports = []

    for r in results:
        if r['status'] == 'success':
            successful_patients += 1
            m = r['metrics']
            total_hours += m['total_duration_hours']
            total_gt_episodes += m['gt_episodes_count']
            total_gt_raw += m.get('gt_episodes_raw_count', m['gt_episodes_count'])
            total_caught_episodes += m['caught_episodes_count']
            all_ewts.extend(m['early_warning_times'])
            all_detections.extend(m.get('detection_times', []))
            total_false_alarms += m['false_alarms_count']
            total_alarm_mins += m.get('total_alarm_mins', 0)
            total_afib_mins += m.get('total_afib_mins', 0)
            total_afib_covered_mins += m.get('afib_covered_mins', 0)
            all_confidences.extend(m.get('alarm_confidences', []))
            patient_reports.append(m)

    patient_reports.sort(key=lambda x: x['patient_id'])

    tp = total_caught_episodes
    fp = total_false_alarms
    sensitivity = (tp / total_gt_episodes * 100) if total_gt_episodes > 0 else 0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    f1_score = (2 * sensitivity * precision) / (sensitivity + precision) if (sensitivity + precision) > 0 else 0

    n_early = len(all_ewts)
    n_detect = len(all_detections)
    mean_ewt = np.mean(all_ewts) if all_ewts else 0
    median_ewt = np.median(all_ewts) if all_ewts else 0
    max_ewt = np.max(all_ewts) if all_ewts else 0
    early_rate = (n_early / tp * 100) if tp > 0 else 0
    far_per_24h = (fp / total_hours * 24) if total_hours > 0 else 0

    total_mins = total_hours * 60.0
    alarm_burden = (total_alarm_mins / total_mins * 100) if total_mins > 0 else 0
    afib_coverage = (total_afib_covered_mins / total_afib_mins * 100) if total_afib_mins > 0 else 0
    nsr_mins = total_mins - total_afib_mins
    nsr_alarm_mins = total_alarm_mins - total_afib_covered_mins
    nsr_far_time = (nsr_alarm_mins / nsr_mins * 100) if nsr_mins > 0 else 0
    mean_conf = np.mean(all_confidences) if all_confidences else 0
    median_conf = np.median(all_confidences) if all_confidences else 0

    patient_sens = []
    for p in patient_reports:
        gt = p['gt_episodes_count']
        if gt > 0:
            patient_sens.append(p['caught_episodes_count'] / gt * 100)
    patient_sens = np.array(patient_sens) if patient_sens else np.array([0])

    # ─── 报告生成 v3 ───
    report_lines = []
    report_lines.append("\n" + "=" * 72)
    report_lines.append("  SYAF CLINICAL CDSS EVALUATION REPORT v3")
    report_lines.append("=" * 72)
    report_lines.append(f"  Model: {model_path}  |  Patients: {successful_patients}/{len(common_ids)}  |  Duration: {total_hours:.1f}h")
    report_lines.append("")

    report_lines.append("─" * 72)
    report_lines.append("  [1] EVENT-LEVEL METRICS")
    report_lines.append("─" * 72)
    report_lines.append(f"  GT episodes (merged):  {total_gt_episodes}")
    if total_gt_raw != total_gt_episodes:
        report_lines.append(f"  GT episodes (raw):     {total_gt_raw}")
    report_lines.append(f"  Caught / Missed / FA:  {tp} / {total_gt_episodes - tp} / {fp}")
    report_lines.append(f"  Sensitivity:          {sensitivity:.1f}%")
    report_lines.append(f"  Precision:            {precision:.1f}%")
    report_lines.append(f"  F1 Score:             {f1_score:.1f}%")
    report_lines.append(f"  FAR (event):          {far_per_24h:.1f} / 24h")
    report_lines.append(f"  Per-patient Sens:     med={np.median(patient_sens):.0f}% min={np.min(patient_sens):.0f}% p25={np.percentile(patient_sens,25):.0f}% p75={np.percentile(patient_sens,75):.0f}% max={np.max(patient_sens):.0f}%")

    report_lines.append("")
    report_lines.append("─" * 72)
    report_lines.append("  [2] TIME-DOMAIN METRICS")
    report_lines.append("─" * 72)
    report_lines.append(f"  AFib coverage:        {afib_coverage:.1f}% of AFib time covered")
    report_lines.append(f"  Alarm burden:         {alarm_burden:.1f}% of total time")
    report_lines.append(f"  NSR alarm time:       {nsr_far_time:.1f}% of NSR time")

    report_lines.append("")
    report_lines.append("─" * 72)
    report_lines.append("  [3] EARLY WARNING QUALITY")
    report_lines.append("─" * 72)
    report_lines.append(f"  Early warning (EWT>0): {n_early}/{tp} ({early_rate:.0f}%)")
    report_lines.append(f"  Detection (EWT=0):    {n_detect}/{tp}")
    if n_early > 0:
        report_lines.append(f"  Mean / Med / Max EWT: {mean_ewt:.1f} / {median_ewt:.1f} / {max_ewt:.1f} min")
    report_lines.append(f"  Alarm confidence:     mean={mean_conf:.2f} median={median_conf:.2f}")

    report_lines.append("")
    report_lines.append("─" * 72)
    report_lines.append("  [4] PER-PATIENT DETAIL")
    report_lines.append("─" * 72)
    report_lines.append(f"  {'ID':<14} {'GT':>4} {'Hit':>4} {'FA':>4} {'Sens':>5} {'EWT':>6} {'Alarm%':>6} {'St'}")

    for p in patient_reports:
        p_id = p['patient_id']
        p_gt = p['gt_episodes_count']
        p_hit = p['caught_episodes_count']
        p_fa = p['false_alarms_count']
        p_sens = (p_hit / p_gt * 100) if p_gt > 0 else 0
        p_ewts = p['early_warning_times']
        p_dets = p.get('detection_times', [])
        p_ewt = np.mean(p_ewts + p_dets) if (p_ewts + p_dets) else 0
        p_alarm_pct = (p.get('total_alarm_mins', 0) / (p['total_duration_hours'] * 60) * 100) if p['total_duration_hours'] > 0 else 0
        if p_hit == 0 and p_gt > 0:
            st = "MISS"
        elif p_hit < p_gt:
            st = "PART"
        elif p_fa > 3:
            st = f"FA={p_fa}"
        else:
            st = "OK"
        report_lines.append(f"  {p_id:<14} {p_gt:>4} {p_hit:>4} {p_fa:>4} {p_sens:>4.0f}% {p_ewt:>5.0f}m {p_alarm_pct:>5.1f}% {st}")

    report_lines.append("")
    report_lines.append("=" * 72)

    final_report_text = "\n".join(report_lines)
    print(final_report_text)

    # 保存报告
    report_filename = os.path.join(OUTPUT_DIR, "evaluation_report_syaf.txt")
    try:
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(final_report_text)
        print(f"\n  Report saved to: {report_filename}")
    except Exception as e:
        print(f"  Report save failed: {e}")
