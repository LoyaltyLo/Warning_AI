import os
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
from logging_config import setup_logging, get_logger

warnings.filterwarnings('ignore')

logger = get_logger(__name__)

# 心搏域常量
WINDOW_BEATS = 600           # ~10 min at 60bpm
STEP_BEATS = 30              # ~30 sec at 60bpm (PAFNet-style fine-grained update)
TIME_STEPS = 6               # 序列长度


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


def _ewm_smooth(values, span=5):
    """指数加权移动平均（等价于 pd.Series.ewm(span=span, adjust=False).mean()）"""
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _suppress_bottom_noise(probs, threshold=0.25, exponent=1.8):
    """底部噪声压制：低于 threshold 的概率做指数压制，抑制低区抖动"""
    result = probs.copy()
    mask = result < threshold
    result[mask] = result[mask] ** exponent
    return result


def _compute_trend(probs, window=3):
    """计算局部趋势：最近window个窗口的线性回归斜率"""
    n = len(probs)
    trend = np.zeros(n)
    for i in range(window - 1, n):
        x = np.arange(window)
        y = probs[i - window + 1:i + 1]
        if len(y) == window:
            slope = np.polyfit(x, y, 1)[0]
            trend[i] = slope
    return trend


def _suppress_flatline_probs(smoothed_probs, local_window=20, std_threshold=0.005,
                              max_suppression=0.30, min_mean=0.80):
    """Suppress probabilities that show near-zero local variance (model saturation artifact).

    Targets the sigmoid saturation ceiling (~0.9247) where the model outputs identical
    values across all windows. Real AFib has dynamics (rising/falling probs) and won't
    have std < 0.005 over 20 windows. Parameters are deliberately tight to avoid
    suppressing legitimate sustained AFib detections.
    """
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


def _suppress_v_recovery(smoothed_probs, lookback=8, drop_threshold=0.12, max_suppression=0.08):
    """
    V型恢复检测：识别"上升后快速回落"的假阳性特征。
    真AFib前驱：概率单向上升直到发作（单调递增）。
    NSR假阳性：概率在中等水平形成V型峰（上升→见顶→回落）。

    对从近期峰值显著回落的窗口施加轻度压制。
    """
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


def _compute_adaptive_thresholds(smoothed_probs, calibration_windows=30, calib_data=None):
    """
    🎯 固定阈值校准 v4 — 轻度个体化，严格保守

    设计原则：
    - 默认阈值已从临床验证中证明有效
    - 仅对高基线患者做轻度抬升（防止噪声误报）
    - 不对低基线患者降低阈值（防止过度敏感）
    - 偏移量仅允许正向（抬升），范围 [0, 0.15]

    返回8个值：
    - p1_enter, p2_enter, p3_enter, p3_trend, exit_thresh, display_thresh
    - p1_sustain, p3_sustain
    """
    DEFAULT_P1 = 0.55
    DEFAULT_P2 = 0.85
    DEFAULT_P3 = 0.35
    DEFAULT_P3_TREND = 0.04
    DEFAULT_EXIT = 0.35
    DEFAULT_P1_SUSTAIN = 4
    DEFAULT_P3_SUSTAIN = 2

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
    """
    🔄 滚动重校准：每隔 recalibrate_every 窗口用最近 calibration_windows 窗口重新校准阈值。

    解决单次校准在长时监控中因基线漂移导致的误报堆积问题（如NSR患者16272在323分钟后开始密集误报）。
    每个窗口位置使用该窗口之前的最新校准参数。
    """
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

    三条独立触发路径，任一路径满足即报警：
    - 路径1 持续中置信度：概率 >= p1_enter 连续 p1_sustain 窗口
    - 路径2 高置信度突发：概率 >= p2_enter 连续 2 窗口
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


def extract_features(rr_window, aux_notes=None):
    """
    心搏域特征提取 v2.0：增强NSR判别力

    改进点：
    1. 容差死区 40→50ms（过滤呼吸性窦性心律不齐）
    2. Soft Noise Gate 阈值 10→20ms（更严格压制低变异时的非线性特征）
    3. 新增呼吸性窦性心律不齐周期性检测（NSR特征，AFib不存在）
    4. 新增SD2绝对值（关键判别特征：NSR的SD2远大于AFib）
    5. 代偿中和阈值收紧 0.92/1.08 → 0.94/1.06（更积极中和早搏）
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

    # 🌟 3. 早搏/二联律代偿中和 (0.94/1.06) — 收紧阈值更积极中和
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

    # 🚀 4. 容差死区 (50.0ms) — 从40ms提高到50ms
    # 过滤呼吸性窦性心律不齐的典型变异（吸气缩短50-80ms）
    rr_diff_clean = np.where(np.abs(rr_diff) < 50.0, 0.0, rr_diff)

    mean_rr = np.mean(rr_data_clean)
    std_rr = np.std(rr_data_clean)
    cv = std_rr / mean_rr if mean_rr > 0 else 0.0
    median_rr = np.median(rr_data_clean)
    mad = np.median(np.abs(rr_data_clean - median_rr))

    rmssd = np.sqrt(np.mean(rr_diff_clean ** 2)) if len(rr_diff_clean) > 0 else 0.0
    pnn50 = np.sum(np.abs(rr_diff_clean) > 50) / len(rr_diff_clean) if len(rr_diff_clean) > 0 else 0.0

    # 🛡️ 5. Soft Noise Gate v2 — 提高下限从10→25ms
    # NSR患者RMSSD常在20-40ms，旧门控几乎不压制
    # 新门控：25ms以下完全压制，50ms以上才完全放开
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
    else:
        pip_raw, pip, sd1, sd2, poincare_ratio = 0.0, 0.0, 0.0, 0.0, 0.0

    # 🌟 6. 呼吸性窦性心律不齐周期性检测
    # NSR特征：RR间期随呼吸周期性波动（0.15-0.40 Hz）
    # AFib特征：RR间期完全不规则，无周期性
    # 如果检测到明显的呼吸周期性 → 降低特征激活度
    respiratory_periodicity = 0.0  # 默认无周期性
    if len(rr_data_clean) > 30 and std_rr > 5.0:
        try:
            time_x_local = np.cumsum(rr_data_clean) / 1000.0
            time_x_local = time_x_local - time_x_local[0]
            if time_x_local[-1] > 30.0:  # 至少30秒数据
                f_interp = interp1d(time_x_local, rr_data_clean, kind='cubic', fill_value="extrapolate")
                fs_local = 4.0
                t_uniform = np.arange(0, time_x_local[-1], 1 / fs_local)
                rr_uniform = f_interp(t_uniform)
                f_psd, pxx_psd = welch(rr_uniform, fs_local, nperseg=min(128, len(rr_uniform)))
                # 呼吸频段功率 0.15-0.40 Hz
                resp_power = np.trapezoid(pxx_psd[(f_psd >= 0.15) & (f_psd < 0.40)],
                                          f_psd[(f_psd >= 0.15) & (f_psd < 0.40)])
                total_power = np.trapezoid(pxx_psd[(f_psd >= 0.04) & (f_psd < 0.40)],
                                           f_psd[(f_psd >= 0.04) & (f_psd < 0.40)])
                if total_power > 1e-6:
                    respiratory_periodicity = resp_power / total_power
                    # 高呼吸周期性 → 这个值接近1.0说明变异主要是呼吸驱动的
        except:
            respiratory_periodicity = 0.0

    # 🌟 7. SD2绝对值 — 关键判别特征
    # NSR窦性心律不齐：SD2 >> SD1，SD2 ≈ 80-150ms
    # AFib前驱：SD2 ≈ SD1，SD2 ≈ 30-60ms
    # 归一化SD2：相对于mean_RR的比例，消除心率差异
    sd2_normalized = sd2 / mean_rr if mean_rr > 0 else 0.0

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

    # 🌟 呼吸周期性压制因子：
    # 如果变异主要是呼吸驱动的（respiratory_periodicity > 0.5），
    # 则对容易受呼吸影响的特征做额外压制
    resp_suppression = 1.0 - np.clip(respiratory_periodicity - 0.25, 0.0, 0.5)

    # 对呼吸敏感特征做周期性压制
    cv_suppressed = cv * resp_suppression
    rmssd_suppressed = rmssd * resp_suppression
    pnn50_suppressed = pnn50 * resp_suppression

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


def get_ground_truth_episodes(aux_notes):
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


def extract_ai_alarm_episodes(time_axis, active_alerts):
    alarms = []
    in_alarm = False
    start_time = None
    for t, is_alert in zip(time_axis, active_alerts):
        if is_alert and not in_alarm:
            in_alarm = True
            start_time = t
        elif not is_alert and in_alarm:
            in_alarm = False
            alarms.append({'start': start_time, 'end': t})
    if in_alarm: alarms.append({'start': start_time, 'end': time_axis[-1]})
    return alarms


def evaluate_single_patient(args):
    record_path, model_path, scaler_path, plot_out_dir = args
    patient_id = os.path.basename(record_path)

    try:
        if os.path.exists(record_path + '.ecg'):
            ann_rhythm = wfdb.rdann(record_path, 'ecg')
        else:
            ann_rhythm = wfdb.rdann(record_path, 'atr')
        native_fs = getattr(ann_rhythm, 'fs', 128)

        if os.path.exists(record_path + '.qrs'):
            samples_native = wfdb.rdann(record_path, 'qrs').sample
        else:
            samples_native = ann_rhythm.sample

        # 🎯 重采样到128Hz：确保RR间期时序精度与训练数据一致
        if native_fs != 128:
            samples = np.round(samples_native * 128.0 / native_fs).astype(int)
        else:
            samples = samples_native

        atr_samples = ann_rhythm.sample
        # 同样重采样标注位置到128Hz
        if native_fs != 128:
            atr_samples = np.round(atr_samples * 128.0 / native_fs).astype(int)
        atr_notes = getattr(ann_rhythm, 'aux_note', None)

        aux_notes_dense = [''] * len(samples)
        if atr_notes is not None:
            current_rhythm = ''
            atr_idx = 0
            for i, beat_sample in enumerate(samples):
                while atr_idx < len(atr_samples) and beat_sample >= atr_samples[atr_idx]:
                    if atr_notes[atr_idx]: current_rhythm = atr_notes[atr_idx]
                    atr_idx += 1
                if i < len(aux_notes_dense):
                    aux_notes_dense[i] = current_rhythm
    except Exception as e:
        return {'status': 'error', 'msg': f"读取失败: {str(e)}"}

    # RR intervals computed from 128Hz-resampled samples (consistent with training)
    rr_intervals = np.zeros(len(samples))
    if len(samples) > 1:
        rr_intervals[1:] = np.diff(samples) / 128.0 * 1000.0
        rr_intervals[0] = rr_intervals[1]

    # 🌟 从 RR 间期推导累积时间（无需 timestamp）
    cum_time_mins = np.zeros(len(rr_intervals))
    if len(rr_intervals) > 1:
        cum_time_mins[1:] = np.cumsum(rr_intervals[1:]) / 60000.0  # ms → minutes

    # 房颤发作区间（心搏索引 → 分钟）
    gt_episodes_idx = get_ground_truth_episodes(aux_notes_dense)
    gt_episodes = []
    for ep in gt_episodes_idx:
        gt_episodes.append({
            'start': cum_time_mins[ep['start_idx']],
            'end': cum_time_mins[min(ep['end_idx'], len(cum_time_mins) - 1)]
        })

    total_duration_mins = cum_time_mins[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AFibAttentionSeq2Seq(input_dim=29, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scaler = joblib.load(scaler_path)

    history_beats = WINDOW_BEATS * TIME_STEPS  # 3600 beats
    if len(rr_intervals) < history_beats:
        return {'status': 'skipped', 'msg': '太短'}

    time_axis_mins = []
    all_raw_seqs = []
    pip_values = []  # 追踪每窗口原始pip值用于P0异位搏动压制
    current_beat = history_beats
    last_valid_features = [0.0] * 28  # 28 features (see extract_features return)

    # Load waveform for ECG strip visualization only (native rate, no resampling)
    eval_signals = None
    eval_actual_fs = None
    try:
        eval_signals, sig_fields = wfdb.rdsamp(record_path)
        eval_actual_fs = sig_fields['fs']
    except Exception:
        pass

    # Feature cache: pre-compute once per window position (95% overlap = huge waste otherwise)
    feature_cache = {}

    def _get_features(beat_pos):
        key = beat_pos // STEP_BEATS  # nearest step-aligned cache key
        if key not in feature_cache:
            w_end = beat_pos
            w_start = max(0, w_end - WINDOW_BEATS)
            if w_end - w_start >= 50:
                feat = extract_features(rr_intervals[w_start:w_end], aux_notes_dense[w_start:w_end])
                feature_cache[key] = feat if feat is not None else last_valid_features
            else:
                feature_cache[key] = last_valid_features
        return feature_cache[key]

    while current_beat < len(rr_intervals):
        sequence_features = []
        for i in range(TIME_STEPS):
            w_end = current_beat - (TIME_STEPS - 1 - i) * WINDOW_BEATS
            feats = _get_features(w_end)
            sequence_features.append(feats)

        final_seq_feats = []
        for j in range(len(sequence_features)):
            curr = sequence_features[j].copy()
            delta_en = 0.0 if j == 0 else curr[4] - sequence_features[j - 1][4]
            curr.append(delta_en)
            final_seq_feats.append(curr)

        all_raw_seqs.append(final_seq_feats)
        pip_values.append(final_seq_feats[-1][6])  # 最后一个窗口的原始pip值
        time_axis_mins.append(cum_time_mins[current_beat])
        current_beat += STEP_BEATS

    if len(all_raw_seqs) == 0: return {'status': 'skipped', 'msg': '无有效特征'}

    # 🌟 使用全局绝对标尺 (RobustScaler) 归一化
    X_raw_array = np.array(all_raw_seqs)
    X_flat = X_raw_array.reshape(-1, 29)
    X_norm_flat = scaler.transform(X_flat)
    X_norm_array = X_norm_flat.reshape(-1, TIME_STEPS, 29)

    X_tensor = torch.tensor(X_norm_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs, _ = model(X_tensor)
        ai_risk_probs = probs[:, -1].cpu().numpy().tolist()

    # =====================================================================
    # 🎯 自适应多路径报警管线 v3.0 — 个体化阈值校准
    # =====================================================================
    raw_probs = np.array(ai_risk_probs)

    # Layer 1: 底部噪声压制
    suppressed = _suppress_bottom_noise(raw_probs, threshold=0.25, exponent=1.8)

    # Layer 2: EWM平滑
    ewm_smoothed = _ewm_smooth(suppressed, span=5)

    # Layer 3: Savitzky-Golay滤波 — 保留更多信号细节（window=11, poly=2）
    if len(ewm_smoothed) >= 11:
        smoothed_probs = savgol_filter(ewm_smoothed, window_length=11, polyorder=2)
        smoothed_probs = np.clip(smoothed_probs, 0.0, 1.0)
    else:
        smoothed_probs = ewm_smoothed

    # 🎯 多尺度趋势一致性门控（温和版）
    trend_3w = _compute_trend(smoothed_probs, window=3)
    trend_7w = _compute_trend(smoothed_probs, window=7)
    consensus_trend = np.minimum(trend_3w, trend_7w)
    # gate: 共识趋势=0→92%通过（原85%）, 共识趋势≥0.03→100%通过
    trend_gate = np.clip(0.95 + consensus_trend * 3.0, 0.95, 1.0)  # 0.92→0.95, 减少信号衰减
    smoothed_probs = smoothed_probs * trend_gate

    # 🎯 个体化阈值校准：用前30个窗口一次性校准，全程使用固定阈值
    (p1_enter, p2_enter, p3_enter, p3_trend,
     exit_thresh, display_thresh, p1_sustain, p3_sustain) = \
        _compute_adaptive_thresholds(smoothed_probs, calibration_windows=30)

    # 🔄 滚动重校准：每30窗口用最近数据重算阈值，适应患者概率分布漂移
    (rolling_p1, rolling_p2, rolling_p3, rolling_p3t,
     rolling_exit, rolling_disp, rolling_p1s, rolling_p3s) = \
        _compute_rolling_thresholds(smoothed_probs, calibration_windows=30, recalibrate_every=30)

    # 自适应多路径报警 v4.0：滚动重校准阈值 + 个体化持续性 + 异位搏动压制
    active_alerts = _adaptive_alert(smoothed_probs, trend_signal=consensus_trend,
                                    p1_enter=rolling_p1, p2_enter=rolling_p2,
                                    p3_enter=rolling_p3, p3_trend=rolling_p3t,
                                    exit_thresh=rolling_exit,
                                    p1_sustain=rolling_p1s, p3_sustain=rolling_p3s,
                                    pip_values=pip_values)
    ai_alarms = extract_ai_alarm_episodes(time_axis_mins, active_alerts)

    # 个体化阈值（标量）
    alert_threshold = display_thresh

    # =====================================================================
    # 📸 临床级可解释性：波形快照抓取与绘图
    # =====================================================================
    strip_out_dir = os.path.join(plot_out_dir, "Warning_Strips")
    os.makedirs(strip_out_dir, exist_ok=True)

    if eval_signals is not None and eval_signals.shape[1] >= 1:
        physical_signal = eval_signals[:, 0]
        physical_fs = eval_actual_fs
    else:
        physical_signal = None

    if physical_signal is not None and len(ai_alarms) > 0:
        for alarm_idx, alarm in enumerate(ai_alarms):
            trigger_min = alarm['start']

            is_false_alarm = True
            MAX_PREDICT_MINS = 120.0
            for gt in gt_episodes:
                clinical_window_start = gt['start'] - MAX_PREDICT_MINS
                clinical_window_end = gt['end']
                if (alarm['start'] <= clinical_window_end) and (alarm['end'] >= clinical_window_start):
                    is_false_alarm = False
                    break

            if is_false_alarm:
                alarm_type = "False Alarm (Misjudgment)"
                title_color = '#E65100'
                file_prefix = "FalseAlarm"
            else:
                alarm_type = "Valid Early Warning"
                title_color = '#B71C1C'
                file_prefix = "ValidWarn"

            start_sec = max(0, trigger_min * 60.0 - 10.0)
            end_sec = start_sec + 30.0
            start_idx = int(start_sec * physical_fs)
            end_idx = int(end_sec * physical_fs)

            if end_idx > len(physical_signal):
                end_idx = len(physical_signal)

            strip_data = physical_signal[start_idx:end_idx]
            time_strip = np.linspace(start_sec, end_sec, len(strip_data))

            fig_strip, ax_strip = plt.subplots(figsize=(18, 4))
            ax_strip.set_facecolor('#FFF5F5')
            ax_strip.grid(which='major', color='#FFB3B3', linewidth=1.2)
            ax_strip.grid(which='minor', color='#FFE6E6', linewidth=0.5)
            ax_strip.minorticks_on()
            ax_strip.plot(time_strip, strip_data, color='black', linewidth=1.0)
            title_text = f"ECG Strip - Patient {patient_id} | {alarm_type} (Triggered at {trigger_min:.1f} Mins)"
            ax_strip.set_title(title_text, fontsize=14, fontweight='bold', color=title_color)
            ax_strip.set_xlabel("Time (Seconds)", fontsize=12)
            ax_strip.set_ylabel("Amplitude (mV)", fontsize=12)
            ax_strip.set_ylim(-2.5, 2.5)
            plt.tight_layout()

            strip_filename = os.path.join(strip_out_dir,
                                          f"P{patient_id}_{file_prefix}_{alarm_idx + 1}_Min{int(trigger_min)}.png")
            plt.savefig(strip_filename, dpi=300)
            plt.close(fig_strip)

    # ---------------- 绘制患者宏观预测波形大屏 ----------------
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(time_axis_mins, smoothed_probs, 'r-', linewidth=2.5, label='AI Real-time Risk (S-G Filtered)')
    ax.fill_between(time_axis_mins, 0, smoothed_probs, where=active_alerts,
                    color='salmon', alpha=0.3, label='Active Clinical Alert')
    ax.axhline(y=alert_threshold, color='orange', linestyle=':', linewidth=2,
               label=f'Adaptive Threshold ({alert_threshold:.2f})')

    for idx, gt in enumerate(gt_episodes):
        label = 'Expert Annotations: AFib' if idx == 0 else ""
        ax.axvspan(gt['start'], gt['end'], color='black', alpha=0.15, label=label)

    history_required_mins = cum_time_mins[history_beats] if history_beats < len(cum_time_mins) else cum_time_mins[-1]
    ax.legend(loc='upper left', fontsize=11)
    ax.set_title(f"Clinical CDSS Dashboard: Patient {patient_id} "
                 f"[P1={p1_enter:.2f}x{p1_sustain} "
                 f"P2={p2_enter:.2f} "
                 f"P3={p3_enter:.2f}x{p3_sustain} "
                 f"Exit={exit_thresh:.2f}]",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Continuous Monitoring Time (Minutes)", fontsize=13)
    ax.set_ylabel("Risk Probability (P_AFib)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(history_required_mins, total_duration_mins)
    plt.tight_layout()

    plot_filename = os.path.join(plot_out_dir, f"patient_{patient_id}_curve.png")
    plt.savefig(plot_filename, dpi=200)
    plt.close(fig)

    # ==========================================
    # 🎯 核心事件级指标评判逻辑 v3
    # ==========================================
    # 改进:
    #  1. GT合并用临床窗口重叠原则（更合理）
    #  2. 增加时间维度指标（总告警时长、AFib覆盖时长）
    #  3. 一告警一命中 + 最优EWT匹配
    #  4. 区分提前预警(EWT>0)、发作检测(EWT=0)、漏检
    MAX_PREDICT_MINS = 120.0

    # --- 合并 GT: 临床窗口重叠原则 ---
    # 两个GT的120min预警窗口有重叠 → 合并为一个临床事件
    merged_gt = []
    for gt in sorted(gt_episodes, key=lambda g: g['start']):
        gt_win_start = gt['start'] - MAX_PREDICT_MINS
        if merged_gt and (gt_win_start <= merged_gt[-1]['end']):
            merged_gt[-1]['end'] = max(merged_gt[-1]['end'], gt['end'])
        else:
            merged_gt.append({'start': gt['start'], 'end': gt['end']})

    # --- 核心指标容器 ---
    metrics = {
        'patient_id': patient_id,
        'total_duration_hours': total_duration_mins / 60.0,
        'gt_episodes_count': len(merged_gt),
        'gt_episodes_raw_count': len(gt_episodes),
        'caught_episodes_count': 0,
        'early_warning_times': [],    # EWT > 0
        'detection_times': [],        # EWT = 0 (发作中检测)
        'false_alarms_count': 0,
        'total_alarm_mins': 0.0,      # 累计告警时长 (min)
        'total_afib_mins': 0.0,       # 累计 AFib 时长 (min, 合并后GT)
        'afib_covered_mins': 0.0,     # AFib 时段中被告警覆盖的时长 (min)
        'alarm_confidences': [],      # 每个告警的平均置信度
    }

    # --- 统计时间维度 ---
    total_alarm_mins = 0.0
    for alarm in ai_alarms:
        dur = alarm['end'] - alarm['start']
        if dur > 0:
            total_alarm_mins += dur
    metrics['total_alarm_mins'] = total_alarm_mins

    total_afib_mins = sum(gt['end'] - gt['start'] for gt in merged_gt)
    metrics['total_afib_mins'] = total_afib_mins

    # --- 一告警一命中匹配 ---
    used_alarm_indices = set()
    matched_alarm_indices = set()

    for gt in merged_gt:
        best_alarm_idx = None
        best_ewt = -999

        for idx, alarm in enumerate(ai_alarms):
            if idx in used_alarm_indices:
                continue
            clinical_window_start = gt['start'] - MAX_PREDICT_MINS
            clinical_window_end = gt['end']
            if (alarm['start'] <= clinical_window_end) and (alarm['end'] >= clinical_window_start):
                effective_start = max(alarm['start'], clinical_window_start)
                ewt = gt['start'] - effective_start  # 可能为负（告警在发作后）
                if ewt > best_ewt:
                    best_ewt = ewt
                    best_alarm_idx = idx

        if best_alarm_idx is not None:
            metrics['caught_episodes_count'] += 1
            used_alarm_indices.add(best_alarm_idx)
            matched_alarm_indices.add(best_alarm_idx)
            # 分类 EWT
            ewt = max(0, best_ewt)
            if best_ewt > 0:
                metrics['early_warning_times'].append(ewt)
            else:
                metrics['detection_times'].append(0.0)

    # --- AFib 覆盖时长: 告警与GT重叠的总时长 ---
    afib_covered = 0.0
    for gt in merged_gt:
        for idx in matched_alarm_indices:
            alarm = ai_alarms[idx]
            overlap_start = max(alarm['start'], gt['start'])
            overlap_end = min(alarm['end'], gt['end'])
            if overlap_start < overlap_end:
                afib_covered += (overlap_end - overlap_start)
    metrics['afib_covered_mins'] = afib_covered

    # --- 告警置信度 ---
    # 取告警时段内 smoothed_probs 的均值作为告警置信度
    for idx in range(len(ai_alarms)):
        alarm = ai_alarms[idx]
        # 找到对应时间窗口的索引范围
        t_start = alarm['start']
        t_end = alarm['end']
        prob_vals = []
        for ti, t in enumerate(time_axis_mins):
            if t_start <= t <= t_end:
                if ti < len(smoothed_probs):
                    prob_vals.append(smoothed_probs[ti])
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
        is_false_alarm = True
        for gt in merged_gt:
            clinical_window_start = gt['start'] - MAX_PREDICT_MINS
            clinical_window_end = gt['end']
            if (alarm['start'] <= clinical_window_end) and (alarm['end'] >= clinical_window_start):
                is_false_alarm = False
                break
        if is_false_alarm:
            metrics['false_alarms_count'] += 1

    return {'status': 'success', 'metrics': metrics}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=128, help='Use model_s<seed>.pth and scaler_s<seed>.pkl')
    args = parser.parse_args()

    setup_logging(log_file="logs/evaluate.log")

    if args.seed is not None:
        model_path = f"best_afib_model_s{args.seed}.pth"
        scaler_path = f"feature_scaler_s{args.seed}.pkl"
    else:
        model_path = "best_afib_model.pth"
        scaler_path = "feature_scaler.pkl"

    # 测试数据集列表
    test_db_paths = [
        r"D:\LoyaltyWorks\datasets\mit-bih-atrial-fibrillation-database-1.0.0\files",
    ]

    for test_db_path in test_db_paths:
        dataset_name = (
            os.path.basename(os.path.normpath(test_db_path)))
        plot_out_dir = f"evaluation_results_{dataset_name}"
        os.makedirs(plot_out_dir, exist_ok=True)

        records = list(set([os.path.join(test_db_path, f.split(".")[0]) for f in os.listdir(test_db_path) if
                            f.endswith(".atr") or f.endswith(".ecg")]))
        args_list = [(rec, model_path, scaler_path, plot_out_dir) for rec in records]

        results = []
        # Single-process (mp.Pool breaks CUDA on Windows spawn)
        for res in tqdm(map(evaluate_single_patient, args_list), total=len(args_list)):
            results.append(res)

        # 数据聚合与高阶统计 v3
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
            else:
                logger.warning(f"Failed to evaluate patient: {r.get('msg')}")

        patient_reports.sort(key=lambda x: x['patient_id'])

        # ── 事件级指标 ──
        sensitivity = (total_caught_episodes / total_gt_episodes * 100) if total_gt_episodes > 0 else 0
        tp = total_caught_episodes
        fp = total_false_alarms
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        f1_score = (2 * sensitivity * precision) / (sensitivity + precision) if (sensitivity + precision) > 0 else 0

        # ── 时间维度指标 ──
        total_mins = total_hours * 60.0
        alarm_burden = (total_alarm_mins / total_mins * 100) if total_mins > 0 else 0
        afib_time_ratio = (total_afib_mins / total_mins * 100) if total_mins > 0 else 0
        afib_coverage = (total_afib_covered_mins / total_afib_mins * 100) if total_afib_mins > 0 else 0
        nsr_mins = total_mins - total_afib_mins
        nsr_alarm_mins = total_alarm_mins - total_afib_covered_mins
        nsr_far_time = (nsr_alarm_mins / nsr_mins * 100) if nsr_mins > 0 else 0  # % NSR时间被告警

        # ── EWT ──
        n_early = len(all_ewts)
        n_detect = len(all_detections)
        mean_ewt = np.mean(all_ewts) if len(all_ewts) > 0 else 0
        median_ewt = np.median(all_ewts) if len(all_ewts) > 0 else 0
        max_ewt = np.max(all_ewts) if len(all_ewts) > 0 else 0
        early_rate = (n_early / tp * 100) if tp > 0 else 0
        far_per_24h = (total_false_alarms / total_hours * 24) if total_hours > 0 else 0

        # ── 告警置信度 ──
        mean_conf = np.mean(all_confidences) if all_confidences else 0
        median_conf = np.median(all_confidences) if all_confidences else 0

        # ── 逐患者分布（分布统计）──
        patient_sens = []
        for p in patient_reports:
            gt = p['gt_episodes_count']
            if gt > 0:
                patient_sens.append(p['caught_episodes_count'] / gt * 100)
        patient_sens = np.array(patient_sens) if patient_sens else np.array([0])

        # ── 报告生成 v3 ──
        report_lines = []
        report_lines.append("\n" + "=" * 72)
        report_lines.append("  CLINICAL CDSS EVALUATION REPORT v3")
        report_lines.append("=" * 72)
        report_lines.append(f"  Model: {model_path}  |  Dataset: {dataset_name}")
        report_lines.append(f"  Patients: {successful_patients}  |  Duration: {total_hours:.1f}h")
        report_lines.append("")

        report_lines.append("─" * 72)
        report_lines.append("  [1] EVENT-LEVEL METRICS")
        report_lines.append("─" * 72)
        report_lines.append(f"  GT episodes (merged):  {total_gt_episodes}")
        if total_gt_raw != total_gt_episodes:
            report_lines.append(f"  GT episodes (raw):     {total_gt_raw}")
        report_lines.append(f"  Caught:               {tp}")
        report_lines.append(f"  Missed:               {total_gt_episodes - tp}")
        report_lines.append(f"  False alarms:         {fp}")
        report_lines.append(f"  Total alarms:         {tp + fp}")
        report_lines.append(f"  Sensitivity:          {sensitivity:.1f}%")
        report_lines.append(f"  Precision / PPV:      {precision:.1f}%")
        report_lines.append(f"  F1 Score:             {f1_score:.1f}%")
        report_lines.append(f"  FAR (event):          {far_per_24h:.1f} / 24h")
        report_lines.append("")
        report_lines.append(f"  Per-patient Sens:     "
                            f"med={np.median(patient_sens):.0f}% "
                            f"min={np.min(patient_sens):.0f}% "
                            f"p25={np.percentile(patient_sens, 25):.0f}% "
                            f"p75={np.percentile(patient_sens, 75):.0f}% "
                            f"max={np.max(patient_sens):.0f}%")

        report_lines.append("")
        report_lines.append("─" * 72)
        report_lines.append("  [2] TIME-DOMAIN METRICS")
        report_lines.append("─" * 72)
        report_lines.append(f"  AFib time ratio:      {afib_time_ratio:.1f}% of total")
        report_lines.append(f"  AFib coverage:        {afib_coverage:.1f}% of AFib time covered by alarm")
        report_lines.append(f"  Alarm burden:         {alarm_burden:.1f}% of total time")
        report_lines.append(f"  NSR alarm time:       {nsr_far_time:.1f}% of NSR time (time-based FAR)")

        report_lines.append("")
        report_lines.append("─" * 72)
        report_lines.append("  [3] EARLY WARNING QUALITY")
        report_lines.append("─" * 72)
        report_lines.append(f"  Early warning (EWT>0): {n_early}/{tp} ({early_rate:.0f}%)")
        report_lines.append(f"  Detection only (EWT=0): {n_detect}/{tp}")
        if n_early > 0:
            report_lines.append(f"  Mean EWT:             {mean_ewt:.1f} min")
            report_lines.append(f"  Median EWT:           {median_ewt:.1f} min")
            report_lines.append(f"  Max EWT:              {max_ewt:.1f} min")
        report_lines.append(f"  Alarm confidence:     mean={mean_conf:.2f} median={median_conf:.2f}")

        report_lines.append("")
        report_lines.append("─" * 72)
        report_lines.append("  [4] PER-PATIENT DETAIL")
        report_lines.append("─" * 72)
        report_lines.append(f"  {'ID':<14} {'GT':>4} {'Hit':>4} {'Miss':>5} {'FA':>4} {'Sens':>5} {'EWT':>6} {'Alarm%':>6} {'Status'}")
        report_lines.append("  " + "-" * 65)

        for p in patient_reports:
            p_id = p['patient_id']
            p_gt = p['gt_episodes_count']
            p_hit = p['caught_episodes_count']
            p_miss = p_gt - p_hit
            p_fa = p['false_alarms_count']
            p_sens = (p_hit / p_gt * 100) if p_gt > 0 else 0
            p_ewts = p['early_warning_times']
            p_dets = p.get('detection_times', [])
            p_ewt = np.mean(p_ewts + p_dets) if (p_ewts + p_dets) else 0
            p_alarm_pct = (p.get('total_alarm_mins', 0) / (p['total_duration_hours'] * 60) * 100) if p['total_duration_hours'] > 0 else 0
            if p_hit == 0 and p_gt > 0:
                st = "MISS"
            elif p_hit < p_gt:
                st = f"PART"
            elif p_fa > 3:
                st = f"FA={p_fa}"
            else:
                st = "OK"
            report_lines.append(
                f"  {p_id:<14} {p_gt:>4} {p_hit:>4} {p_miss:>5} {p_fa:>4} {p_sens:>4.0f}% {p_ewt:>5.0f}m {p_alarm_pct:>5.1f}% {st}")

        report_lines.append("")
        report_lines.append("=" * 72)

        final_report_text = "\n".join(report_lines)
        logger.info(final_report_text)

        report_filename = os.path.join(plot_out_dir, f"evaluation_report_{dataset_name}.txt")
        try:
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(final_report_text)
            logger.info(f"终极评估报告已生成并导出至: {report_filename}")
        except Exception as e:
            logger.exception(f"报告导出失败: {e}")
