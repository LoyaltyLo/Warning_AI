"""
Dual-lead QRS morphology features for AFib vs ectopic beat discrimination.

Three window-level features aggregated from per-beat QRS segment analysis:
  qrs_template_corr   — mean correlation of each beat's QRS to the median template
  qrs_outlier_ratio   — proportion of beats with template correlation below 0.85
  inter_lead_corr_drop — proportion of beats where inter-lead correlation drops
                          more than 0.1 below the window median

AFib:  consistent narrow QRS → high template_corr, low outlier_ratio, low ilc_drop
Ectopy: variable QRS morphology → lower template_corr, bimodal distribution
NSR:    near-perfect consistency → very high template_corr, near-zero outlier_ratio
"""

import numpy as np
from scipy.signal import butter, filtfilt

# Bandpass filter for QRS isolation
QRS_BP_LOW = 5.0
QRS_BP_HIGH = 30.0
QRS_BP_ORDER = 4

# QRS segment window around R-peak annotation (ms)
QRS_PRE_MS = 150
QRS_POST_MS = 200

# Template correlation threshold for outlier classification
TEMPLATE_CORR_THRESHOLD = 0.85

# Inter-lead correlation drop threshold
INTER_LEAD_DROP_THRESHOLD = 0.1

# Minimum beat counts
MIN_BEATS_FOR_TEMPLATE = 10
MIN_BEATS_FOR_MORPHOLOGY = 30


def preprocess_ecg(signal, fs, bp_low=QRS_BP_LOW, bp_high=QRS_BP_HIGH, order=QRS_BP_ORDER):
    """Bandpass filter (5-30Hz) and z-score normalize a raw ECG lead."""
    if len(signal) < 3 * order:
        return np.zeros_like(signal)
    if np.std(signal) < 1e-8:
        return np.zeros_like(signal)

    nyq = fs / 2.0
    b, a = butter(order, [bp_low / nyq, bp_high / nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    return filtered


def realign_to_peak(signal, beat_sample, fs, search_ms=25):
    """Refine beat position to nearest R-wave peak within search window."""
    search_n = int(search_ms * fs / 1000)
    s = int(beat_sample)
    lo = max(0, s - search_n)
    hi = min(len(signal), s + search_n)
    if hi <= lo:
        return s
    return lo + np.argmax(np.abs(signal[lo:hi]))


def extract_qrs_segments(signal, beat_samples_relative, fs, pre_ms=QRS_PRE_MS,
                         post_ms=QRS_POST_MS):
    """Extract QRS waveform segments around each beat annotation (single lead).

    Beat positions are first realigned to the nearest signal peak within ±25ms
    to compensate for annotation jitter.
    """
    pre_n = int(pre_ms * fs / 1000)
    post_n = int(post_ms * fs / 1000)
    seg_len = pre_n + post_n

    if seg_len <= 0 or len(beat_samples_relative) == 0:
        return np.zeros((0, seg_len), dtype=np.float32)

    segments = np.full((len(beat_samples_relative), seg_len), np.nan, dtype=np.float32)
    for i, s in enumerate(beat_samples_relative):
        s = realign_to_peak(signal, s, fs)
        if s - pre_n >= 0 and s + post_n <= len(signal):
            segments[i] = signal[s - pre_n:s + post_n]

    valid = ~np.isnan(segments).any(axis=1)
    return segments[valid]


def _build_robust_template(segments, keep_fraction=0.7):
    """Iterative template construction: keep only top `keep_fraction` beats."""
    if len(segments) < MIN_BEATS_FOR_TEMPLATE:
        return None

    segs = segments.copy()
    for iteration in range(2):
        template = np.median(segs, axis=0)
        t_std = np.std(template)
        if t_std < 1e-8:
            return None
        template = (template - np.mean(template)) / t_std

        corrs = np.zeros(len(segs))
        for i, s in enumerate(segs):
            s_std = np.std(s)
            if s_std < 1e-8:
                corrs[i] = -1.0
            else:
                s_norm = (s - np.mean(s)) / s_std
                c = np.corrcoef(s_norm, template)[0, 1]
                corrs[i] = 0.0 if np.isnan(c) else c

        n_keep = max(MIN_BEATS_FOR_TEMPLATE, int(len(segs) * keep_fraction))
        keep_idx = np.argsort(corrs)[-n_keep:]
        segs = segs[keep_idx]

    # Return final template
    template = np.median(segs, axis=0)
    template = (template - np.mean(template)) / (np.std(template) + 1e-8)
    return template


def compute_morphology_features(segments_lead1, segments_lead2):
    """Compute per-beat template correlation and inter-lead correlation.

    Uses iterative template refinement: builds an initial template from the
    median of all beats, keeps the top 70% best-matching beats, rebuilds the
    template from those, and repeats. This is robust to misdetected beats and
    noise segments that would otherwise corrupt the template.
    """
    n_beats = len(segments_lead1) if segments_lead1 is not None else 0
    result = {
        'template_corr': np.full(n_beats, np.nan),
        'inter_lead_corr': np.full(n_beats, np.nan),
    }

    if n_beats < MIN_BEATS_FOR_TEMPLATE:
        return result

    has_l2 = segments_lead2 is not None and len(segments_lead2) == n_beats

    # Build robust templates
    template_l1 = _build_robust_template(segments_lead1)
    if template_l1 is None:
        return result

    template_l2 = None
    if has_l2:
        template_l2 = _build_robust_template(segments_lead2)

    # Per-beat metrics against robust templates
    for i in range(n_beats):
        seg1 = segments_lead1[i]
        seg1_std = np.std(seg1)
        if seg1_std < 1e-8:
            corr_l1 = 0.0
        else:
            seg1_norm = (seg1 - np.mean(seg1)) / seg1_std
            corr_l1 = np.corrcoef(seg1_norm, template_l1)[0, 1]
            if np.isnan(corr_l1):
                corr_l1 = 0.0

        if has_l2 and template_l2 is not None:
            seg2 = segments_lead2[i]
            seg2_std = np.std(seg2)
            if seg2_std < 1e-8:
                corr_l2 = 0.0
            else:
                seg2_norm = (seg2 - np.mean(seg2)) / seg2_std
                corr_l2 = np.corrcoef(seg2_norm, template_l2)[0, 1]
                if np.isnan(corr_l2):
                    corr_l2 = 0.0

            result['template_corr'][i] = (corr_l1 + corr_l2) / 2.0
            ilc = np.corrcoef(seg1_norm, seg2_norm)[0, 1]
            result['inter_lead_corr'][i] = 0.0 if np.isnan(ilc) else ilc
        else:
            result['template_corr'][i] = corr_l1

    return result


def aggregate_morphology(per_beat_template_corr, per_beat_inter_lead_corr):
    """Aggregate per-beat metrics into window-level scalar features.

    Uses trimmed mean (excludes bottom 10% of correlations) to prevent
    a small number of noise-corrupted beats from dominating the window mean.
    """
    tc = per_beat_template_corr[~np.isnan(per_beat_template_corr)]

    if len(tc) < MIN_BEATS_FOR_MORPHOLOGY:
        return [0.0, 0.0, 0.0]

    # Trimmed mean: exclude lowest 10% of correlations
    n_trim = max(0, int(len(tc) * 0.1))
    tc_trimmed = np.sort(tc)[n_trim:] if n_trim > 0 else tc

    qrs_template_corr = float(np.clip(np.mean(tc_trimmed), 0.0, 1.0))
    qrs_outlier_ratio = float(np.clip(np.mean(tc < TEMPLATE_CORR_THRESHOLD), 0.0, 1.0))

    ilc = per_beat_inter_lead_corr
    has_ilc = ilc is not None and not np.all(np.isnan(ilc))
    inter_lead_corr_drop = 0.0

    if has_ilc:
        ilc_valid = ilc[~np.isnan(ilc)]
        if len(ilc_valid) >= MIN_BEATS_FOR_MORPHOLOGY:
            median_ilc = np.median(ilc_valid)
            inter_lead_corr_drop = float(np.clip(
                np.mean(ilc_valid < (median_ilc - INTER_LEAD_DROP_THRESHOLD)),
                0.0, 1.0
            ))

    return [qrs_template_corr, qrs_outlier_ratio, inter_lead_corr_drop]


def extract_window_morphology(record_path, beat_samples, window_start_idx,
                              window_end_idx, fs):
    """Main entry point: extract 3 morphology features for one window.

    Args:
        record_path: path to WFDB record without extension
        beat_samples: all beat sample positions (at target fs)
        window_start_idx: start beat index (inclusive)
        window_end_idx: end beat index (exclusive)
        fs: target sampling rate (Hz) — beat_samples are in this rate

    Returns:
        [qrs_template_corr, qrs_outlier_ratio, inter_lead_corr_drop]
        All 0.0 if waveform unavailable or computation fails.
    """
    try:
        import wfdb
        signals, wf_fields = wfdb.rdsamp(record_path)
    except Exception:
        return [0.0, 0.0, 0.0]

    actual_fs = wf_fields['fs']
    n_leads = signals.shape[1]

    window_beats = beat_samples[window_start_idx:window_end_idx]
    if len(window_beats) < MIN_BEATS_FOR_MORPHOLOGY:
        return [0.0, 0.0, 0.0]

    # Convert beat positions from target fs to actual signal fs
    if actual_fs != fs:
        window_beats = np.round(window_beats * actual_fs / fs).astype(int)

    # Determine signal slice: first to last beat with 200ms padding
    pad_samples = int(0.2 * actual_fs)
    sig_start = max(0, int(window_beats[0]) - pad_samples)
    sig_end = min(signals.shape[0], int(window_beats[-1]) + pad_samples)

    # Relative beat positions within the signal slice
    beats_rel = window_beats - sig_start

    # Process each lead
    segments = [None, None]
    for lead in range(min(n_leads, 2)):
        sig_slice = signals[sig_start:sig_end, lead].astype(np.float64)
        sig_preproc = preprocess_ecg(sig_slice, actual_fs)
        segments[lead] = extract_qrs_segments(sig_preproc, beats_rel, actual_fs)

    morpho = compute_morphology_features(segments[0], segments[1])
    return aggregate_morphology(morpho['template_corr'], morpho['inter_lead_corr'])


def extract_window_morphology_from_signal(signals, actual_fs, beat_samples,
                                          window_start_idx, window_end_idx, fs):
    """Same as extract_window_morphology but with pre-loaded signals array.

    Args:
        signals: (n_samples, n_leads) numpy array from wfdb.rdsamp
        actual_fs: sampling rate of the signals
        beat_samples: beat positions at target fs
        window_start_idx, window_end_idx: beat index range
        fs: target sampling rate (beat_samples are in this rate)
    """
    n_leads = signals.shape[1]

    window_beats = beat_samples[window_start_idx:window_end_idx]
    if len(window_beats) < MIN_BEATS_FOR_MORPHOLOGY:
        return [0.0, 0.0, 0.0]

    if actual_fs != fs:
        window_beats = np.round(window_beats * actual_fs / fs).astype(int)

    pad_samples = int(0.2 * actual_fs)
    sig_start = max(0, int(window_beats[0]) - pad_samples)
    sig_end = min(signals.shape[0], int(window_beats[-1]) + pad_samples)

    beats_rel = window_beats - sig_start

    segments = [None, None]
    for lead in range(min(n_leads, 2)):
        sig_slice = signals[sig_start:sig_end, lead].astype(np.float64)
        sig_preproc = preprocess_ecg(sig_slice, actual_fs)
        segments[lead] = extract_qrs_segments(sig_preproc, beats_rel, actual_fs)

    morpho = compute_morphology_features(segments[0], segments[1])
    return aggregate_morphology(morpho['template_corr'], morpho['inter_lead_corr'])
