# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Clinical-grade AFib early warning system. A PyTorch LSTM + causal attention model ingests RR interval features (sliding windows of 600 heartbeats × 6 time steps) and outputs per-window AFib risk probabilities. A multi-path state-machine alert system converts probabilities into clinical alarms. The system is evaluated on MIT-BIH AFib and NSR databases with event-level metrics (sensitivity, precision, FAR/24h, early warning time).

## Common commands

```bash
# Activate virtual environment
source .venv/Scripts/activate  # or: .venv\Scripts\activate

# Generate training tensors (run both)
python batch_processor_shdb.py     # AFib patients → mixed_tensors_train/
python batch_processor_nsr2db.py   # Healthy NSR patients → mixed_tensors_train/

# Train the model
python train.py                    # saves best_afib_model.pth + feature_scaler.pkl

# Evaluate on both datasets
python batch_evaluate_cdss.py      # outputs to evaluation_results_*/
```

For full training specifications (datasets, evaluation criteria, best practices), invoke: `/afib-training-spec`

## Architecture

### Model (`train.py`)

`AFibAttentionSeq2Seq(input_dim=18, hidden_dim=128)` — ~300K params.
- Input: `(batch, 6, 18)` — 6 time steps, each 18 features (14 HRV + 3 morphology + delta_entropy)
- Architecture: InputProjection → 2×LSTM (with residual + LayerNorm + Dropout) → CausalAttention → Fusion → Sigmoid
- Output: `(batch, 6)` per-step AFib probability

Loss: `ClinicalCombinedLoss` — weighted BCE + Focal Loss (γ=2.0) + FP penalty (margin=0.3, weight=2.0)
Data split: `GroupShuffleSplit` by patient ID (80/20) — no cross-patient leakage
Normalization: `RobustScaler` saved as `feature_scaler.pkl`

### Feature extraction (17 base features)

`extract_features()` returns 14 HRV features and is **duplicated in 4 files**: `batch_processor_shdb.py`, `batch_processor_nsr2db.py`, `batch_evaluate_cdss.py`, `batch_processor_icentia11k.py`. They must stay identical.

`ecg_morphology.py` adds 3 dual-lead QRS morphology features: `qrs_template_corr`, `qrs_outlier_ratio`, `inter_lead_corr_drop`. These are computed at the caller level (sequence assembly) and merged with HRV features to form 17 base features. The 18th feature (delta_entropy) is appended during sequence assembly.

Pipeline within extract_features:
1. Ectopic beat masking (V, A, a, J, S types)
2. SQA gating (physiological range + diff spikes)
3. Median filter (kernel=3) → PAC compensatory neutralization (0.94/1.06)
4. Tolerance dead zone (50ms) — filters respiratory sinus arrhythmia
5. Soft Noise Gate (threshold 20ms) — attenuates nonlinear features when RMSSD is low
6. Respiratory periodicity detection (PSD 0.15–0.40 Hz) → suppresses CV, RMSSD, pNN50

Returns: `[cv_suppressed, mad, rmssd_suppressed, pnn50_suppressed, samp_en, dfa_alpha1, pip, sd1, poincare_ratio, lf_hf_ratio, pip_raw, dfa_raw, bigeminy_corr, bimodality_ratio]`

The 15th feature (delta_entropy) is computed as the difference in samp_en between consecutive windows during sequence assembly.

### Alert pipeline (`batch_evaluate_cdss.py`)

Post-processing chain applied to model output probabilities:

1. **Bottom noise suppression** — `threshold=0.20, exponent=1.8`: probabilities below 0.20 are exponentially suppressed
2. **EWM smoothing** — span=5
3. **Savitzky-Golay filter** — window=11, polyorder=2
4. **Trend signal** — 3-window linear regression slope
5. **Adaptive threshold calibration** — `_compute_adaptive_thresholds()`: uses first 30 windows to compute per-patient thresholds (positive-only shift from baseline 0.15)
6. **Multi-path alert state machine** — `_adaptive_alert()`:
   - Path 1: sustained medium confidence (P ≥ p1_enter for p1_sustain windows)
   - Path 2: high confidence burst (P ≥ p2_enter for 2 windows)
   - Path 3: trend acceleration (P ≥ p3_enter AND slope ≥ p3_trend for p3_sustain windows)
   - State machine: IDLE → ALARM (triggered by any path) → COOLDOWN (5 windows) → IDLE
   - Exit: P < exit_thresh for 3 consecutive windows
7. **Event-level evaluation** — an alarm within 120 min before GT AFib onset counts as caught

### Evaluation metrics

Per-dataset report (`evaluation_results_*/evaluation_report_*.txt`):
- Sensitivity/Recall: caught episodes / total GT episodes
- Precision/PPV: true positives / total alarms
- F1 score
- FAR: false alarms per 24 hours
- Early Warning Time (EWT): minutes from first alarm to AFib onset (capped at 120 min)

## Critical constraints

- **Feature dimension lock**: The model is `input_dim=18`. All copies of `extract_features()` MUST return exactly 14 HRV features. `ecg_morphology.py` adds 3 morphology features. Total = 17 base + delta_entropy = 18D. Adding/removing/changing feature order breaks everything.
- **Model-scaler pairing**: `best_afib_model.pth` and `feature_scaler.pkl` are a matched pair. If you retrain, both are regenerated.
- **Training data must match evaluation features**: If you modify `extract_features` or `ecg_morphology.py`, you must regenerate training tensors and retrain.
- **NSR database**: MIT-BIH Normal Sinus Rhythm Database (18 records, 2-lead 128Hz ECG). Replaces NSR2DB (RR-only).
- **Patient-level split**: Training uses GroupShuffleSplit by patient ID. Do NOT shuffle at sample level — it leaks patient data across train/val.
- **Post-processing is fragile**: Even small changes to threshold logic, smoothing parameters, or noise suppression can swing FAR by 2-3x. Always run both AFib AND NSR evaluation after any change.
