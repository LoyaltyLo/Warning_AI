"""
NSR data augmentation: time-scale RR intervals to simulate different heart rates.
Stretching/compressing RR by ±7-15% preserves HRV structure while simulating
patients with naturally higher/lower resting HR.

Scale factors: 0.85, 0.93, 1.08, 1.15
- 0.85x RR -> HR +17.6% (e.g., 60->71 bpm)
- 0.93x RR -> HR +7.5%  (e.g., 60->65 bpm)
- 1.08x RR -> HR -7.4%  (e.g., 60->56 bpm)
- 1.15x RR -> HR -13.0% (e.g., 60->52 bpm)

Each scale factor produces ~1x the original NSR sequences, total ~4x NSR data.
Handles both MIT-BIH NSR (.atr) and NSR2DB (.ecg) databases.
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import wfdb
import torch
import numpy as np
from tqdm import tqdm

from batch_processor_nsr2db import (
    extract_features, resample_ecg_dataset,
    OUTPUT_DIR, TIME_STEPS, WINDOW_BEATS, STEP_BEATS, CONTINUITY_BEATS
)

SCALE_FACTORS = [0.85, 0.93, 1.08, 1.15]
os.makedirs(OUTPUT_DIR, exist_ok=True)


def augment_record(record_name, ann_type, prefix):
    """Process one NSR record, generating augmented tensors with RR scaling.

    Args:
        record_name: path to record (without extension)
        ann_type: 'atr' for MIT-BIH NSR, 'ecg' for NSR2DB
        prefix: output filename prefix (e.g. 'healthy_mitbih_nsr_aug', 'healthy_nsr2db_aug')
    """
    base_name = os.path.basename(record_name)

    # Check if all outputs already exist
    all_exist = True
    for sf in SCALE_FACTORS:
        save_path = os.path.join(OUTPUT_DIR, f"{prefix}_sf{sf:.2f}_{base_name}.pt")
        if not os.path.exists(save_path):
            all_exist = False
            break
    if all_exist:
        return f"Already done: {base_name}"

    try:
        ann = wfdb.rdann(record_name, ann_type)
        fs = getattr(ann, 'fs', 128)
        samples = ann.sample
    except Exception:
        return f"Failed (read): {base_name}"

    if fs != 128:
        samples, fs = resample_ecg_dataset(samples, original_fs=fs, target_fs=128)

    rr_intervals = np.zeros(len(samples))
    if len(samples) > 1:
        rr_intervals[1:] = np.diff(samples) / fs * 1000.0
        rr_intervals[0] = rr_intervals[1]

    total_beats = len(rr_intervals)
    results = []

    for sf in SCALE_FACTORS:
        save_path = os.path.join(OUTPUT_DIR, f"{prefix}_sf{sf:.2f}_{base_name}.pt")
        if os.path.exists(save_path):
            continue

        all_slices = []
        start_idx = 0
        while start_idx + WINDOW_BEATS <= total_beats:
            rr_window = rr_intervals[start_idx:start_idx + WINDOW_BEATS] * sf
            feats = extract_features(rr_window)
            if feats is not None:
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

        if len(X) > 0:
            tensor_data = {
                'record': f"{prefix}_sf{sf:.2f}_{base_name}",
                'X': torch.tensor(X, dtype=torch.float32),
                'Y': torch.tensor(Y, dtype=torch.float32)
            }
            torch.save(tensor_data, save_path)
            results.append(f"  sf={sf:.2f}: {len(X)} sequences")

    if results:
        return f"Augmented: {base_name}\n" + "\n".join(results)
    return f"No new: {base_name}"


if __name__ == '__main__':
    print(f"Scale factors: {SCALE_FACTORS}")

    # DB1: MIT-BIH NSR
    # db_mitbih = r"D:\LoyaltyWorks\datasets\mit-bih-normal-sinus-rhythm-database-1.0.0\mit-bih-normal-sinus-rhythm-database-1.0.0"
    # mitbih_records = sorted(set([
    #     os.path.join(db_mitbih, f.split(".")[0])
    #     for f in os.listdir(db_mitbih) if f.endswith(".atr")
    # ]))
    # print(f"\nFound {len(mitbih_records)} MIT-BIH NSR records")
    # for rec in tqdm(mitbih_records, total=len(mitbih_records), desc="MIT-BIH NSR Aug"):
    #     result = augment_record(rec, 'atr', 'healthy_mitbih_nsr_aug')
    #     if "Failed" in result:
    #         tqdm.write(f"  {result}")

    # DB2: NSR2DB
    db_nsr2 = r"D:\LoyaltyWorks\datasets\normal-sinus-rhythm-rr-interval-database-1.0.0\normal-sinus-rhythm-rr-interval-database-1.0.0"
    nsr2_records = sorted(set([
        os.path.join(db_nsr2, f.split(".")[0])
        for f in os.listdir(db_nsr2) if f.endswith(".ecg")
    ]))
    print(f"\nFound {len(nsr2_records)} NSR2DB records")
    for rec in tqdm(nsr2_records, total=len(nsr2_records), desc="NSR2DB Aug"):
        result = augment_record(rec, 'ecg', 'healthy_nsr2db_aug')
        if "Failed" in result:
            tqdm.write(f"  {result}")

    print("\nAugmentation complete.")
