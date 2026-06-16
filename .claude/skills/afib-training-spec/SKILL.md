---
name: afib-training-spec
description: 房颤风险预警模型训练规范。在用户提到训练模型、重新训练、retrain、微调模型、数据处理、模型评估、F1优化、降低误判时使用。确保训练流程规范、数据集正确、评估标准明确。
dependencies: python>=3.10, torch>=2.0, wfdb, scipy, antropy, numpy, scikit-learn, joblib, matplotlib, tqdm
---

# 房颤风险预警模型 — 训练规范

## 1. 数据集总览

| 数据集 | 路径 | 患者数 | 用途       |
|--------|------|--------|----------|
| SHDB (Japanese Holter AFib) | `C:\LoyaltyLo\datasets\shdb-af-a-japanese-holter-ecg-database-of-atrial-fibrillation-1.0.1\shdb-af-a-japanese-holter-ecg-database-of-atrial-fibrillation-1.0.1` | 59 | 训练       |
| LTAFDB Train (Long-Term AFib) | `C:\LoyaltyLo\datasets\ltafdb_train` | 55 | 训练       |
| LTAFDB Test (Long-Term AFib) | `C:\LoyaltyLo\datasets\ltafdb_test` | 20 | 训练       |
| NSR RR (Normal Sinus Rhythm) | `C:\LoyaltyLo\datasets\normal-sinus-rhythm-rr-interval-database-1.0.0\normal-sinus-rhythm-rr-interval-database-1.0.0` | 54 | 训练（全负样本） |

总计：**114 名房颤患者（SHDB + LTAFDB train） + 54 名健康人（NSR RR）= 168 名训练患者**，20 名房颤患者（LTAFDB test）作为留出测试集。

注意：NSR RR 数据库使用 `.ecg` 标注文件（非 `.atr`），预处理时需特殊处理。MIT-BIH AFib/Normal Sinus Rhythm 数据库（mit-bih-* 前缀）仅用于批量评估，不用于训练。

## 2. 数据预处理规范

### 2.1 执行顺序（必须严格遵循）

```bash
# 激活虚拟环境
source .venv/Scripts/activate

# 步骤1: 处理所有房颤训练数据集
# 修改 batch_processor_shdb.py 使其处理三个 AFib 数据集：
#   - SHDB (59 patients): .atr annotations
#   - LTAFDB train (55 patients): .atr annotations
#   - 输出到 mixed_tensors_train/
python batch_processor_shdb.py

# 步骤2: 处理正常窦性心律数据集
# 修改 batch_processor_nsr2db.py 使其处理：
#   - NSR RR (54 patients): .ecg annotations
#   - 输出到 mixed_tensors_train/
python batch_processor_nsr2db.py
```

### 2.2 预处理关键参数

```python
WINDOW_BEATS = 600        # 每窗口 600 心搏 (~10 分钟 @60bpm)
STEP_BEATS = 120          # 红区步长 120 心搏 (~2 分钟)
TIME_STEPS = 6            # 序列长度 6 步
GREEN_SPACING_BEATS = 600 # 绿区最小间距
AFIB_PROXIMITY_BEATS = 3600  # 高危过渡区 (~60 分钟)
SAFE_DISTANCE_BEATS = 7200   # 安全区 (~120 分钟)
CONTINUITY_BEATS = 2100      # 序列连续性校验 (~35 分钟)
```

### 2.3 标签策略（训练专用软标签）

```
- 窗口与房颤区间重叠 → label = 1.0（红区/正样本）
- 距下一个房颤 ≤ 3600 心搏 → label = 0.8 ~ 1.0（高危过渡区，线性退化）
- 距房颤 ≥ 7200 心搏 → label = 0.0（绿区/负样本）
```

### 2.4 预处理校验清单

执行完预处理后必须确认：
- [ ] `mixed_tensors_train/` 中 `.pt` 文件数 ≈ 168（患者总数）
- [ ] 各 `.pt` 文件包含 `X` 字段，shape 为 `(N, 6, 15)`（15 维特征）
- [ ] 特征顺序与 `extract_features()` 返回值一致
- [ ] 所有患者级标签分布合理（正负样本比例约 1:3 ~ 1:5）

## 3. 训练规范

### 3.1 执行命令

```bash
python train.py
# 输出：best_afib_model.pth + feature_scaler.pkl
```

### 3.2 必须保证的架构参数

```python
AFibAttentionSeq2Seq(input_dim=15, hidden_dim=128)  # 不可随意修改
```

### 3.3 训练超参数（已调优，修改需谨慎）

- 优化器：AdamW (lr=1e-3, weight_decay=1e-3)
- 学习率调度：CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=1e-5)
- 梯度裁剪：max_norm=1.0
- 早停：连续 25 epoch 无 AUROC 改善则停止
- 最大 150 epoch
- 批大小：128
- 损失函数：`ClinicalCombinedLoss(gamma=2.0, pos_weight=1.2, fp_margin=0.2, fp_weight=4.0)`
- NSR负样本过采样：2x（训练时自动识别全负样本患者并复制2份）

### 3.4 数据划分（铁律：患者级隔离）

```python
GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# 同一患者的所有样本必须在同一 split 中
```

**绝对禁止**随机打乱样本级数据——会导致同一患者的不同时间窗口分布在 train/val 中，造成评估结果虚假偏高。

### 3.5 训练完成后的强制验证

训练完成后必须检查：
- [ ] 模型参数量 = 299,906（`input_dim=15, hidden_dim=128`）
- [ ] `feature_scaler.pkl` 已更新
- [ ] 训练日志中最佳 AUROC > 0.85
- [ ] 备份当前 `best_afib_model.pth` 和 `feature_scaler.pkl`（如 `best_afib_model_YYYYMMDD.pth`）

## 4. 评估规范

### 4.1 评估数据集

训练完成后，使用 `batch_evaluate_cdss.py` 在以下数据集上评估：

| 数据集 | 路径 | 目的 |
|--------|------|------|
| MIT-BIH AFib | `C:\LoyaltyLo\datasets\mit-bih-atrial-fibrillation-database-1.0.0` | 测试敏感度和提前预警能力 |
| MIT-BIH NSR | `C:\LoyaltyLo\datasets\mit-bih-normal-sinus-rhythm-database-1.0.0\mit-bih-normal-sinus-rhythm-database-1.0.0` | 纯假阳性测试 |

### 4.2 目标指标（必须达标）

| 指标 | 目标              | 说明 |
|------|-----------------|------|
| **F1 Score** | ≥ 88%           | 综合平衡度（核心指标） |
| **Sensitivity (Recall)** | ≥ 85%           | 真实发作的捕获率 |
| **Precision (PPV)** | ≥ 90%           | 报警的可靠性 |
| **FAR (NSR)** | ≤ 0.5 次/24h     | 健康人误报率（关键！） |
| **Early Warning Time** | ≥ 60 min (mean) | 平均提前预警时间 |
| **LTAFDB Test FAR** | ≤ 2.0 次/24h     | 留出测试集误报率 |

### 4.3 评估报告解读

评估完成后查看 `evaluation_results_*/evaluation_report_*.txt`，关注：
1. 逐患者误报数——单个患者误报 > 5 次需要排查
2. NSR FAR——健康人误报率是最敏感的指标
3. 漏报患者的特征——是否有特定患者始终无法预警

## 5. 降低误判的调优策略（按优先级排列）

### 策略 A：模型级优化（影响最大）

1. **增加 NSR 训练数据**：如果 NSR RR (54 patients) 全部使用后 FAR 仍高，可加入更多健康人数据
2. **调整 ClinicalCombinedLoss 的 fp_weight**：当前为 2.0，可尝试 3.0-5.0 加强 FP 惩罚
3. **调整 fp_margin**：当前 0.3，降低到 0.2 使 FP 惩罚更早触发
4. **类权重平衡**：确保正负样本比例合理，必要时对负样本降采样

### 策略 B：后处理优化（模型不改）

后处理管线位于 `batch_evaluate_cdss.py` 的 `evaluate_single_patient()` 中：

1. **底部噪声压制**：提高 `threshold`（当前 0.20），范围 0.20-0.30
2. **自适应阈值**：调整 `_compute_adaptive_thresholds` 中的 `DEFAULT_P1`（当前 0.55）
3. **持续性要求**：增加 `DEFAULT_P1_SUSTAIN`（当前 4），范围 4-6
4. **冷却期**：延长 cooldown（当前 5 窗口），范围 5-10

### 策略 C：特征工程优化

1. **呼吸周期性压制**：当前 `resp_suppression` 对 NSR 患者压制不足时，降低 `respiratory_periodicity` 的激活阈值（当前 0.3）
2. **Soft Noise Gate**：提高 gate 激活阈值（当前 20ms），范围 20-30ms

## 6. 从研究文献中提炼的关键最佳实践

基于 2025-2026 年 AFib 检测最新研究：

1. **RR 间期 + 频谱特征融合**：IEEE TAI 2025 研究表明，RR 间期特征与傅里叶幅度谱 (FMS) 融合可实现 99.81% 特异性
2. **更长的分析窗口**：≥50-100 RR 间期（当前 600 心搏已远超此标准，无需修改）
3. **多数据集联合训练**：跨数据集的异构训练是降低误判的关键（当前已覆盖 SHDB + LTAFDB + NSR）
4. **噪声感知训练**：训练时注入基线漂移、肌电干扰等噪声增强可提升模型鲁棒性
5. **受试者独立验证**：分组交叉验证按患者划分而非随机打乱（当前已正确实现）
6. **类别加权损失**：对少数类（正样本）使用加权 BCE 以提升敏感度

## 7. 常见错误与避坑指南

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| `extract_features` 三个副本不同步 | 训练/评估特征分布不一致 | 确保 `batch_evaluate_cdss.py`、`batch_processor_shdb.py`、`batch_processor_nsr2db.py` 中的 `extract_features()` 完全一致 |
| 模型与 scaler 不匹配 | 评估结果随机波动 | 每次训练后 scaler 和 model 必须成对使用 |
| 未按患者级划分数据 | 验证集 AUROC 虚高但测试集崩塌 | 检查 `GroupShuffleSplit` 参数 |
| NSR 数据集路径嵌套 | NSR 评估结果为 0 患者 | NSR 路径需包含子目录：`...\normal-sinus-rhythm-rr-interval-database-1.0.0\normal-sinus-rhythm-rr-interval-database-1.0.0` |
| 训练时未包含 NSR 负样本 | 模型对正常 HRV 变异过度敏感 | 确保 `batch_processor_nsr2db.py` 执行完成且输出到同一 `mixed_tensors_train/` |
| 修改特征不重跑预处理 | 训练崩溃或维度不匹配 | 修改 `extract_features` 后必须重新运行两个 batch_processor |
| LTAFDB test 参与训练 | 留出测试集指标失真 | LTAFDB test 的 20 个患者绝不用于训练或验证 |

## 8. 优化建议存档规则

**每次分析测试结果并给出优化建议后，必须将结论保存为文本文件**，存放于 `analysis_archive/` 目录下。

### 8.1 存档格式

```
analysis_archive/
├── analysis_20260517_01.txt   # 日期 + 序号
├── analysis_20260517_02.txt
└── ...
```

### 8.2 存档内容模板

每个存档文件必须包含以下部分：

```text
=================================================================
 🏥 房颤风险预警模型 — 优化分析报告
=================================================================
日期：YYYY-MM-DD HH:MM
触发原因：（用户提出什么问题 / 发现什么异常）

-----------------------------------------------------------------
 📊 当前评估指标
-----------------------------------------------------------------
（AFib + NSR 数据集的核心指标总览表）

-----------------------------------------------------------------
 🔍 问题诊断
-----------------------------------------------------------------
（逐患者分析，问题归类）

-----------------------------------------------------------------
 💡 优化建议（按优先级排列）
-----------------------------------------------------------------
（每条建议含：策略名称、影响评估、风险等级、参考来源）

-----------------------------------------------------------------
 📎 参考资料
-----------------------------------------------------------------
（文献链接）
=================================================================
```

### 8.3 执行规则

- **每次给出优化建议后必须存档**，不论用户是否明确要求
- 文件名使用当天的分析序号：`analysis_YYYYMMDD_NN.txt`
- 先检查 `analysis_archive/` 目录是否存在，不存在则创建
- 内容使用中文，指标表格对齐
- 存档完成后提醒用户文件路径
