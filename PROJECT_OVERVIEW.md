# Warn_AI 项目学习笔记

## 1. 项目概述

这是一个**临床级房颤（Atrial Fibrillation, AFib）实时监控与提前预警系统**。核心目标：通过对心电信号的连续监测，在房颤发作前提前预警（最长可提前 120 分钟），为临床干预争取宝贵时间。

项目基于 PyTorch 深度学习，使用 LSTM + 因果注意力机制的序列模型，将 RR 间期序列特征映射为房颤风险概率，并配备完整的训练管线、批量评估系统和实时监控 UI。

---

## 2. 项目架构总览

```
Warn_AI/
├── train.py                    # 模型定义 + 训练主循环
├── UI_main.py                  # 实时监控大屏桌面应用 (PySide6)
├── batch_processor_shdb.py     # 数据预处理：房颤数据集 → 训练张量
├── batch_processor_nsr2db.py   # 数据预处理：正常窦性心律数据集 → 训练张量
├── batch_evaluate_cdss.py      # 批量评估管线：逐患者评估 + 报告生成
├── best_afib_model.pth         # 已训练的最佳模型权重 (~1.2GB)
├── feature_scaler.pkl          # 全局 RobustScaler 归一化器
├── mixed_tensors_train/        # 预处理后的混合训练数据
├── evaluation_results_files/   # 对 LTAFDB 数据集的评估结果
└── evaluation_results_mit-bih-normal-sinus-rhythm-database-1.0.0/  # 对 NSR 数据集的评估结果
```

---

## 3. 核心 Python 文件详解

### 3.1 train.py — 模型定义与训练

**模型架构：`AFibAttentionSeq2Seq`**

```
输入: (batch, 6, 11) — 6个时间步，每步11维特征
  │
  ├── Input Projection: Linear(11 → hidden_dim) + ReLU
  │
  ├── LSTM Layer 1 + LayerNorm + Dropout + 残差连接
  ├── LSTM Layer 2 + LayerNorm + Dropout + 残差连接
  │
  ├── 因果注意力 (Causal Attention): 只能关注当前及过去时刻
  │   └── 下三角 mask 确保时间因果性
  │
  ├── 融合层: Concat(LSTM输出, 注意力上下文) → Linear + ReLU
  │
  └── 输出: Sigmoid → (batch, 6) 每步的房颤概率
```

关键参数：
- `input_dim=11`: 10维基础特征 + 1维 delta_entropy
- `hidden_dim=128`: 隐藏层维度
- 总参数量可通过训练日志确认（通常数十万级）

**损失函数：`ClinicalCombinedLoss`**
三重机制的临床组合损失：
1. **加权 BCE**：正样本权重 1.2，缓解类别不平衡
2. **Focal Loss 调制**（γ=2.0）：聚焦难分样本
3. **假阳性惩罚**：负样本预测值超过 0.3 时施加惩罚，权重 2.0

**数据加载：`get_dataloaders_with_scaler`**
- 从 `mixed_tensors_train/` 加载所有 `.pt` 文件
- 使用 `GroupShuffleSplit` 按**患者级别**划分训练/验证集（80/20），防止同一患者数据泄露
- 使用 `RobustScaler` 做全局归一化（对异常值鲁棒）
- 归一化器保存为 `feature_scaler.pkl`

**训练策略：**
- 优化器：AdamW (lr=1e-3, weight_decay=1e-3)
- 学习率调度：CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- 梯度裁剪：max_norm=1.0
- 早停：连续 25 epoch 无 AUROC 改善则停止
- 最大 150 epoch
- 时序加权：靠近当前时刻的标签权重更高（0.4 → 1.0）

---

### 3.2 batch_processor_shdb.py — 房颤数据集预处理

**数据来源：** MIT-BIH LTAFDB（Long-Term AF Database）
**目标：** 将原始 `.atr` 标注文件转换为训练用 `.pt` 张量

**核心逻辑：**
1. 读取每个患者的 RR 间期序列和标注
2. 从标注中提取房颤发作区间 `(AFIB` 标记
3. 滑动窗口采样（窗口=600 心搏，步长=120 心搏）：
   - **红区（正样本）**：窗口与房颤区间重叠 → label = 1.0
   - **高危过渡区**：距下一个房颤 ≤ 3600 心搏 → label = 0.8~1.0（线性退化）
   - **绿区（负样本）**：距房颤 ≥ 7200 心搏 → label = 0.0
4. 使用软标签（soft labels）建模房颤前后的渐变过程
5. 绿区间隔至少 600 心搏以节省存储

**输出格式：** 每个患者一个 `.pt` 文件
```python
{
    'record': '患者ID',
    'X': FloatTensor (N, 6, 11),  # N个序列，6时间步，11维特征
    'Y': FloatTensor (N, 6)        # N个序列的软标签
}
```

---

### 3.3 batch_processor_nsr2db.py — 正常窦性心律数据集预处理

**数据来源：** MIT-BIH Normal Sinus Rhythm RR Interval Database (nsr2db)
**目标：** 生成全负样本（健康人），与房颤数据混合训练，增强模型对正常心律的判别力

**特点：**
- 所有样本 label = 0.0（完全健康）
- 采样步长更大（600 心搏），因健康人数据变化缓慢
- 输出到同一 `mixed_tensors_train/` 目录，由训练脚本统一加载

**混合训练的意义：** 防止模型将任何非房颤的 HRV 正常变异误判为高风险。

---

### 3.4 batch_evaluate_cdss.py — 批量评估系统

这是项目最复杂的文件，实现了完整的临床级评估管线。

**输入：** 房颤数据集路径（如 LTAFDB 或 NSR 数据集）

**处理流程：**

1. **滑动窗口推理**：对每个患者从头到尾滑窗，每窗口提取 10 维特征 + delta_entropy → 11 维输入
2. **三层平滑管线：**
   - Layer 1: 底部噪声压制（< 0.2 的概率做指数压制）
   - Layer 2: 指数加权移动平均（EWM, span=5）
   - Layer 3: Savitzky-Golay 滤波（window=11, polyorder=2）
3. **自适应阈值校准 `_compute_adaptive_thresholds`：**
   - 使用前 30 个窗口的概率值校准个体化阈值
   - 非线性偏移：高基线患者指数级抬升阈值
   - 噪声分级：baseline_std > 0.08 的患者额外抬升
   - 输出 8 个参数：p1_enter, p2_enter, p3_enter, p3_trend, exit_thresh, display_thresh, p1_sustain, p3_sustain
4. **自适应多路径报警系统 `_adaptive_alert`：**
   - **路径1（持续中置信度）**：概率 ≥ p1_enter 连续 p1_sustain 窗口
   - **路径2（高置信度突发）**：概率 ≥ p2_enter 连续 2 窗口
   - **路径3（趋势加速）**：概率 ≥ p3_enter 且趋势斜率 ≥ p3_trend，连续 p3_sustain 窗口
   - 退出条件：概率 < exit_thresh 连续 3 窗口
   - 冷却期：5 窗口
   - 状态机：IDLE → ALARM → COOLDOWN → IDLE
5. **事件级指标计算：**
   - 每个 GT 发作如果在前 120 分钟内被预警 → 成功捕获
   - 不在任何 GT 窗口内的报警 → 假阳性
6. **可视化输出：**
   - 每个患者的风险趋势曲线图
   - 每个报警触发的 ECG 波形快照（分 ValidWarn 和 FalseAlarm）
   - 综合评估报告（TXT）

**特征提取 v2.0 改进：**
- 容差死区 40→50ms：过滤呼吸性窦性心律不齐
- Soft Noise Gate 10→20ms：更严压制低变异时的非线性特征
- 呼吸性窦性心律不齐周期性检测：识别 NSR 特有的呼吸驱动 HRV
- SD2 绝对值归一化
- 代偿中和阈值收紧 0.94/1.06

**10 维基础特征：**
1. CV（变异系数，呼吸周期性压制）
2. MAD（中位数绝对偏差）
3. RMSSD（呼吸周期性压制）
4. pNN50（呼吸周期性压制）
5. 样本熵 (Sample Entropy × gate_weight)
6. DFA α1（去趋势波动分析 × gate_weight）
7. PIP（符号变化比例 × gate_weight）
8. SD1（Poincaré 图短轴 × gate_weight）
9. Poincaré Ratio (SD1/SD2 × gate_weight)
10. LF/HF 比率（频谱分析 × gate_weight）

+ 第 11 维：ΔEntropy（当前窗口与上一窗口的样本熵差）

---

### 3.5 UI_main.py — 实时监控桌面应用

基于 PySide6 + pyqtgraph 构建的临床监控大屏。

**核心组件：**

1. **MedicalDataEngine**：双轨数据引擎
   - 轨道 1：物理波形信号（给 UI 显示）
   - 轨道 2：RR 间期标注（给 AI 分析）
   - 使用 QMutex 保护共享指针
   - 从第 60 分钟开始播放（确保 AI 冷启动有足够数据）

2. **AISentinelThread**：异步 AI 哨兵线程
   - 每秒一次 AI 推理
   - 6 窗口序列（每窗口 10 分钟）= 覆盖过去 60 分钟
   - 加载 7 维模型（与训练模型兼容但 input_dim=7）
   - 发送 `risk_updated` 信号给 UI 线程

3. **ModernCDSSDashboard**：主监控界面
   - ECG 波形实时刷新（20 倍速播放）
   - AI 风险趋势图（Savitzky-Golay 滤波后显示）
   - 持续性状态机报警：
     - 连续 5 帧 > 0.6 → 红色高危报警
     - 单帧 > 0.6 → 黄色注意
     - 低概率 → 绿色正常

**注意：** UI 中模型使用了 `input_dim=7, hidden_dim=8` 的小模型配置，而训练使用的是 `input_dim=11, hidden_dim=128`。这意味着 UI 使用的模型和训练脚本产出的模型配置不匹配，需要留意。

---

## 4. 数据集

| 数据集 | 用途 | 处理器 |
|--------|------|--------|
| MIT-BIH LTAFDB（长期房颤数据库） | 训练正样本 + 评估 | batch_processor_shdb.py |
| MIT-BIH NSR（正常窦性心律数据库） | 训练负样本 + 评估 | batch_processor_nsr2db.py |
| MIT-BIH Normal Sinus Rhythm RR Interval DB | 训练负样本 | batch_processor_nsr2db.py |

---

## 5. 评估结果

### 5.1 LTAFDB 数据集（25 名患者，含房颤）

| 指标 | 数值 |
|------|------|
| 敏感度 (Recall) | 83.61% (250/299) |
| 阳性预测值 (Precision) | 95.79% (250/261) |
| F1-Score | 89.29% |
| 24h 假阳性率 | 1.06 次/24h |
| 平均提前预警时间 | 94.3 分钟 |
| 中位提前预警时间 | 120.0 分钟 |

### 5.2 NSR 数据集（18 名健康人）

| 指标 | 数值 |
|------|------|
| 假阳性 | 33 次 / 386 小时 |
| 24h 假阳性率 | 2.05 次/24h |

健康人数据集有 33 次误报，主要集中在个别基线噪声高的患者（如 16272, 16795, 19093）。

---

## 6. 技术亮点

1. **软标签策略**：不是二分标签（0/1），而是根据距房颤的时间距离给渐变标签，为模型提供更丰富的监-督信号
2. **患者级数据隔离**：使用 GroupShuffleSplit 确保同一患者不会同时出现在训练集和验证集
3. **个体化自适应阈值**：根据每个患者的前 30 个窗口校准专属阈值，解决患者间基线差异问题
4. **三道防线报警体系**：底部噪声压制 → S-G 滤波 → 持续性状态机，层层过滤降低误报
5. **抗早搏特化**：特征提取中对室性早搏(V)、房性早搏(A)等异位心搏进行掩码和代偿中和
6. **呼吸周期性压制**：识别 NSR 特有的呼吸性窦性心律不齐，将其从风险特征中压制
7. **RobustScaler 全局归一化**：使用中位数和四分位距，对 ECG 中常见的异常值不敏感

---

## 7. 存在的问题与注意点

1. **模型维度不匹配**：`train.py` 训练的是 `input_dim=11, hidden_dim=128`，但 `UI_main.py` 加载时使用 `input_dim=7, hidden_dim=8`。需确认实际使用的模型权重与代码是否匹配。

2. **硬编码路径**：多处使用绝对路径（如 `D:\LoyaltyWorks\datasets\...`），迁移到其他环境需手动修改。

3. **单进程评估**：`batch_evaluate_cdss.py` 中 `mp.Pool(1)` 强制单进程，可改为 `mp.cpu_count()` 提高速度。

4. **`best_afib_model.pth` 文件约 1.2GB**：需确认是否误保存了优化器状态或其他非权重数据。正常 128 hidden_dim 的模型权重应该在几十 MB 量级。

5. **UI 模型的 input_dim=7**：意味着 UI 版的 `extract_features` 返回 6 维基础特征 + delta_entropy，与训练版的 10 维基础特征不一致。

---

## 8. 运行方式

```bash
# 1. 数据预处理
python batch_processor_shdb.py    # 处理房颤数据集
python batch_processor_nsr2db.py  # 处理正常窦性心律数据集

# 2. 模型训练
python train.py

# 3. 批量评估
python batch_evaluate_cdss.py

# 4. 启动实时监控 UI
python UI_main.py
```

---

## 9. 依赖库

- **PyTorch**: 深度学习框架
- **wfdb**: WFDB 格式心电数据读取
- **PySide6 + pyqtgraph**: 桌面 UI 与实时波形渲染
- **scipy**: 信号处理（Welch 频谱、Savitzky-Golay 滤波、中值滤波、插值）
- **antropy**: 心率变异性非线性特征（样本熵、DFA）
- **numpy, pandas**: 数据处理
- **scikit-learn**: 数据划分、标准化、评估指标
- **joblib**: RobustScaler 持久化
- **matplotlib**: 评估报告可视化
- **tqdm**: 进度条