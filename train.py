import os
import glob
import torch
import joblib
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import RobustScaler
import numpy as np
import warnings

from logging_config import setup_logging, get_logger

warnings.filterwarnings('ignore')

logger = get_logger(__name__)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"随机种子已设定: {seed}")


# ==========================================
# 🌟 1. 患者级分层数据加载 + RobustScaler
# ==========================================
def get_dataloaders_with_scaler(tensor_dir="./mixed_tensors_train", batch_size=128, seed=42):
    all_files = glob.glob(os.path.join(tensor_dir, "*.pt"))
    if len(all_files) == 0:
        raise ValueError(f"未找到 .pt 文件！请检查 {tensor_dir} 目录。")

    logger.info(f"正在加载 {len(all_files)} 个患者的数据...")

    # 患者级加载：按文件分组，防止同一患者数据泄露
    patient_data = {}
    for file_path in all_files:
        data = torch.load(file_path, weights_only=True)
        record_name = data.get('record', os.path.basename(file_path))
        patient_data[record_name] = {
            'X': data['X'].numpy(),
            'Y': data['Y'].numpy()
        }

    # 拼接所有数据，记录每个样本所属患者
    all_X, all_Y, all_groups = [], [], []
    for record_name, pdata in patient_data.items():
        n_samples = pdata['X'].shape[0]
        all_X.append(pdata['X'])
        all_Y.append(pdata['Y'])
        all_groups.extend([record_name] * n_samples)

    X_tensor = np.concatenate(all_X, axis=0)
    Y_tensor = np.concatenate(all_Y, axis=0)
    groups = np.array(all_groups)

    logger.info(f"数据总览: X_shape: {X_tensor.shape}, Y_shape: {Y_tensor.shape}, "
          f"患者数: {len(patient_data)}")

    # 分层患者级切分：确保AFib患者同时出现在训练集和验证集
    # 判断每个患者类型
    patient_has_risk = {}
    for record_name, pdata in patient_data.items():
        # label >= 0.8 视为风险患者（AFib或灰区）
        patient_has_risk[record_name] = np.any(pdata['Y'] >= 0.8)

    risk_patients = [k for k, v in patient_has_risk.items() if v]
    safe_patients = [k for k, v in patient_has_risk.items() if not v]

    rng = np.random.RandomState(seed)
    rng.shuffle(risk_patients)
    rng.shuffle(safe_patients)

    # 风险患者：至少1个在验证集
    n_risk_val = max(1, int(len(risk_patients) * 0.2))
    risk_val = set(risk_patients[:n_risk_val])
    risk_train = set(risk_patients[n_risk_val:])

    # 安全患者：80/20
    n_safe_val = max(0, int(len(safe_patients) * 0.2))
    safe_val = set(safe_patients[:n_safe_val])
    safe_train = set(safe_patients[n_safe_val:])

    train_patients = risk_train | safe_train
    val_patients = risk_val | safe_val

    # 构建索引
    train_idx = [i for i, g in enumerate(groups) if g in train_patients]
    val_idx = [i for i, g in enumerate(groups) if g in val_patients]

    logger.info(f"风险患者: {len(risk_patients)} (训练{len(risk_train)}, 验证{len(risk_val)})")

    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    Y_train, Y_val = Y_tensor[train_idx], Y_tensor[val_idx]

    logger.info(f"训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本 "
          f"(患者级隔离，零泄露)")

    # 🌟 全局绝对标尺：RobustScaler
    logger.info("正在使用 RobustScaler 计算全局生理绝对基线...")
    scaler = RobustScaler()

    N_train, seq_len, num_features = X_train.shape
    X_train_flat = X_train.reshape(-1, num_features)
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_flat.reshape(N_train, seq_len, num_features)

    N_val = X_val.shape[0]
    X_val_flat = X_val.reshape(-1, num_features)
    X_val_flat = scaler.transform(X_val_flat)
    X_val_scaled = X_val_flat.reshape(N_val, seq_len, num_features)

    joblib.dump(scaler, f"feature_scaler_s{seed}.pkl")
    logger.info("全局生理基线标尺 (feature_scaler.pkl) 已保存！")

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ==========================================
# 🌟 2. 增强版注意力 Seq2Seq 模型
# ==========================================
class AFibAttentionSeq2Seq(nn.Module):
    """
    增强版房颤注意力序列模型 v2.0：
    - 双层 LSTM + 残差连接
    - 因果注意力机制
    - 梯度流友好的融合层
    """
    def __init__(self, input_dim=11, hidden_dim=64):
        super(AFibAttentionSeq2Seq, self).__init__()
        self.hidden_dim = hidden_dim

        # 输入投影层（将11维映射到hidden_dim，方便残差连接）
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 双层LSTM
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.4)

        # 因果注意力
        self.attention_fc = nn.Linear(hidden_dim, 1)

        # 融合层：LSTM输出 + 注意力上下文
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入投影
        proj = torch.relu(self.input_proj(x))

        # LSTM Layer 1 + 残差
        lstm1_out, _ = self.lstm1(proj)
        lstm1_out = self.layer_norm1(lstm1_out)
        lstm1_out = self.dropout(lstm1_out)
        lstm1_out = lstm1_out + proj  # 残差连接

        # LSTM Layer 2 + 残差
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.layer_norm2(lstm2_out)
        lstm2_out = self.dropout(lstm2_out)
        lstm2_out = lstm2_out + lstm1_out  # 残差连接

        # 因果注意力
        seq_len = lstm2_out.size(1)
        attn_scores = self.attention_fc(lstm2_out).transpose(1, 2)
        scores_expanded = attn_scores.expand(-1, seq_len, -1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        scores_masked = scores_expanded.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))
        attn_weights = F.softmax(scores_masked, dim=2)
        context = torch.bmm(attn_weights, lstm2_out)

        # 融合 + 输出
        fused_state = torch.relu(self.fusion_fc(torch.cat([lstm2_out, context], dim=-1)))
        probs = self.sigmoid(self.output_layer(fused_state)).squeeze(-1)
        return probs, attn_weights[:, -1, :]


# ==========================================
# 🌟 3. 组合损失函数：Focal + FP惩罚
# ==========================================
class ClinicalCombinedLoss(nn.Module):
    """
    临床组合损失 v2.0：
    1. Soft Focal Loss — 聚焦难分样本
    2. FP置信度惩罚 — 压制假阳性
    3. 时序加权 — 靠近发作时刻权重更高
    """
    def __init__(self, gamma=2.0, pos_weight=1.2, fp_margin=0.3, fp_weight=2.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.fp_margin = fp_margin
        self.fp_weight = fp_weight

    def forward(self, probs, targets):
        eps = 1e-7
        probs_c = torch.clamp(probs, min=eps, max=1.0 - eps)

        # (1) 加权BCE
        bce = -(self.pos_weight * targets * torch.log(probs_c) +
                (1 - targets) * torch.log(1 - probs_c))

        # (2) Focal调制：对难分样本（|target - pred|大）施加更强梯度
        modulating = torch.abs(targets - probs_c) ** self.gamma
        focal_bce = modulating * bce

        # (3) FP惩罚：负样本(target<0.1)被高置信度预测(>fp_margin)
        neg_mask = (targets < 0.1).float()
        fp_confident = torch.relu(probs_c - self.fp_margin) * neg_mask
        fp_penalty = self.fp_weight * fp_confident

        return (focal_bce + fp_penalty).mean()


# ==========================================
# 4. 临床级评估函数
# ==========================================
def evaluate_clinical_metrics(y_true, y_pred):
    y_true_binary = (y_true > 0.1).astype(int)

    if len(np.unique(y_true_binary)) < 2:
        return 0.0, 0.0, 1.0

    auroc = roc_auc_score(y_true_binary, y_pred)
    auprc = average_precision_score(y_true_binary, y_pred)
    brier = brier_score_loss(y_true_binary, y_pred)
    return auroc, auprc, brier


# ==========================================
# 🌟 5. 训练主循环（含早停 + 学习率预热）
# ==========================================
def train_model(seed=42):
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用计算设备: {device}")

    train_loader, val_loader = get_dataloaders_with_scaler(seed=seed)

    model = AFibAttentionSeq2Seq(input_dim=16, hidden_dim=128).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-3)
    # criterion = ClinicalCombinedLoss(gamma=2.0, pos_weight=3.0, fp_margin=0.3, fp_weight=2.0)
    criterion = ClinicalCombinedLoss(gamma=2.0, pos_weight=3.0, fp_margin=0.30, fp_weight=4.0)

    # P0-1: T_max=50→300, 匹配num_epochs, 避免后250轮LR锁死在1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=300, eta_min=1e-5
    )

    num_epochs = 300
    best_auroc = 0.0
    patience_counter = 0
    patience_limit = 80  # 早停耐心

    for epoch in range(1, num_epochs + 1):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            # 时序加权：靠近当前时刻权重更高
            time_weights = torch.linspace(0.4, 1.0, steps=batch_Y.size(1)).to(device)
            batch_Y_weighted = batch_Y * time_weights

            optimizer.zero_grad()
            probs, _ = model(batch_X)
            loss = criterion(probs, batch_Y_weighted)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                time_weights = torch.linspace(0.4, 1.0, steps=batch_Y.size(1)).to(device)
                batch_Y_weighted = batch_Y * time_weights

                probs, _ = model(batch_X)
                val_loss += criterion(probs, batch_Y_weighted).item() * batch_X.size(0)

                all_preds.extend(probs.view(-1).cpu().numpy())
                all_trues.extend(batch_Y.view(-1).cpu().numpy())

        val_loss /= len(val_loader.dataset)
        auroc, auprc, brier = evaluate_clinical_metrics(np.array(all_trues), np.array(all_preds))

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch:03d}/{num_epochs}] Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | "
              f"Brier: {brier:.4f} | LR: {current_lr:.6f}")

        # 🌟 基于AUROC的早停
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), f"best_afib_model_s{seed}.pth")
            logger.info(f"  --> [模型已保存] 新最佳 AUROC: {best_auroc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(f"早停触发！连续 {patience_limit} 轮无改善，最佳 AUROC: {best_auroc:.4f}")
                break

    logger.info(f"训练完成！最佳 AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=32)
    args = parser.parse_args()
    setup_logging(log_file=f"logs/train_s{args.seed}.log")
    train_model(seed=args.seed)
