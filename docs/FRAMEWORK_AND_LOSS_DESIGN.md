# ChemCPA 框架与损失函数设计详解

## 目录
1. [整体框架概述](#整体框架概述)
2. [模型架构](#模型架构)
3. [损失函数设计](#损失函数设计)
4. [训练流程](#训练流程)
5. [关键组件说明](#关键组件说明)

---

## 整体框架概述

ChemCPA (Chemical Compound Perturbation Autoencoder) 是一个用于预测单细胞水平药物扰动响应的深度学习框架。该模型能够预测未知药物对细胞基因表达的影响，这对于药物发现和精准医疗具有重要意义。

### 核心思想

ChemCPA 基于**条件变分自编码器 (Conditional VAE)** 的思想，结合了**对抗学习 (Adversarial Learning)** 来实现以下目标：

1. **学习细胞的基础状态**（basal state）的潜在表示
2. **解耦药物效应和细胞类型信息**，使模型能够泛化到新的药物和细胞类型组合
3. **预测给定药物和剂量下的基因表达变化**

### 主要特点

- **药物嵌入学习**: 支持多种分子表示方法（如 SMILES、Morgan 指纹、GNN 等）
- **剂量响应建模**: 通过可学习的剂量响应曲线（dose-response curves）建模药物效应
- **协变量处理**: 支持处理细胞类型、批次等协变量信息
- **对抗性解耦**: 使用对抗训练确保潜在表示中不包含药物和协变量信息

---

## 模型架构

ChemCPA 的核心模型是 `ComPert` 类（位于 `chemCPA/model.py`），包含以下主要组件：

### 1. 编码器 (Encoder)

```python
self.encoder = MLP(
    [num_genes] + [autoencoder_width] * autoencoder_depth + [dim],
    append_layer_width=append_layer_width,
    append_layer_position="first",
)
```

**功能**: 将基因表达数据编码为低维潜在表示（latent representation）

**输入**: 
- `genes`: 基因表达向量，维度为 `[batch_size, num_genes]`

**输出**: 
- `latent_basal`: 基础潜在表示，维度为 `[batch_size, dim]`，其中 `dim` 是潜在空间维度（默认 256）

**架构细节**:
- 多层感知机 (MLP)
- 默认宽度: 512
- 默认深度: 4 层
- 使用 Batch Normalization 和 ReLU 激活

### 2. 解码器 (Decoder)

```python
self.decoder = MLP(
    [dim] + [autoencoder_width] * autoencoder_depth + [num_genes * 2],
    last_layer_act=decoder_activation,
    append_layer_width=(2 * append_layer_width) if append_layer_width else None,
    append_layer_position="last",
)
```

**功能**: 从潜在表示重建基因表达，输出均值和方差

**输入**: 
- `latent_treated`: 经药物扰动后的潜在表示，维度为 `[batch_size, dim]`

**输出**: 
- `gene_reconstructions`: 重建的基因表达，维度为 `[batch_size, num_genes * 2]`
  - 前 `num_genes` 列: 预测的均值 (mean)
  - 后 `num_genes` 列: 预测的方差 (variance)

### 3. 药物嵌入 (Drug Embeddings)

```python
self.drug_embeddings = torch.nn.Embedding(num_drugs, dim)
self.drug_embedding_encoder = MLP(
    [drug_embeddings.embedding_dim] + 
    [embedding_encoder_width] * embedding_encoder_depth + 
    [dim]
)
```

**功能**: 将药物表示映射到潜在空间

**支持的药物表示**:
- 可训练的嵌入向量
- 预训练的化学表示（如 GROVER、ChemVAE、JT-VAE 等）

### 4. 剂量响应模块 (Dosers)

ChemCPA 支持三种剂量响应建模方式：

#### a) Generalized Sigmoid

```python
self.dosers = GeneralizedSigmoid(num_drugs, device, nonlin=doser_type)
```

公式:
- **Sigmoid**: `dose_effect = sigmoid(dose * β + b) - sigmoid(b)`
- **Log-Sigmoid**: `dose_effect = sigmoid(log(1 + dose) * β + b) - sigmoid(b)`

其中 `β` 和 `b` 是可学习参数，每个药物有独立的参数。

#### b) MLP Doser

```python
for _ in range(num_drugs):
    self.dosers.append(
        MLP([1] + [dosers_width] * dosers_depth + [1], batch_norm=False)
    )
```

每个药物有独立的 MLP 网络来建模其剂量响应曲线。

#### c) Amortized Doser

```python
self.dosers = MLP(
    [drug_embeddings.embedding_dim + 1] + 
    [dosers_width] * dosers_depth + 
    [1]
)
```

使用单个 MLP 网络，输入包括药物嵌入和剂量，适用于组合药物场景。

### 5. 对抗网络 (Adversaries)

#### 药物对抗器

```python
self.adversary_drugs = MLP(
    [dim] + [adversary_width] * adversary_depth + [num_drugs]
)
```

**目标**: 从 `latent_basal` 中移除药物信息，确保基础潜在表示不包含药物特征。

#### 协变量对抗器

```python
for num_covariate in num_covariates:
    self.adversary_covariates.append(
        MLP([dim] + [adversary_width] * adversary_depth + [num_covariate])
    )
```

**目标**: 从 `latent_basal` 中移除协变量信息（如细胞类型、批次等）。

### 6. 前向传播流程

```python
def predict(self, genes, drugs_idx, dosages, covariates):
    # 1. 编码：获取基础潜在表示
    latent_basal = self.encoder(genes)
    
    # 2. 计算药物嵌入（经剂量调节）
    drug_embedding = self.compute_drug_embeddings_(drugs_idx, dosages)
    
    # 3. 添加药物效应
    latent_treated = latent_basal + drug_embedding
    
    # 4. 添加协变量效应
    for cov_type, emb_cov in enumerate(self.covariates_embeddings):
        cov_idx = covariates[cov_type].argmax(1)
        latent_treated = latent_treated + emb_cov(cov_idx)
    
    # 5. 解码：重建基因表达
    gene_reconstructions = self.decoder(latent_treated)
    
    # 6. 分离均值和方差
    dim = gene_reconstructions.size(1) // 2
    mean = gene_reconstructions[:, :dim]
    var = F.softplus(gene_reconstructions[:, dim:])
    
    return torch.cat([mean, var], dim=1)
```

**关键点**:
- 潜在空间采用**加性模型**: `latent_treated = latent_basal + drug_embedding + covariate_embedding`
- 这种设计允许模型学习可组合的表示，便于泛化到新的药物-细胞类型组合

---

## 损失函数设计

ChemCPA 的损失函数设计是其核心创新之一，结合了重建损失和对抗损失来实现解耦学习。

### 总体损失函数

训练过程交替优化两组参数：

1. **对抗器参数** (每 `adversary_steps` 步更新一次)
2. **自编码器和剂量器参数** (其他步骤更新)

### 1. 重建损失 (Reconstruction Loss)

#### Gaussian NLL Loss

```python
self.loss_autoencoder = torch.nn.GaussianNLLLoss()
reconstruction_loss = self.loss_autoencoder(input=mean, target=genes, var=var)
```

**数学公式**:

$$
\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{G} \left[ \frac{1}{2} \log(\sigma_{ij}^2) + \frac{(x_{ij} - \mu_{ij})^2}{2\sigma_{ij}^2} \right]
$$

其中:
- $N$: 批次大小
- $G$: 基因数量
- $x_{ij}$: 真实基因表达值
- $\mu_{ij}$: 预测的均值
- $\sigma_{ij}^2$: 预测的方差

**特点**:
- 同时建模均值和方差，能够捕获基因表达的不确定性
- 方差通过 `softplus` 激活确保为正值: `var = F.softplus(raw_var)`

#### 替代损失函数

模型还支持其他损失函数（在 `chemCPA/model.py` 中定义）:

**a) Negative Binomial Loss**

```python
class NBLoss(torch.nn.Module):
    def forward(self, yhat, y, eps=1e-8):
        dim = yhat.size(1) // 2
        mu = yhat[:, :dim]
        theta = yhat[:, dim:]  # inverse dispersion
        
        t1 = torch.lgamma(theta + eps) + torch.lgamma(y + 1.0) - torch.lgamma(y + theta + eps)
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) + \
             (y * (torch.log(theta + eps) - torch.log(mu + eps)))
        return torch.mean(t1 + t2)
```

适用于计数数据（如 scRNA-seq 原始计数）。

**b) Gaussian Loss**

```python
class GaussianLoss(torch.nn.Module):
    def forward(self, yhat, y):
        dim = yhat.size(1) // 2
        mean = yhat[:, :dim]
        variance = yhat[:, dim:]
        
        term1 = variance.log().div(2)
        term2 = (y - mean).pow(2).div(variance.mul(2))
        return (term1 + term2).mean()
```

类似于 GaussianNLLLoss，但实现略有不同。

### 2. 对抗损失 (Adversarial Loss)

#### 药物对抗损失

```python
self.loss_adversary_drugs = torch.nn.BCEWithLogitsLoss()
adversary_drugs_predictions = self.adversary_drugs(latent_basal)
adversary_drugs_loss = self.loss_adversary_drugs(
    adversary_drugs_predictions, 
    multi_hot_targets
)
```

**目的**: 
- 训练对抗器来**最大化**从 `latent_basal` 预测药物的能力
- 训练编码器来**最小化**对抗器的预测能力（通过梯度反转）

**数学表达**:

$$
\mathcal{L}_{\text{adv-drug}} = \text{BCE}(D_{\text{drug}}(h_{\text{basal}}), \text{drug\_labels})
$$

其中 $D_{\text{drug}}$ 是药物对抗器，$h_{\text{basal}}$ 是基础潜在表示。

#### 协变量对抗损失

```python
self.loss_adversary_covariates = nn.ModuleList([
    torch.nn.CrossEntropyLoss() for _ in num_covariates
])

for i, adv in enumerate(self.adversary_covariates):
    pred_cov = adv(latent_basal)
    adversary_covariates_loss += self.loss_adversary_covariates[i](
        pred_cov, 
        covariates[i].argmax(1)
    )
```

**目的**: 确保潜在表示中不包含协变量（如细胞类型）信息。

**数学表达**:

$$
\mathcal{L}_{\text{adv-cov}} = \sum_{k=1}^{K} \text{CE}(D_{\text{cov}_k}(h_{\text{basal}}), \text{cov\_labels}_k)
$$

其中 $K$ 是协变量类型的数量。

### 3. 梯度惩罚 (Gradient Penalty)

```python
def compute_gradient_penalty(out, x):
    grads = torch.autograd.grad(out, x, create_graph=True)[0]
    return (grads ** 2).mean()

penalty_scale = self.model.hparams["penalty_adversary"]  # 默认值: 3
loss_adversary_total = (
    adversary_drugs_loss + 
    adversary_covariates_loss + 
    penalty_scale * gradient_penalty
)
```

**目的**: 
- 稳定对抗训练
- 防止梯度爆炸/消失
- 类似于 WGAN-GP 中的梯度惩罚

**数学表达**:

$$
\mathcal{L}_{\text{penalty}} = \lambda_{\text{penalty}} \cdot \mathbb{E}\left[\|\nabla_{h} D(h)\|^2\right]
$$

### 4. 组合损失函数

#### 对抗器更新 (每 `adversary_steps` 步)

```python
loss_adversary = (
    adversary_drugs_loss + 
    adversary_covariates_loss + 
    penalty_adversary * gradient_penalty
)
```

**目标**: 最大化对抗器的分类能力。

#### 自编码器更新 (其他步骤)

```python
loss_autoencoder = (
    reconstruction_loss - 
    reg_adversary * adversary_drugs_loss - 
    reg_adversary_cov * adversary_covariates_loss
)
```

**关键参数**:
- `reg_adversary`: 药物对抗正则化系数（默认值: 5）
- `reg_adversary_cov`: 协变量对抗正则化系数（默认值: 1.0）

**数学表达**:

$$
\mathcal{L}_{\text{AE}} = \mathcal{L}_{\text{recon}} - \lambda_{\text{drug}} \cdot \mathcal{L}_{\text{adv-drug}} - \lambda_{\text{cov}} \cdot \mathcal{L}_{\text{adv-cov}}
$$

**解释**:
- **减号**: 实现梯度反转效果，编码器尝试最小化对抗器的预测能力
- **平衡**: `reg_adversary` 控制解耦程度与重建质量的权衡

### 损失函数的设计理念

1. **解耦学习**: 通过对抗损失确保 `latent_basal` 只包含细胞的基础状态，不包含药物和协变量信息

2. **可组合性**: 加性模型 (`latent_treated = latent_basal + drug_emb + cov_emb`) 允许独立学习各个因子的效应

3. **不确定性量化**: 预测均值和方差，提供预测的置信度估计

4. **稳定训练**: 梯度惩罚和交替优化策略确保训练稳定性

---

## 训练流程

### 训练步骤 (位于 `chemCPA/lightning_module.py`)

```python
def training_step(self, batch, batch_idx):
    optimizers = self.optimizers()
    optimizer_autoencoder = optimizers[0]
    optimizer_adversaries = optimizers[1]
    optimizer_dosers = optimizers[2]
    
    # 1. 前向传播
    (mean, var), latents = self.forward(batch, return_latent_basal=True)
    latent_basal = latents[0]
    genes = batch[0]
    
    # 2. 计算重建损失
    reconstruction_loss = self.model.loss_autoencoder(mean, genes, var)
    
    # 3. 计算对抗损失
    adversary_drugs_loss = ...
    adversary_covariates_loss = ...
    
    # 4. 交替更新
    if (self.global_step % adversary_steps) == 0:
        # 更新对抗器
        loss_adversary_total = (
            adversary_drugs_loss + 
            adversary_covariates_loss + 
            penalty_adversary * gradient_penalty
        )
        optimizer_adversaries.zero_grad()
        self.manual_backward(loss_adversary_total)
        optimizer_adversaries.step()
    else:
        # 更新自编码器和剂量器
        loss_ae = (
            reconstruction_loss - 
            reg_adversary * adversary_drugs_loss - 
            reg_adversary_cov * adversary_covariates_loss
        )
        optimizer_autoencoder.zero_grad()
        optimizer_dosers.zero_grad()
        self.manual_backward(loss_ae)
        optimizer_autoencoder.step()
        optimizer_dosers.step()
```

### 优化器配置

```python
def configure_optimizers(self):
    # 1. 自编码器优化器（编码器、解码器、协变量嵌入）
    optimizer_autoencoder = torch.optim.Adam(
        autoencoder_parameters,
        lr=1e-3,  # 默认学习率
        weight_decay=1e-6
    )
    
    # 2. 对抗器优化器
    optimizer_adversaries = torch.optim.Adam(
        adversaries_parameters,
        lr=3e-4,
        weight_decay=1e-4
    )
    
    # 3. 剂量器优化器
    optimizer_dosers = torch.optim.Adam(
        dosers_parameters,
        lr=1e-3,
        weight_decay=1e-7
    )
    
    # 学习率调度器（每 45 个 epoch 衰减）
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=45, 
        gamma=0.9
    )
```

### 训练策略

1. **交替优化**: 
   - 对抗器每 `adversary_steps`（默认 3）步更新一次
   - 其他步骤更新自编码器和剂量器

2. **梯度裁剪**: 
   ```python
   self.clip_gradients(optimizer_autoencoder, gradient_clip_val=1, gradient_clip_algorithm="norm")
   ```

3. **学习率衰减**: StepLR，每 45 个 epoch 乘以 0.9

4. **早停**: 基于验证集 R² 分数

---

## 关键组件说明

### 1. 药物嵌入计算

```python
def compute_drug_embeddings_(self, drugs_idx, dosages):
    # 1. 获取药物嵌入
    gathered_embeddings = self.drug_embeddings.weight[drugs_idx]
    
    # 2. 计算剂量调节因子
    scaled_dosages = self.dosers(dosages, drugs_idx)
    
    # 3. 转换嵌入（如果需要）
    if self.drug_embedding_encoder is not None:
        transformed = self.drug_embedding_encoder(gathered_embeddings)
    
    # 4. 应用剂量调节
    scaled_embeddings = transformed * scaled_dosages.unsqueeze(-1)
    
    # 5. 组合药物（用于多药物组合）
    combo_embedding = scaled_embeddings.sum(dim=1)
    
    return combo_embedding
```

### 2. 多任务支持

模型可以扩展支持多任务学习，例如同时预测：
- 基因表达变化
- 细胞表型变化
- 药物毒性

### 3. 迁移学习

通过 `append_layer_width` 参数，模型支持在不同基因集之间进行迁移学习：

```python
# 预训练模型: 10000 个基因
pretrained_model = ComPert(num_genes=10000, ...)

# 微调模型: 5000 个基因
# 添加 henc 和 hdec 层来适配新的基因维度
finetuned_model = ComPert(
    num_genes=10000,  # 保持预训练维度
    append_layer_width=5000,  # 新的基因数量
    ...
)
```

### 4. 评估指标

#### R² Score (位于 `chemCPA/train.py`)

```python
def evaluate_r2(model, dataset, genes_control):
    """
    评估四个指标:
    - R2_mean: 所有基因均值的 R² 分数
    - R2_mean_de: 差异表达基因均值的 R² 分数
    - R2_var: 所有基因方差的 R² 分数
    - R2_var_de: 差异表达基因方差的 R² 分数
    """
    mean_score, var_score, mean_score_de, var_score_de = [], [], [], []
    # ... 评估代码 ...
    return [mean(mean_score), mean(mean_score_de), mean(var_score), mean(var_score_de)]
```

#### Log-Fold Change R²

```python
def evaluate_logfold_r2(autoencoder, ds_treated, ds_ctrl):
    """
    评估预测的 log-fold change 的准确性
    log2(treated / control) 的 R² 分数
    """
    eps = 1e-5
    pred = torch.log2((y_pred + eps) / (y_ctrl + eps))
    true = torch.log2((y_true + eps) / (y_ctrl + eps))
    r2 = compute_r2(true, pred)
```

#### Disentanglement Score

```python
def evaluate_disentanglement(autoencoder, dataloader):
    """
    评估潜在表示的解耦程度
    训练分类器来预测药物和协变量
    分数越低表示解耦越好
    """
    # 训练分类器
    disentanglement_classifier = MLP([...])
    # ... 训练 400 个 epoch ...
    
    # 计算准确率
    acc = torch.sum(pred == labels) / len(labels)
    return acc.item()
```

---

## 超参数

### 默认超参数 (位于 `chemCPA/model.py`)

```python
self.hparams = {
    # 潜在空间维度
    "dim": 256,
    
    # 剂量器
    "dosers_width": 64,
    "dosers_depth": 2,
    "dosers_lr": 1e-3,
    "dosers_wd": 1e-7,
    
    # 自编码器
    "autoencoder_width": 512,
    "autoencoder_depth": 4,
    "autoencoder_lr": 1e-3,
    "autoencoder_wd": 1e-6,
    
    # 对抗器
    "adversary_width": 128,
    "adversary_depth": 3,
    "adversary_lr": 3e-4,
    "adversary_wd": 1e-4,
    "adversary_steps": 3,
    
    # 正则化系数
    "reg_adversary": 5,
    "penalty_adversary": 3,
    
    # 训练
    "batch_size": 128,
    "step_size_lr": 45,
    
    # 嵌入编码器
    "embedding_encoder_width": 512,
    "embedding_encoder_depth": 0,
}
```

### 超参数搜索

模型支持随机超参数搜索：

```python
# seed=0: 使用默认值
model = ComPert(..., seed=0, hparams="")

# seed!=0: 随机采样超参数
model = ComPert(..., seed=42, hparams="")

# 自定义超参数（JSON 格式）
custom_hparams = '{"dim": 512, "autoencoder_width": 1024}'
model = ComPert(..., hparams=custom_hparams)
```

---

## 总结

ChemCPA 通过以下关键设计实现了对未知药物扰动的预测：

1. **解耦表示学习**: 使用对抗训练将细胞基础状态、药物效应和协变量解耦

2. **灵活的损失函数**: 
   - 重建损失确保准确预测基因表达
   - 对抗损失确保表示的解耦性
   - 梯度惩罚稳定训练

3. **剂量响应建模**: 可学习的剂量-响应曲线捕获非线性药物效应

4. **不确定性量化**: 预测均值和方差，提供置信度估计

5. **可扩展架构**: 支持多种药物表示、多任务学习和迁移学习

这种设计使 ChemCPA 能够：
- 预测未见过的药物的效应（通过化学嵌入泛化）
- 预测新的药物-细胞类型组合（通过解耦学习）
- 处理组合药物（通过加性模型）
- 量化预测不确定性（通过方差预测）

---

## 参考文献

- **论文**: [Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution](https://openreview.net/pdf?id=vRrFVHxFiXJ), NeurIPS 2022
- **代码**: [chemCPA GitHub Repository](https://github.com/theislab/chemCPA)
- **主要文件**:
  - `chemCPA/model.py`: 模型架构和损失函数
  - `chemCPA/lightning_module.py`: PyTorch Lightning 训练逻辑
  - `chemCPA/train.py`: 评估指标和训练辅助函数

---

## 附录: 代码示例

### 创建和训练模型

```python
import torch
from chemCPA.model import ComPert
from chemCPA.data.data import load_dataset_splits

# 加载数据
datasets = load_dataset_splits(...)

# 创建模型
model = ComPert(
    num_genes=5000,
    num_drugs=100,
    num_covariates=[10],  # 10 种细胞类型
    device="cuda",
    seed=0,  # 使用默认超参数
    doser_type="logsigm",
    decoder_activation="linear",
    hparams="",
)

# 训练（使用 PyTorch Lightning）
from chemCPA.lightning_module import ChemCPA

lightning_model = ChemCPA(config, dataset_config)
trainer = L.Trainer(max_epochs=100)
trainer.fit(lightning_model, datamodule)
```

### 预测新的药物扰动

```python
# 准备输入
genes_control = torch.randn(10, 5000)  # 10 个对照细胞
drug_idx = torch.tensor([5])  # 药物 ID
dosage = torch.tensor([1.0])  # 剂量
covariate = torch.zeros(10, 10)  # 细胞类型 one-hot
covariate[:, 0] = 1  # 所有细胞都是类型 0

# 预测
with torch.no_grad():
    prediction, _ = model.predict(
        genes=genes_control,
        drugs_idx=drug_idx.expand(10, 1),
        dosages=dosage.expand(10, 1),
        covariates=[covariate],
    )
    
    # 提取均值和方差
    dim = prediction.size(1) // 2
    mean_pred = prediction[:, :dim]
    var_pred = prediction[:, dim:]
    
print(f"Predicted mean shape: {mean_pred.shape}")
print(f"Predicted variance shape: {var_pred.shape}")
```
