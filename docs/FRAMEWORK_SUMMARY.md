# ChemCPA Framework Quick Reference

[中文完整文档](./FRAMEWORK_AND_LOSS_DESIGN.md) | [English Version](#english-version)

---

## 中文快速参考

### 一句话总结

ChemCPA 是一个基于条件变分自编码器和对抗学习的深度学习框架，用于预测单细胞水平的药物扰动响应。

### 核心架构

```
输入基因表达 (genes)
      ↓
  编码器 (Encoder)
      ↓
基础潜在表示 (latent_basal) ←── 对抗器移除药物/协变量信息
      ↓
      + 药物嵌入 (drug_embedding × doser(dose))
      + 协变量嵌入 (covariate_embedding)
      ↓
扰动后潜在表示 (latent_treated)
      ↓
  解码器 (Decoder)
      ↓
预测基因表达 (mean, variance)
```

### 损失函数组成

1. **重建损失** (Reconstruction Loss)
   - 类型: Gaussian NLL Loss
   - 公式: `L_recon = GaussianNLL(predicted_mean, true_genes, predicted_var)`
   - 目标: 准确预测基因表达

2. **对抗损失** (Adversarial Loss)
   - 药物对抗: `L_adv_drug = BCE(adversary_drug(latent_basal), drug_labels)`
   - 协变量对抗: `L_adv_cov = CE(adversary_cov(latent_basal), cov_labels)`
   - 目标: 解耦表示，确保 latent_basal 不包含药物/协变量信息

3. **梯度惩罚** (Gradient Penalty)
   - 公式: `L_penalty = λ * E[||∇_h D(h)||²]`
   - 目标: 稳定对抗训练

### 训练策略

```python
if step % adversary_steps == 0:
    # 更新对抗器
    Loss = L_adv_drug + L_adv_cov + λ_penalty * L_penalty
else:
    # 更新自编码器和剂量器
    Loss = L_recon - λ_drug * L_adv_drug - λ_cov * L_adv_cov
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | 256 | 潜在空间维度 |
| `autoencoder_width` | 512 | 自编码器隐藏层宽度 |
| `autoencoder_depth` | 4 | 自编码器层数 |
| `reg_adversary` | 5 | 药物对抗正则化系数 |
| `penalty_adversary` | 3 | 梯度惩罚系数 |
| `adversary_steps` | 3 | 对抗器更新频率 |
| `batch_size` | 128 | 批次大小 |

### 主要功能

- ✅ 预测未见药物的效应（泛化能力）
- ✅ 预测新的药物-细胞类型组合
- ✅ 支持组合药物
- ✅ 量化预测不确定性
- ✅ 支持迁移学习（不同基因集）

---

## English Version

### One-Line Summary

ChemCPA is a deep learning framework based on conditional variational autoencoders and adversarial learning for predicting cellular responses to drug perturbations at single-cell resolution.

### Core Architecture

```
Input Gene Expression (genes)
      ↓
  Encoder
      ↓
Basal Latent (latent_basal) ←── Adversaries remove drug/covariate info
      ↓
      + Drug Embedding (drug_embedding × doser(dose))
      + Covariate Embedding
      ↓
Treated Latent (latent_treated)
      ↓
  Decoder
      ↓
Predicted Gene Expression (mean, variance)
```

### Loss Function Components

1. **Reconstruction Loss**
   - Type: Gaussian NLL Loss
   - Formula: `L_recon = GaussianNLL(predicted_mean, true_genes, predicted_var)`
   - Purpose: Accurately predict gene expression

2. **Adversarial Loss**
   - Drug Adversary: `L_adv_drug = BCE(adversary_drug(latent_basal), drug_labels)`
   - Covariate Adversary: `L_adv_cov = CE(adversary_cov(latent_basal), cov_labels)`
   - Purpose: Disentangle representations, ensure latent_basal is drug/covariate-free

3. **Gradient Penalty**
   - Formula: `L_penalty = λ * E[||∇_h D(h)||²]`
   - Purpose: Stabilize adversarial training

### Training Strategy

```python
if step % adversary_steps == 0:
    # Update adversaries
    Loss = L_adv_drug + L_adv_cov + λ_penalty * L_penalty
else:
    # Update autoencoder and dosers
    Loss = L_recon - λ_drug * L_adv_drug - λ_cov * L_adv_cov
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 256 | Latent space dimension |
| `autoencoder_width` | 512 | Autoencoder hidden layer width |
| `autoencoder_depth` | 4 | Number of autoencoder layers |
| `reg_adversary` | 5 | Drug adversary regularization coefficient |
| `penalty_adversary` | 3 | Gradient penalty coefficient |
| `adversary_steps` | 3 | Adversary update frequency |
| `batch_size` | 128 | Batch size |

### Key Capabilities

- ✅ Predict effects of unseen drugs (generalization)
- ✅ Predict novel drug-cell type combinations
- ✅ Support for combination therapies
- ✅ Quantify prediction uncertainty
- ✅ Support transfer learning (different gene sets)

### Architecture Components

#### 1. Encoder
- **Input**: Gene expression `[batch_size, num_genes]`
- **Output**: Basal latent representation `[batch_size, dim]`
- **Architecture**: MLP with BatchNorm and ReLU

#### 2. Decoder
- **Input**: Treated latent representation `[batch_size, dim]`
- **Output**: Predicted gene expression `[batch_size, num_genes * 2]`
  - First half: mean predictions
  - Second half: variance predictions
- **Architecture**: MLP with optional activation on last layer

#### 3. Drug Embeddings
- **Options**: 
  - Learnable embeddings
  - Pre-trained chemical representations (GROVER, ChemVAE, JT-VAE, etc.)
- **Processing**: 
  - Transform via `drug_embedding_encoder`
  - Scale by dose-response curve (doser)

#### 4. Dosers (Dose-Response Models)
- **Generalized Sigmoid**: `sigmoid(dose * β + b) - sigmoid(b)`
- **MLP**: Separate network per drug
- **Amortized**: Single network for all drugs

#### 5. Adversaries
- **Drug Adversary**: Predicts which drug(s) from `latent_basal`
- **Covariate Adversary**: Predicts cell type/covariates from `latent_basal`
- **Purpose**: Force encoder to remove drug/covariate information

### Forward Pass

```python
# 1. Encode to basal state
latent_basal = encoder(genes)

# 2. Add drug effect
drug_emb = drug_embeddings[drug_idx] * doser(dose, drug_idx)
latent_treated = latent_basal + drug_emb

# 3. Add covariate effect
latent_treated += covariate_embeddings[cov_idx]

# 4. Decode to gene expression
reconstruction = decoder(latent_treated)
mean, var = split(reconstruction)
```

### Loss Function Mathematics

#### Autoencoder Update
$$L_{AE} = L_{recon} - \lambda_{drug} \cdot L_{adv\_drug} - \lambda_{cov} \cdot L_{adv\_cov}$$

Where:
- Negative signs implement gradient reversal
- Encoder learns to "fool" adversaries

#### Adversary Update
$$L_{adversary} = L_{adv\_drug} + L_{adv\_cov} + \lambda_{penalty} \cdot \mathbb{E}[\|\nabla_h D(h)\|^2]$$

Where:
- Adversaries learn to predict drug/covariate
- Gradient penalty stabilizes training

### Evaluation Metrics

1. **R² Score**
   - Mean R²: All genes
   - Mean R² DE: Differentially expressed genes only
   - Variance R²: All genes
   - Variance R² DE: Differentially expressed genes only

2. **Log-Fold Change R²**
   - Measures accuracy of predicted expression changes
   - Formula: `R²(log₂(treated/control)_pred, log₂(treated/control)_true)`

3. **Disentanglement Score**
   - Train classifier on latent representations
   - Lower score = better disentanglement
   - Measures how well drug/covariate info is removed

### Quick Start Example

```python
from chemCPA.model import ComPert
from chemCPA.lightning_module import ChemCPA
import lightning as L

# Create model
model = ComPert(
    num_genes=5000,
    num_drugs=100,
    num_covariates=[10],
    device="cuda",
    seed=0,  # Use default hyperparameters
    doser_type="logsigm",
)

# Predict
with torch.no_grad():
    prediction, _ = model.predict(
        genes=genes_control,
        drugs_idx=drug_idx,
        dosages=dosage,
        covariates=[cell_type],
    )
    mean = prediction[:, :num_genes]
    var = prediction[:, num_genes:]
```

### File Structure

```
chemCPA/
├── model.py              # ComPert model, loss functions
├── lightning_module.py   # PyTorch Lightning training logic
├── train.py             # Evaluation metrics, utilities
├── data/
│   └── data.py          # Dataset classes
└── embedding.py         # Chemical representation utilities

docs/
├── chemCPA.png                        # Architecture diagram
├── FRAMEWORK_AND_LOSS_DESIGN.md      # Detailed Chinese documentation
└── FRAMEWORK_SUMMARY.md              # This quick reference
```

### References

- **Paper**: [Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution](https://openreview.net/pdf?id=vRrFVHxFiXJ), NeurIPS 2022
- **Code**: https://github.com/theislab/chemCPA
- **Talk**: https://m2d2.io/talks/m2d2/predicting-single-cell-perturbation-responses-for-unseen-drugs/

---

## Key Insights

### Why Adversarial Training?

The adversarial training ensures that `latent_basal` only captures the cell's baseline state, not the drug or cell type. This enables:
1. **Compositionality**: Effects can be added independently
2. **Generalization**: Works for unseen drug-cell combinations
3. **Interpretability**: Clear separation of factors

### Why Predict Variance?

Predicting both mean and variance allows:
1. **Uncertainty quantification**: Know when the model is uncertain
2. **Better likelihood**: Gaussian NLL Loss is more principled than MSE
3. **Gene-specific uncertainty**: Some genes are harder to predict

### Why Additive Model?

The additive formulation `latent_treated = latent_basal + drug_emb + cov_emb`:
1. **Enables composition**: Can combine multiple drugs
2. **Simplifies learning**: Linear assumption in latent space
3. **Supports transfer**: Learn components independently

---

**For detailed explanations with mathematical derivations and code examples, see [FRAMEWORK_AND_LOSS_DESIGN.md](./FRAMEWORK_AND_LOSS_DESIGN.md) (Chinese).**
