# ChemCPA 损失函数设计可视化
# ChemCPA Loss Function Design Visualization

## 训练流程图 / Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ChemCPA 训练流程                                     │
│                         ChemCPA Training Process                            │
└─────────────────────────────────────────────────────────────────────────────┘

输入批次 / Input Batch:
┌──────────────────────────────────────────────────────────────────────────┐
│  genes [batch_size, num_genes]      基因表达 / Gene Expression          │
│  drugs_idx [batch_size, combo_size]  药物索引 / Drug Indices            │
│  dosages [batch_size, combo_size]    剂量 / Dosages                     │
│  covariates [batch_size, num_cov]    协变量 / Covariates (cell type)   │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                          前向传播 / Forward Pass                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
                    ┌───────────────────────────┐
                    │   Encoder (MLP)           │
                    │   [genes → latent_basal]  │
                    └───────────────────────────┘
                                    ↓
                          latent_basal
                    [batch_size, dim=256]
                    基础潜在表示（无药物信息）
                    Basal Latent (drug-free)
                                    ↓
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ↓                           ↓                           ↓
┌──────────────┐          ┌──────────────────┐       ┌─────────────────┐
│Drug Adversary│          │ Drug Embeddings  │       │Cov Adversary    │
│试图预测药物   │          │                  │       │试图预测细胞类型 │
│Tries to      │          │drug_emb[drug_idx]│       │Tries to predict │
│predict drug  │          │     ×            │       │cell type        │
│              │          │doser(dose)       │       │                 │
└──────────────┘          └──────────────────┘       └─────────────────┘
        ↓                           ↓                           ↓
  adv_drug_pred                drug_embedding            adv_cov_pred
        │                           │                           │
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ↓
                          latent_treated = 
                          latent_basal + 
                          drug_embedding + 
                          covariate_embedding
                                    ↓
                    ┌───────────────────────────┐
                    │   Decoder (MLP)           │
                    │   [latent → gene_recon]   │
                    └───────────────────────────┘
                                    ↓
                    gene_reconstructions
                    [batch_size, num_genes*2]
                                    ↓
                    ┌───────────────────────────┐
                    │  Split into mean & var    │
                    └───────────────────────────┘
                          ↓              ↓
                       mean [...]     var [...]
                                    
                                    
┌──────────────────────────────────────────────────────────────────────────┐
│                          损失计算 / Loss Computation                     │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 重建损失 / Reconstruction Loss                                       │
│                                                                         │
│    L_recon = GaussianNLLLoss(mean, genes, var)                         │
│                                                                         │
│    = 1/N Σ [ 1/2 * log(σ²) + (x - μ)² / (2σ²) ]                      │
│                                                                         │
│    目标：准确重建基因表达                                                │
│    Goal: Accurately reconstruct gene expression                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. 药物对抗损失 / Drug Adversarial Loss                                │
│                                                                         │
│    L_adv_drug = BCEWithLogitsLoss(adv_drug_pred, drug_labels)         │
│                                                                         │
│    目标：                                                               │
│    - 对抗器：最大化预测准确率                                            │
│    - 编码器：最小化预测准确率（通过梯度反转）                             │
│                                                                         │
│    Goal:                                                                │
│    - Adversary: maximize prediction accuracy                           │
│    - Encoder: minimize prediction accuracy (gradient reversal)         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. 协变量对抗损失 / Covariate Adversarial Loss                         │
│                                                                         │
│    L_adv_cov = Σ CrossEntropyLoss(adv_cov_pred[i], cov_labels[i])    │
│                                                                         │
│    目标：移除细胞类型等协变量信息                                         │
│    Goal: Remove cell type and other covariate information              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 4. 梯度惩罚 / Gradient Penalty                                          │
│                                                                         │
│    L_penalty = λ_penalty * E[||∇_h D(h)||²]                           │
│                                                                         │
│    其中 λ_penalty = 3 (默认)                                           │
│    目标：稳定对抗训练，防止梯度爆炸                                       │
│    Goal: Stabilize adversarial training, prevent gradient explosion     │
└─────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                    交替优化策略 / Alternating Optimization               │
└──────────────────────────────────────────────────────────────────────────┘

Step % adversary_steps == 0 ? 
│
├─ YES (每 3 步 / Every 3 steps)
│  └─> 更新对抗器 / Update Adversaries:
│      
│      ┌────────────────────────────────────────────────────────────┐
│      │  L_adversary = L_adv_drug +                                │
│      │                L_adv_cov +                                 │
│      │                λ_penalty * L_penalty                       │
│      │                                                            │
│      │  optimizer_adversaries.zero_grad()                         │
│      │  L_adversary.backward()                                    │
│      │  optimizer_adversaries.step()                              │
│      └────────────────────────────────────────────────────────────┘
│
└─ NO (其他步骤 / Other steps)
   └─> 更新自编码器和剂量器 / Update Autoencoder & Dosers:
       
       ┌────────────────────────────────────────────────────────────┐
       │  L_ae = L_recon -                                          │
       │         λ_drug * L_adv_drug -                             │
       │         λ_cov * L_adv_cov                                 │
       │                                                            │
       │  其中：                                                     │
       │  λ_drug = 5 (reg_adversary)                               │
       │  λ_cov = 1.0 (reg_adversary_cov)                          │
       │                                                            │
       │  optimizer_autoencoder.zero_grad()                         │
       │  optimizer_dosers.zero_grad()                              │
       │  L_ae.backward()                                           │
       │  clip_gradients(max_norm=1)                                │
       │  optimizer_autoencoder.step()                              │
       │  optimizer_dosers.step()                                   │
       └────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│                 关键设计思想 / Key Design Principles                     │
└──────────────────────────────────────────────────────────────────────────┘

1. 梯度反转 (Gradient Reversal)
   ─────────────────────────────
   
   编码器优化：L_ae = L_recon - λ * L_adv
                                 ↑
                          负号实现梯度反转
                          Minus sign reverses gradient
   
   结果：编码器学习"欺骗"对抗器，移除药物/协变量信息
   Result: Encoder learns to "fool" adversaries, removing drug/cov info


2. 加性组合 (Additive Composition)
   ────────────────────────────────
   
   latent_treated = latent_basal + drug_emb + cov_emb
   
   优点：
   ✓ 各因子独立学习 / Factors learned independently
   ✓ 支持组合药物 / Supports drug combinations
   ✓ 易于泛化 / Easy to generalize


3. 不确定性建模 (Uncertainty Modeling)
   ──────────────────────────────────────
   
   输出：(mean, variance)
   
   优点：
   ✓ 量化预测置信度 / Quantifies prediction confidence
   ✓ 更好的概率建模 / Better probabilistic modeling
   ✓ 基因特异性不确定性 / Gene-specific uncertainty


4. 交替训练 (Alternating Training)
   ────────────────────────────────
   
   对抗器步骤：maximize L_adv (学习预测)
   编码器步骤：minimize L_adv (学习隐藏)
   
   平衡：通过 adversary_steps=3 控制
   Balance: Controlled by adversary_steps=3
```

## 损失函数权重平衡 / Loss Weight Balancing

```
┌─────────────────────────────────────────────────────────────────────┐
│                    损失函数各组件的相对重要性                        │
│              Relative Importance of Loss Components                 │
└─────────────────────────────────────────────────────────────────────┘

L_ae = L_recon - 5 * L_adv_drug - 1 * L_adv_cov
       ───┬───   ──┬──           ──┬──
          │        │                │
          │        │                └─> 协变量解耦
          │        │                    Covariate disentanglement
          │        │
          │        └─────────────────> 药物解耦（权重更大）
          │                             Drug disentanglement (higher weight)
          │
          └──────────────────────────> 重建质量（基础目标）
                                       Reconstruction quality (primary goal)

权重选择的影响 / Impact of Weight Choices:
─────────────────────────────────────────

reg_adversary = 5 (λ_drug)
├─ 太小 (< 1)：解耦不充分，泛化能力差
│             Too small: Poor disentanglement, bad generalization
│
├─ 合适 (3-7)：平衡重建质量和解耦效果
│             Balanced: Good reconstruction + disentanglement
│
└─ 太大 (> 10)：过度解耦，损失重建质量
              Too large: Over-disentanglement, poor reconstruction


penalty_adversary = 3 (λ_penalty)
├─ 太小：对抗训练不稳定
│        Too small: Unstable adversarial training
│
├─ 合适：梯度稳定，训练顺畅
│        Balanced: Stable gradients, smooth training
│
└─ 太大：限制对抗器学习能力
         Too large: Limits adversary learning
```

## 优化器配置 / Optimizer Configuration

```
┌──────────────────────────────────────────────────────────────────────┐
│                        三个独立的优化器                               │
│                     Three Independent Optimizers                     │
└──────────────────────────────────────────────────────────────────────┘

1. optimizer_autoencoder
   ├─ 参数: encoder + decoder + drug_emb_encoder + cov_embeddings
   ├─ 学习率: 1e-3
   ├─ 权重衰减: 1e-6
   └─ 调度: StepLR(step=45, gamma=0.9)

2. optimizer_adversaries
   ├─ 参数: adversary_drugs + adversary_covariates
   ├─ 学习率: 3e-4 (更小，更稳定)
   ├─ 权重衰减: 1e-4
   └─ 调度: StepLR(step=45, gamma=0.9)

3. optimizer_dosers
   ├─ 参数: dosers (剂量响应曲线)
   ├─ 学习率: 1e-3
   ├─ 权重衰减: 1e-7 (最小，避免过度正则化)
   └─ 调度: StepLR(step=45, gamma=0.9)

更新频率 / Update Frequency:
────────────────────────────

Step 0: adversaries ✓
Step 1: autoencoder + dosers ✓
Step 2: autoencoder + dosers ✓
Step 3: adversaries ✓
Step 4: autoencoder + dosers ✓
Step 5: autoencoder + dosers ✓
...
循环 / Repeats
```

## 损失函数演化 / Loss Evolution During Training

```
训练早期 (Early Training)
─────────────────────────
L_recon: 高 (High) ████████████ ~ 1000
L_adv:   高 (High) ██████████ ~ 0.8
解耦效果：差 (Poor disentanglement)
重建质量：差 (Poor reconstruction)


训练中期 (Mid Training)
───────────────────────
L_recon: 中 (Mid) ████ ~ 100
L_adv:   中 (Mid) ████ ~ 0.5
解耦效果：改善 (Improving)
重建质量：改善 (Improving)


训练后期 (Late Training)
────────────────────────
L_recon: 低 (Low) ██ ~ 50
L_adv:   低 (Low) ██ ~ 0.3
解耦效果：好 (Good disentanglement)
重建质量：好 (Good reconstruction)


理想收敛 (Ideal Convergence)
─────────────────────────────
L_recon: 稳定 (Stable) █ ~ 30-50
L_adv:   稳定接近随机 (Stable near random) ~ 0.2-0.4
解耦效果：优秀 (Excellent)
重建质量：优秀 (Excellent)
```

## 实际训练示例 / Practical Training Example

```python
# 位置: chemCPA/lightning_module.py
# Location: chemCPA/lightning_module.py

def training_step(self, batch, batch_idx):
    """
    单个训练步骤示例
    Example of a single training step
    """
    
    # ═══════════════════════════════════════════════════
    # 步骤 1: 前向传播
    # Step 1: Forward pass
    # ═══════════════════════════════════════════════════
    (mean, var), latents = self.forward(batch, return_latent_basal=True)
    latent_basal = latents[0]
    genes = batch[0]
    
    # ═══════════════════════════════════════════════════
    # 步骤 2: 计算所有损失
    # Step 2: Compute all losses
    # ═══════════════════════════════════════════════════
    
    # 重建损失 / Reconstruction loss
    L_recon = self.model.loss_autoencoder(mean, genes, var)
    # 结果示例 / Example result: 52.3
    
    # 药物对抗损失 / Drug adversarial loss
    adv_drug_pred = self.model.adversary_drugs(latent_basal)
    L_adv_drug = BCEWithLogitsLoss(adv_drug_pred, drug_labels)
    # 结果示例 / Example result: 0.31
    
    # 协变量对抗损失 / Covariate adversarial loss  
    L_adv_cov = 0.0
    for i, adv in enumerate(self.model.adversary_covariates):
        pred = adv(latent_basal)
        L_adv_cov += CrossEntropyLoss(pred, cov_labels[i])
    # 结果示例 / Example result: 0.42
    
    # 梯度惩罚 / Gradient penalty
    L_penalty = compute_gradient_penalty(adv_drug_pred, latent_basal)
    # 结果示例 / Example result: 0.05
    
    # ═══════════════════════════════════════════════════
    # 步骤 3: 根据当前步骤决定更新哪些参数
    # Step 3: Decide which parameters to update
    # ═══════════════════════════════════════════════════
    
    if self.global_step % 3 == 0:
        # ───────────────────────────────────────────────
        # 对抗器更新 / Adversary update
        # ───────────────────────────────────────────────
        L_total = L_adv_drug + L_adv_cov + 3 * L_penalty
        # L_total = 0.31 + 0.42 + 3*0.05 = 0.88
        
        optimizer_adversaries.zero_grad()
        L_total.backward()
        optimizer_adversaries.step()
        
        print(f"[Adversary] Loss = {L_total:.3f}")
        
    else:
        # ───────────────────────────────────────────────
        # 自编码器和剂量器更新 / Autoencoder & doser update
        # ───────────────────────────────────────────────
        L_total = L_recon - 5*L_adv_drug - 1*L_adv_cov
        # L_total = 52.3 - 5*0.31 - 1*0.42 = 50.33
        
        optimizer_autoencoder.zero_grad()
        optimizer_dosers.zero_grad()
        L_total.backward()
        
        # 梯度裁剪 / Gradient clipping
        clip_gradients(optimizer_autoencoder, max_norm=1)
        clip_gradients(optimizer_dosers, max_norm=1)
        
        optimizer_autoencoder.step()
        optimizer_dosers.step()
        
        print(f"[Autoencoder] Loss = {L_total:.3f}")
    
    # ═══════════════════════════════════════════════════
    # 步骤 4: 记录指标
    # Step 4: Log metrics
    # ═══════════════════════════════════════════════════
    self.log("reconstruction_loss", L_recon)  # 52.3
    self.log("adversary_drugs_loss", L_adv_drug)  # 0.31
    self.log("adversary_covariates_loss", L_adv_cov)  # 0.42
```

## 调试技巧 / Debugging Tips

```
常见问题诊断 / Common Issues Diagnosis
─────────────────────────────────────

问题 / Issue: NaN in predictions
├─ 检查 / Check: 方差是否为负
│               Is variance negative?
├─ 解决 / Fix: 使用 softplus(var) 确保正值
│              Use softplus(var) to ensure positive
└─ 代码 / Code: var = F.softplus(gene_recon[:, dim:])


问题 / Issue: 对抗器损失不下降
              Adversary loss not decreasing
├─ 检查 / Check: 对抗器是否有梯度
│               Does adversary have gradients?
├─ 检查 / Check: latent_basal 是否 requires_grad=True
│               Is latent_basal.requires_grad=True?
└─ 解决 / Fix: 确保在对抗器更新时不 detach latent_basal
              Ensure not detaching latent_basal during adversary update


问题 / Issue: 重建损失过高
              Reconstruction loss too high
├─ 检查 / Check: 学习率是否合适
│               Is learning rate appropriate?
├─ 检查 / Check: reg_adversary 是否过大
│               Is reg_adversary too large?
└─ 解决 / Fix: 降低 reg_adversary 或增加 autoencoder_lr
              Reduce reg_adversary or increase autoencoder_lr


问题 / Issue: 泛化能力差
              Poor generalization
├─ 检查 / Check: 对抗损失是否足够低
│               Is adversary loss low enough?
├─ 检查 / Check: 是否过拟合训练集
│               Overfitting on training set?
└─ 解决 / Fix: 增加 reg_adversary，使用更强的正则化
              Increase reg_adversary, use stronger regularization


问题 / Issue: 训练不稳定
              Training unstable
├─ 检查 / Check: 梯度是否爆炸
│               Are gradients exploding?
├─ 检查 / Check: 批次大小是否太小
│               Is batch size too small?
└─ 解决 / Fix: 增加 penalty_adversary，使用梯度裁剪
              Increase penalty_adversary, use gradient clipping
```

---

## 总结 / Summary

ChemCPA 的损失函数设计巧妙地结合了：
ChemCPA's loss function design cleverly combines:

1. **重建损失** → 确保预测准确性
   **Reconstruction loss** → Ensures prediction accuracy

2. **对抗损失 + 梯度反转** → 实现解耦学习
   **Adversarial loss + gradient reversal** → Achieves disentanglement

3. **梯度惩罚** → 稳定训练过程
   **Gradient penalty** → Stabilizes training

4. **交替优化** → 平衡对抗双方
   **Alternating optimization** → Balances adversarial parties

这种设计使模型能够：
This design enables the model to:

✓ 学习可组合的表示 (Learn compositional representations)
✓ 泛化到新的药物-细胞组合 (Generalize to novel drug-cell combinations)
✓ 量化预测不确定性 (Quantify prediction uncertainty)
✓ 稳定高效地训练 (Train stably and efficiently)
