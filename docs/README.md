# ChemCPA Documentation Index
# ChemCPA 文档索引

This directory contains comprehensive documentation on the ChemCPA framework, its architecture, and loss function design.

本目录包含关于 ChemCPA 框架、架构和损失函数设计的完整文档。

---

## 📚 Documentation Files / 文档文件

### 1. [FRAMEWORK_SUMMARY.md](./FRAMEWORK_SUMMARY.md)
**Quick Reference Guide | 快速参考指南**

- **Language**: Bilingual (Chinese + English)
- **Content**: 
  - One-line summary of ChemCPA
  - Core architecture overview
  - Loss function components
  - Key hyperparameters
  - Quick start examples
- **Best for**: 
  - Getting started quickly
  - Understanding the big picture
  - Looking up specific parameters

**适合**:
- 快速入门
- 理解整体架构
- 查询特定参数

---

### 2. [FRAMEWORK_AND_LOSS_DESIGN.md](./FRAMEWORK_AND_LOSS_DESIGN.md)
**Detailed Framework Documentation | 详细框架文档**

- **Language**: Chinese (中文)
- **Content**:
  - In-depth explanation of ChemCPA architecture
  - Detailed mathematical formulations
  - Component-by-component breakdown
  - Training strategies and optimizers
  - Evaluation metrics
  - Hyperparameter explanations
  - Code examples
- **Best for**:
  - Deep understanding of the model
  - Implementing your own version
  - Modifying the architecture
  - Research purposes

**适合**:
- 深入理解模型原理
- 实现自己的版本
- 修改架构
- 研究目的

---

### 3. [LOSS_VISUALIZATION.md](./LOSS_VISUALIZATION.md)
**Loss Function Design Visualization | 损失函数设计可视化**

- **Language**: Bilingual (Chinese + English)
- **Content**:
  - ASCII art diagrams of training flow
  - Visual representation of loss components
  - Step-by-step training process
  - Optimizer configurations
  - Loss evolution during training
  - Debugging tips and common issues
- **Best for**:
  - Visual learners
  - Understanding the training process
  - Debugging training issues
  - Tuning hyperparameters

**适合**:
- 视觉学习者
- 理解训练流程
- 调试训练问题
- 调整超参数

---

### 4. [chemCPA.png](./chemCPA.png)
**Architecture Diagram | 架构图**

- **Type**: PNG Image
- **Content**: Visual overview of the ChemCPA architecture
- **Best for**: Understanding the overall model structure at a glance

**适合**: 一目了然地理解整体模型结构

---

## 🎯 How to Use This Documentation / 如何使用这些文档

### For Beginners / 初学者

1. **Start here**: [FRAMEWORK_SUMMARY.md](./FRAMEWORK_SUMMARY.md)
   - Read the "One-Line Summary" section
   - Look at the "Core Architecture" diagram
   - Review the "Quick Start Example"

2. **Then**: [LOSS_VISUALIZATION.md](./LOSS_VISUALIZATION.md)
   - Study the training flow diagram
   - Understand the loss components visually

3. **Finally**: [FRAMEWORK_AND_LOSS_DESIGN.md](./FRAMEWORK_AND_LOSS_DESIGN.md)
   - Deep dive into specific components you're interested in

### For Researchers / 研究人员

1. **Start**: [FRAMEWORK_AND_LOSS_DESIGN.md](./FRAMEWORK_AND_LOSS_DESIGN.md)
   - Read the mathematical formulations
   - Understand the design principles

2. **Reference**: [LOSS_VISUALIZATION.md](./LOSS_VISUALIZATION.md)
   - Check the training strategy details
   - Review hyperparameter effects

3. **Implement**: Use the code examples from all documents

### For Practitioners / 实践者

1. **Quick lookup**: [FRAMEWORK_SUMMARY.md](./FRAMEWORK_SUMMARY.md)
   - Check hyperparameter defaults
   - Review the quick start code

2. **Debugging**: [LOSS_VISUALIZATION.md](./LOSS_VISUALIZATION.md)
   - Use the "Debugging Tips" section
   - Check common issues and solutions

3. **Deep dive**: [FRAMEWORK_AND_LOSS_DESIGN.md](./FRAMEWORK_AND_LOSS_DESIGN.md)
   - When you need detailed explanations

---

## 🔑 Key Topics Covered / 涵盖的关键主题

### Architecture Components / 架构组件

- ✅ Encoder (编码器)
- ✅ Decoder (解码器)
- ✅ Drug Embeddings (药物嵌入)
- ✅ Dosers (剂量响应模块)
- ✅ Adversaries (对抗网络)
- ✅ Covariate Embeddings (协变量嵌入)

### Loss Functions / 损失函数

- ✅ Reconstruction Loss (重建损失)
  - Gaussian NLL Loss
  - Negative Binomial Loss
  - Gaussian Loss
- ✅ Adversarial Losses (对抗损失)
  - Drug Adversary
  - Covariate Adversary
- ✅ Gradient Penalty (梯度惩罚)

### Training Strategies / 训练策略

- ✅ Alternating Optimization (交替优化)
- ✅ Gradient Reversal (梯度反转)
- ✅ Gradient Clipping (梯度裁剪)
- ✅ Learning Rate Scheduling (学习率调度)

### Evaluation Metrics / 评估指标

- ✅ R² Scores
- ✅ Log-Fold Change R²
- ✅ Disentanglement Scores

---

## 📖 Recommended Reading Order / 推荐阅读顺序

### Path 1: Quick Start / 快速开始
```
FRAMEWORK_SUMMARY.md (15 min)
    ↓
LOSS_VISUALIZATION.md - Training Flow section (10 min)
    ↓
Start coding!
```

### Path 2: Comprehensive Understanding / 全面理解
```
chemCPA.png (2 min)
    ↓
FRAMEWORK_SUMMARY.md (20 min)
    ↓
FRAMEWORK_AND_LOSS_DESIGN.md (60 min)
    ↓
LOSS_VISUALIZATION.md (30 min)
```

### Path 3: Research Deep Dive / 研究深入
```
FRAMEWORK_AND_LOSS_DESIGN.md (Complete, 90 min)
    ↓
Original Paper (NeurIPS 2022)
    ↓
LOSS_VISUALIZATION.md (For implementation details)
    ↓
Source code (chemCPA/model.py, lightning_module.py)
```

---

## 🔗 Related Resources / 相关资源

### External Links / 外部链接

- **Paper**: [NeurIPS 2022](https://openreview.net/pdf?id=vRrFVHxFiXJ)
- **Original Repository**: [theislab/chemCPA](https://github.com/theislab/chemCPA)
- **Talk**: [M2D2 Reading Club](https://m2d2.io/talks/m2d2/predicting-single-cell-perturbation-responses-for-unseen-drugs/)

### Source Code Files / 源代码文件

- `chemCPA/model.py` - Model architecture and loss functions
- `chemCPA/lightning_module.py` - PyTorch Lightning training logic
- `chemCPA/train.py` - Evaluation metrics and utilities
- `chemCPA/data/data.py` - Dataset classes

---

## 💡 Tips / 提示

### For Understanding the Model / 理解模型

1. **Start visual**: Look at diagrams before reading text
2. **Code alongside**: Keep the actual code open while reading
3. **Run examples**: Try the quick start examples yourself
4. **Ask "why"**: Understand the motivation behind each design choice

### For Implementation / 实现

1. **Use defaults first**: Start with default hyperparameters
2. **Monitor losses**: Track all loss components during training
3. **Debug systematically**: Use the debugging section in LOSS_VISUALIZATION.md
4. **Iterate gradually**: Make small changes and test

### For Research / 研究

1. **Read the paper**: Original NeurIPS 2022 paper provides theoretical background
2. **Compare formulations**: Check if implementation matches paper
3. **Ablation studies**: Understand each component's contribution
4. **Explore variations**: Try different loss weights, architectures

---

## 🤝 Contributing / 贡献

If you find errors or have suggestions for improving these documents:

1. Open an issue on GitHub
2. Submit a pull request with corrections
3. Contact the maintainers

如果您发现错误或有改进文档的建议：

1. 在 GitHub 上开启 issue
2. 提交带有修正的 pull request
3. 联系维护者

---

## 📝 Document Metadata / 文档元数据

- **Created**: 2025
- **Language**: Chinese (中文) + English
- **Maintainer**: ChemCPA Community
- **Last Updated**: Check git commit history

---

## Summary Table / 文档总结表

| Document | Language | Length | Content Focus | Best For |
|----------|----------|--------|---------------|----------|
| FRAMEWORK_SUMMARY.md | Bilingual | ~350 lines | Quick reference | Getting started |
| FRAMEWORK_AND_LOSS_DESIGN.md | Chinese | ~750 lines | Detailed explanation | Deep understanding |
| LOSS_VISUALIZATION.md | Bilingual | ~500 lines | Visual diagrams | Training/debugging |
| chemCPA.png | Visual | 1 image | Architecture | Quick overview |

| 文档 | 语言 | 长度 | 内容重点 | 适合人群 |
|------|------|------|----------|----------|
| FRAMEWORK_SUMMARY.md | 双语 | ~350行 | 快速参考 | 入门者 |
| FRAMEWORK_AND_LOSS_DESIGN.md | 中文 | ~750行 | 详细说明 | 深入学习 |
| LOSS_VISUALIZATION.md | 双语 | ~500行 | 可视化图表 | 训练/调试 |
| chemCPA.png | 可视化 | 1张图 | 架构图 | 快速浏览 |

---

**Happy Learning! / 祝学习愉快！** 🚀
