# Lecture 08 — 深度学习基础

> "让计算机自己找出特征" —— 深度学习的这个核心想法，我们从感知机讲起，一路走到从零实现反向传播、PyTorch 训练循环、防过拟合与训练稳定化，直到 GPU 与分布式训练的概念。
> 公式降到最低，靠比喻和亲手运行的代码来理解。

## 这门课你将学到

- 线性模型到底解不了什么问题 (XOR)，神经网络又是怎么解开它的
- 感知机、激活函数、损失函数、梯度下降、反向传播 —— 只用 numpy 亲手把深度学习的零件一个个装起来
- PyTorch 张量、autograd、优化器、nn.Module —— 亲身体会从零实现的那几十行如何在框架里缩成几行
- 基于 Dataset/DataLoader 的标准训练循环，以及怎么读学习曲线
- 用 dropout、weight decay、早停来阻止过拟合
- 用学习率调度 (warmup·cosine)、梯度裁剪、批归一化让训练稳定下来
- GPU 为什么快，混合精度 (fp16/bf16) 和分布式训练 (DDP/FSDP) 是什么 —— 亲手给 8B 模型的训练显存做一份估算

## 先修课程

- **lecture03 — 数据处理** (要能读懂 NumPy 数组运算)
- **lecture06 — 机器学习入门** (特别是 level03 线性回归、level05 逻辑回归、level09 过拟合)

level00~04 只用 numpy，从 level05 起使用 PyTorch (CPU)。所有实战都是小规模的，不需要笔记本电脑以外的设备，在 CPU 上 90 秒内跑完。

## 关卡目录

| 关卡 | 标题 | 难度 |
|---|---|---|
| [level00](level00_why_neural_networks/README.md) | 神经网络为什么会出现 | ⭐ |
| [level01](level01_perceptron_from_scratch/README.md) | 亲手实现感知机 | ⭐⭐ |
| [level02](level02_activation_functions/README.md) | 激活函数 | ⭐⭐ |
| [level03](level03_loss_gradient_descent/README.md) | 损失函数与梯度下降 | ⭐⭐⭐ |
| [level04](level04_backprop_from_scratch/README.md) | 从零实现反向传播 | ⭐⭐⭐⭐ |
| [level05](level05_pytorch_tensors/README.md) | PyTorch 第一步 — 张量 | ⭐⭐⭐ |
| [level06](level06_autograd_optimizers/README.md) | autograd 与优化器 | ⭐⭐⭐ |
| [level07](level07_mlp_classifier/README.md) | 搭建 MLP 分类模型 | ⭐⭐⭐ |
| [level08](level08_training_loops/README.md) | 训练循环、批次与轮次 | ⭐⭐⭐ |
| [level09](level09_regularization_dropout/README.md) | 防止过拟合的技巧 | ⭐⭐⭐⭐ |
| [level10](level10_lr_schedules_stability/README.md) | 学习率与训练稳定化 | ⭐⭐⭐⭐ |
| [level11](level11_gpu_amp_distributed/README.md) | GPU、混合精度与分布式训练概念 | ⭐⭐⭐⭐⭐ |

## 快速路线 (时间紧就只学这 5 关)

1. **level00** — 不知道神经网络为什么必要，后面全都会变成死记硬背。
2. **level04** — 从零实现一次反向传播，深度学习就从"魔法"变成"机器"。
3. **level06** — 对照着看 autograd 是怎么替掉 level04 里那些手工活的。
4. **level08** — 实战代码的骨架，标准训练循环。后面所有课程 (视觉·NLP·LLM) 都会复用它。
5. **level09** — 实战中最常遇到的事故 (过拟合) 的预防办法。

## 这门课在实际工作中的应用场景

- **模型效果评审会上**: 你能自己读懂并说出"loss 曲线长成这样，就是过拟合了"(level08~09)。
- **外包与协作验收时**: 从供应商发来的训练日志里，看出学习率问题和发散的征兆 (level03, level10)。
- **基础设施预算申报时**: 对"训练这个模型需要多少 GPU 显存"给出有依据的估算数字 (level11)。
- **通往后续课程的桥梁**: lecture09 (计算机视觉)、lecture10 (NLP)、lecture12 (LLM) 里所有模型，都跑在这里搭出来的训练循环之上。
