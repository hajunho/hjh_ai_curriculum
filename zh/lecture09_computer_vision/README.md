# Lecture 09 — 计算机视觉

> 从计算机把图像读成一张"数字网格"的那一刻开始，到用 CNN 分类图形，
> 再到目标检测、分割、视觉 Transformer — 这是一段用代码复现"眼睛在做什么"的旅程。

## 这门课你将学到

- 图像在计算机内部是如何被表示成数字的 (像素、通道、分辨率)
- 卷积 (Convolution) 为什么是专为图像而生的运算 — 亲手实现一遍来验证
- 如何逐层设计 CNN (卷积神经网络) 的结构，并计算参数量
- 把图形图像分类从训练→评估→错误分析完整跑通的全过程
- 用数据增强和迁移学习，在"数据不够"时把性能拉起来的技术
- 目标检测 (边界框、IoU)、分割、OCR 在产业现场落地的原理
- 视觉 Transformer (ViT) 与 CLIP 式的图文对齐 — 最新视觉 AI 的语法

所有实战都不联网下载，只用仓库内 numpy 亲手画出来的图形图像
(`common/hjh_data.py` 里的 `shape_images`)。没有那些著名数据集，
CNN 的原理照样可以全部证明出来。

## 先修课程

- **lecture03 — 数据处理** (要能读懂 NumPy 数组的索引与切片)
- **lecture06 — 机器学习入门** (训练/评估分离、准确率与误差的概念)
- **lecture08 — 深度学习基础** (PyTorch 张量、训练循环、MLP — 从 level03 开始需要)

level00~02 只用 numpy，所以可以先跳过 lecture08 来尝个鲜。

## 关卡目录

| 关卡 | 标题 | 难度 |
|---|---|---|
| [level00](level00_how_computers_see/README.md) | 计算机是怎么"看"图像的 | ⭐ |
| [level01](level01_pixels_channels/README.md) | 像素、通道与图像运算 | ⭐⭐ |
| [level02](level02_convolution_explained/README.md) | 理解卷积 (Convolution) | ⭐⭐⭐ |
| [level03](level03_building_cnn/README.md) | 搭建 CNN 结构 | ⭐⭐⭐ |
| [level04](level04_pooling_features/README.md) | 池化与特征的层级 | ⭐⭐⭐ |
| [level05](level05_shape_classification/README.md) | 图形图像分类实战 | ⭐⭐⭐ |
| [level06](level06_data_augmentation/README.md) | 数据增强 | ⭐⭐⭐ |
| [level07](level07_transfer_learning/README.md) | 迁移学习 | ⭐⭐⭐⭐ |
| [level08](level08_image_pipeline/README.md) | 图像分类实战流水线 | ⭐⭐⭐⭐ |
| [level09](level09_object_detection/README.md) | 目标检测的原理 | ⭐⭐⭐⭐ |
| [level10](level10_segmentation_ocr_industry/README.md) | 分割、OCR 与工业质检 | ⭐⭐⭐⭐ |
| [level11](level11_vit_multimodal/README.md) | 视觉 Transformer 与多模态 | ⭐⭐⭐⭐⭐ |

## 快速路线 (时间紧就只学这 5 关)

1. **level00** — 图像 = 数字网格。这一句话没用身体理解透，后面的一切看起来都像魔法
2. **level02** — 用 numpy 亲手实现卷积: 把 CNN 的心脏打开来看
3. **level05** — 图形分类完整跑通: 训练→评估→误分类分析的全周期
4. **level07** — 迁移学习: 实务中九成的图像任务都是用这种方式解决的
5. **level09** — 目标检测: 连"什么东西在哪里"都能回答的原理

## 这门课在实际工作中的应用场景

- **制造质量检测**: 用产线摄像头自动检出不良 (划痕、污渍、缺件) (level05, 08, 10)
- **文档自动化**: 从合同、发票扫描件中找出文字区域并转成文本的 OCR 流水线 (level09, 10)
- **零售与物流**: 从货架照片确认商品位置与缺货，仓库内物品计数 (level09)
- **医疗与安全**: 阅片辅助、监控异常情况识别 — 判断"检测结果能信到什么程度"的标准 (level08, 09)
- **外包与方案验收**: 收到一份写着"准确率 99%"的视觉方案书时，能追问数据划分、错误分析、是否做了增强 (level06, 08)
- **理解最新技术**: 在"给我们的服务接个多模态 AI"这种讨论里，知道 ViT、CLIP 是什么并能对话 (level11)

## 学习建议

- 每一关都设计成边运行 `main.py` 边阅读。这门课尤其以 PNG 图片为核心产出，
  所以请务必打开各关 `outputs/` 文件夹里生成的图片看一看。
- 运行方式: 用仓库根目录的虚拟环境执行 `python3 main.py` (环境配置详见根目录的 `SETUP.md`)
- 涉及 torch 训练的关卡 (level03~09, 11) 也全部设计成小规模，在 CPU 上几十秒内跑完。
- 数据全部由代码当场画出，不需要联网。
