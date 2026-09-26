# Lecture 09 — Computer Vision

> From the moment a computer reads an image as a "grid of numbers", through classifying shapes with a CNN,
> all the way to object detection, segmentation, and Vision Transformers — a journey that recreates in code what your eyes do.

## What You Will Learn

- How an image is represented as numbers inside a computer (pixels, channels, resolution)
- Why convolution is an operation purpose-built for images — verified by implementing it yourself
- How to design a CNN (convolutional neural network) layer by layer and count its parameters
- The full cycle of shape-image classification: training → evaluation → error analysis
- Techniques for boosting performance "when data is scarce": data augmentation and transfer learning
- How object detection (bounding boxes, IoU), segmentation, and OCR work in industrial settings
- Vision Transformers (ViT) and CLIP-style image–text alignment — the grammar of modern vision AI

Every exercise runs without any internet downloads, using only shape images drawn on the spot
with numpy inside this repository (`shape_images` in `common/hjh_data.py`). You can prove
every principle of CNNs without a single famous dataset.

## Prerequisites

- **lecture03 — Working with Data** (you should be able to read NumPy array indexing and slicing)
- **lecture06 — Introduction to Machine Learning** (train/test splits, the concepts of accuracy and error)
- **lecture08 — Deep Learning Foundations** (PyTorch tensors, training loops, MLPs — needed from level03 onward)

level00–02 use only numpy, so feel free to sample them first even if you have skipped lecture08.

## Level Index

| Level | Title | Difficulty |
|---|---|---|
| [level00](level00_how_computers_see/README.md) | How Computers See Images | ⭐ |
| [level01](level01_pixels_channels/README.md) | Pixels, Channels, and Image Operations | ⭐⭐ |
| [level02](level02_convolution_explained/README.md) | Understanding Convolution | ⭐⭐⭐ |
| [level03](level03_building_cnn/README.md) | Building a CNN Architecture | ⭐⭐⭐ |
| [level04](level04_pooling_features/README.md) | Pooling and the Hierarchy of Features | ⭐⭐⭐ |
| [level05](level05_shape_classification/README.md) | Hands-On: Classifying Shape Images | ⭐⭐⭐ |
| [level06](level06_data_augmentation/README.md) | Data Augmentation | ⭐⭐⭐ |
| [level07](level07_transfer_learning/README.md) | Transfer Learning | ⭐⭐⭐⭐ |
| [level08](level08_image_pipeline/README.md) | A Real-World Image Classification Pipeline | ⭐⭐⭐⭐ |
| [level09](level09_object_detection/README.md) | How Object Detection Works | ⭐⭐⭐⭐ |
| [level10](level10_segmentation_ocr_industry/README.md) | Segmentation, OCR, and Industrial Inspection | ⭐⭐⭐⭐ |
| [level11](level11_vit_multimodal/README.md) | Vision Transformers and Multimodal Models | ⭐⭐⭐⭐⭐ |

## Fast Track (If You Only Have Time for These 5)

1. **level00** — An image = a grid of numbers. Until you internalize that one sentence, everything after it looks like magic
2. **level02** — Implement convolution yourself in numpy: we open up the heart of the CNN
3. **level05** — A complete shape-classification run: the full training → evaluation → misclassification-analysis cycle
4. **level07** — Transfer learning: nine out of ten image tasks in industry are solved this way
5. **level09** — Object detection: the principle behind answering "what, and where"

## Where This Lecture Shows Up at Work

- **Manufacturing quality inspection**: automatically detecting defects (scratches, stains, missing parts) with production-line cameras (level05, 08, 10)
- **Document automation**: OCR pipelines that find text regions in scanned contracts and receipts and convert them to text (level09, 10)
- **Retail and logistics**: locating products and spotting out-of-stock gaps in shelf photos, counting items in a warehouse (level09)
- **Healthcare and safety**: assisting diagnostic reads, spotting anomalies on CCTV — and the judgment call of "how far should we trust the detections" (level08, 09)
- **Vendor and solution review**: when a vision-solution proposal claims "99% accuracy", the ability to probe how the data was split, whether errors were analyzed, and whether augmentation was used (level06, 08)
- **Keeping up with the state of the art**: knowing what ViT and CLIP actually are when someone says "let's add multimodal AI to our service" (level11)

## Tips for Working Through It

- Every level is designed to be read while you run `main.py` yourself. In this lecture especially,
  the PNG figures are the key output — be sure to open the images generated in each level's `outputs/` folder.
- To run: use the virtual environment at the repository root, `python3 main.py` (see the root `SETUP.md` for environment setup)
- Even the levels with torch training (level03–09, 11) are deliberately small enough to finish in tens of seconds on a CPU.
- All data is drawn on the fly by the code itself, so no internet connection is required.
