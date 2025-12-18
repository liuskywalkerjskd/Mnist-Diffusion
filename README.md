# Minimal Conditional DDPM (MNIST) - PyTorch Implementation

这是一个基于 PyTorch 实现的条件生成扩散模型 (Conditional DDPM)，用于生成 MNIST 手写数字。

本项目实现了一个轻量级的 Context U-Net，结合了 Classifier-Free Guidance (CFG) 技术，不仅可以指定生成特定的数字（0-9），还可以可视化扩散模型的去噪过程。

## ✨ 项目特点

* 极简代码：核心逻辑清晰，易于理解扩散模型（DDPM）原理。
* 条件生成：支持指定数字生成（例如：“给我生成一个数字 5”）。
* 去噪可视化：支持生成从纯噪声到清晰数字的视频演示。
* 无分类器引导 (CFG)：训练中引入了 Context Mask，增强了生成的控制力。

## 📦 环境依赖

请确保安装了以下 Python 库：

```bash
pip install torch torchvision numpy opencv-python pillow
```

(注：opencv-python 用于生成视频，pillow 用于拼接图片)

## 📂 文件结构说明

本项目包含 4 个核心 Python 文件，各司其职：

### 1. ddpm_model.py (核心模型)

这是项目的“大脑”。

* ContextUnet: 定义了一个带有时序嵌入（Time Embedding）和上下文嵌入（Context Embedding）的 U-Net 网络。
* DDPM: 定义了扩散过程的前向加噪和反向去噪逻辑。包含了 sample 函数，实现了基于 Classifier-Free Guidance 的采样算法。

### 2. train_ddpm.py (训练脚本)

这是模型的“健身房”。

* 负责下载 MNIST 数据集。
* 初始化模型并将其移动到 GPU（如果可用）。
* 执行 30 个 Epoch 的训练循环，并定期保存模型权重到 ./weights/ddpm_mnist.pth。
* 训练时采用了 drop_prob=0.1 来随机丢弃条件，以支持 CFG 推理。

### 3. inference_ddpm.py (推理与生成)

这是模型的“画笔”。支持两种模式：

* 图片模式 (默认)：加载训练好的权重，生成指定数字的网格图。
* 视频模式 (--video)：生成数字 0-9 从完全噪声逐渐变清晰的 MP4 视频，直观展示扩散过程。

### 4. stitch_images.py (工具脚本)

这是一个简单的辅助工具。

* 它会读取 generated_outputs 文件夹下的 digit_0_grid.png 到 digit_9_grid.png。
* 将这 10 张图片拼成一个 2行 x 5列 的大图 (combined_2x5_grid.png)，方便统一展示结果。

## 🚀 快速开始

### 第一步：训练模型

运行训练脚本，训练过程在 GTX 1060 或 Colab T4 上通常只需几分钟。

```bash
python train_ddpm.py
```

训练完成后，权重文件将保存在 weights/ddpm_mnist.pth 。

### 第二步：生成图像

生成单个数字（例如数字 5）：

```bash
python inference_ddpm.py --digit 5 --count 9
```

* --digit: 想要生成的数字 (0-9)。
* --count: 生成的数量（建议平方数，如 9, 16, 25）。
* 结果保存在 generated_outputs/digit_5_grid.png 。

生成去噪过程视频：

```bash
python inference_ddpm.py --video
```

* 结果保存在 generated_outputs/denoising_process_0to9.mp4。

### 第三步：拼接所有数字 (可选)

如果你想生成一张包含 0-9 所有数字的汇总图：

1. 先生成所有数字的图片（可以使用以下 Shell 脚本或手动运行）：

   Windows PowerShell

   ```powershell
   0..9 | ForEach-Object { python inference_ddpm.py --digit $_ --count 9 }
   ```

   Linux / Mac

   ```bash
   for i in {0..9}; do python inference_ddpm.py --digit $i --count 9; done
   ```
2. 运行拼接脚本：

   ```
   python stitch_images.py
   ```
3. 查看生成的 combined_2x5_grid.png。

## 📊 结果预览

![1766044209241](image/README/1766044209241.png)

## 📝 引用与参考

* [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
