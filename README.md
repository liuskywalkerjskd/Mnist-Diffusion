# Minimal Conditional DDPM (MNIST) - PyTorch Implementation

English | [中文](README_CN.md)

This is a PyTorch implementation of a Conditional Denoising Diffusion Probabilistic Model (DDPM) for generating MNIST handwritten digits.

The project implements a lightweight Context U-Net combined with Classifier-Free Guidance (CFG) technology, which not only allows generating specific digits (0-9) but also visualizes the denoising process of the diffusion model.

## ✨ Key Features

* **Minimal Code**: Clear core logic, easy to understand the principles of Diffusion Models (DDPM).
* **Conditional Generation**: Supports generating specified digits (e.g., "generate a digit 5 for me").
* **Denoising Visualization**: Supports generating videos demonstrating the transformation from pure noise to clear digits.
* **Classifier-Free Guidance (CFG)**: Context Mask is introduced during training to enhance generation control.

## 📦 Dependencies

Please ensure the following Python libraries are installed:

```bash
pip install torch torchvision numpy opencv-python pillow
```

(Note: opencv-python is used for video generation, pillow is used for image stitching)

## 📂 File Structure

This project contains 4 core Python files, each serving a specific purpose:

### 1. ddpm_model.py (Core Model)

This is the project's "brain".

* **ContextUnet**: Defines a U-Net network with Time Embedding and Context Embedding.
* **DDPM**: Defines the forward noising and reverse denoising logic of the diffusion process. Includes the sample function, implementing the Classifier-Free Guidance-based sampling algorithm.

### 2. train_ddpm.py (Training Script)

This is the model's "gym".

* Downloads the MNIST dataset.
* Initializes the model and moves it to GPU (if available).
* Executes a 30-epoch training loop and periodically saves model weights to `./weights/ddpm_mnist.pth`.
* Uses drop_prob=0.1 during training to randomly drop conditions, supporting CFG inference.

### 3. inference_ddpm.py (Inference & Generation)

This is the model's "paintbrush". Supports two modes:

* **Image Mode (default)**: Loads trained weights and generates grid images of specified digits.
* **Video Mode (--video)**: Generates MP4 videos showing digits 0-9 gradually becoming clear from complete noise, intuitively demonstrating the diffusion process.

### 4. stitch_images.py (Utility Script)

This is a simple auxiliary tool.

* Reads digit_0_grid.png through digit_9_grid.png from the generated_outputs folder.
* Stitches these 10 images into a 2-row × 5-column large image (combined_2x5_grid.png) for unified result display.

## 🚀 Quick Start

> **📌 Note:** This project includes a pre-trained model file saved in `./weights/ddpm_mnist.pth`. You can skip the training step and directly use this model for inference to see the results.

### Step 1: Train the Model (Optional)

If you want to train the model yourself, run the training script. The training process typically takes only a few minutes on a GTX 1060 or Colab T4.

```bash
python train_ddpm.py
```

After training completes, the weight file will be saved in `weights/ddpm_mnist.pth`.

### Step 2: Generate Images

Generate a single digit (e.g., digit 5):

```bash
python inference_ddpm.py --digit 5 --count 9
```

* `--digit`: The digit you want to generate (0-9).
* `--count`: The number to generate (square numbers like 9, 16, 25 are recommended).
* Results are saved in `generated_outputs/digit_5_grid.png`.

Generate denoising process video:

```bash
python inference_ddpm.py --video
```

* Results are saved in `generated_outputs/denoising_process_0to9.mp4`.

### Step 3: Stitch All Digits (Optional)

If you want to generate a summary image containing all digits 0-9:

1. First generate images for all digits (you can use the following shell script or run manually):

   **Windows PowerShell**

   ```powershell
   0..9 | ForEach-Object { python inference_ddpm.py --digit $_ --count 9 }
   ```

   **Linux / Mac**

   ```bash
   for i in {0..9}; do python inference_ddpm.py --digit $i --count 9; done
   ```

2. Run the stitching script:

   ```bash
   python stitch_images.py
   ```

3. View the generated `combined_2x5_grid.png`.

## 📊 Result Preview

![1766044209241](image/README/1766044209241.png)

## 📝 Citation & References

* [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
