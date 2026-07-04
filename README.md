下面是帮你**续写后的 README（已整理成论文/开源项目风格，可直接用）**：

---

# TDV-Net

**Multimodal TDV-Net unveils multitask Arctic sea ice segmentation via text-guided dual-domain modeling**

## 📌 Overview

This repository contains the official implementation of **TDV-Net**, a multimodal framework for multitask Arctic sea ice segmentation.
TDV-Net integrates **text-guided semantic modeling** and **dual-domain visual feature learning** to jointly predict:

* Sea Ice Concentration (SIC)
* Stage of Development (SOD)
* Floe Size (FLOE)

The model leverages RemoteCLIP-based text embeddings to inject semantic priors into visual representations, enabling more robust Arctic sea ice understanding under complex conditions.

---

## ⚙️ Model Highlights

* **Dual-domain visual encoder** (spatial + frequency-aware representation)
* **Text-guided semantic injection (TGSI)** for class-aware feature enhancement
* **Multitask decoding heads** for SIC / SOD / FLOE prediction
* Designed for **SAR-based Arctic sea ice analysis**

---

## 📦 Pretrained Weights

We release all pretrained weights used in the paper.

📥 **Google Drive Download Link:**
[https://drive.google.com/drive/folders/XXXXXXXXXXXX](https://drive.google.com/drive/folders/XXXXXXXXXXXX)

> Replace the above link with the official shared folder link.

The folder includes:

* TDV-Net checkpoints
* Ablation model weights
* Training logs
* Configuration files

---

## 🧠 RemoteCLIP-ViT Model

TDV-Net uses **RemoteCLIP-ViT-L/14** for text embedding initialization.

📎 Official RemoteCLIP repository and weights:
[https://github.com/RemoteCLIP/RemoteCLIP](https://github.com/RemoteCLIP/RemoteCLIP)

If you use TGSI module, please download the pretrained model from the above repository and set the path in the configuration:

```bash
remoteclip_pretrained=/path/to/remoteclip-vit-l14.pt
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-repo/TDV-Net.git
cd TDV-Net
pip install -r requirements.txt
```

### Training

```bash
python train.py --config configs/tdvnet.yaml
```

### Inference

```bash
python test.py --weights path_to_checkpoint.pth
```

---

## 📊 Task Definition

TDV-Net supports three output heads:

| Task | Description           |
| ---- | --------------------- |
| SIC  | Sea Ice Concentration |
| SOD  | Stage of Development  |
| FLOE | Ice Floe Size         |

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{tdvnet2026,
  title={Multimodal TDV-Net unveils multitask Arctic sea ice segmentation via text-guided dual-domain modeling},
  year={2026}
}
```

---

## 📬 Contact

For questions or collaboration:

* Email: [your_email@example.com](mailto:your_email@example.com)
* Project: TDV-Net Arctic Sea Ice Segmentation
