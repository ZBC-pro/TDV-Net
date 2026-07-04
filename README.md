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
(https://drive.google.com/drive/folders/1KZMXgHsXUuztxPI44jKeFj29KYHLbop)

The folder includes:

* TDV-Net checkpoints
* diff checkpoints

---

## 🧠 RemoteCLIP-ViT Model

TDV-Net uses **RemoteCLIP-ViT-L** for text embedding initialization.

📎 Official RemoteCLIP repository and weights:
[https://github.com/RemoteCLIP/RemoteCLIP](https://github.com/RemoteCLIP/RemoteCLIP)

---

## 📊 Task Definition

TDV-Net supports three output heads:

| Task | Description           |
| ---- | --------------------- |
| SIC  | Sea Ice Concentration |
| SOD  | Stage of Development  |
| FLOE | Ice Floe Size         |

---

## 📬 Contact

For questions or collaboration:

* Email: [huangmeng@jou.edu.cn]
* Project: TDV-Net Arctic Sea Ice Segmentation
