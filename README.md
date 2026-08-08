# CardioPulmoRAD-AI
CardioPulmoRAD-AI is a multimodal explainable AI framework for early cardiopulmonary risk stratification and tele-triage. It fuses chest radiographs (DenseNet-121) and 13 clinical vitals (MLP) with dual XAI (Grad-CAM + SHAP) to support frontline health workers and decision support including specialist-in-the-loop referral in low-resource settings.
# CardioPulmoRAD-AI 🫀🫁🤖

> **A Multimodal Explainable Artificial Intelligence Framework for Early Cardiopulmonary Risk Stratification and Referral Decision Support Using Chest Radiographs and Clinical Data**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/IEEE_Access-Submitted-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

---

## 📌 Overview

**CardioPulmoRAD-AI** is an end-to-end multimodal deep learning and Explainable Artificial Intelligence (XAI) framework engineered for rapid triage, risk stratification, and specialist-in-the-loop tele-consultation in resource-constrained primary healthcare settings. 

By dynamically fusing high-resolution spatial feature embeddings from posterior-anterior (PA) chest radiographs with 13 structured physiological parameters (vital signs and clinical variables), CardioPulmoRAD-AI overcomes the diagnostic blind spots inherent in single-modality vision models.

---

## 🌟 Key Features

* **Multimodal Intermediate Fusion:** Concatenates spatial embeddings from a pre-trained **DenseNet-121** backbone ($V_{\text{img}} \in \mathbb{R}^{1024}$) and clinical embeddings from a multi-layer perceptron (**TabularMLP**, $V_{\text{clin}} \in \mathbb{R}^{128}$) into a unified latent feature representation ($V_{\text{fused}} \in \mathbb{R}^{1152}$).
* **Dual-Modality Explainable AI (XAI):**
  * **Grad-CAM** for spatial radiological feature attribution and visual heatmap localization.
  * **SHAP (SHapley Additive exPlanations)** for clinical variable contribution scoring.
* **Specialist-in-the-Loop Tele-Triage Gateway:** A 3-tier risk stratification protocol designed to connect frontline health workers with remote cardiologists, radiologists, and pulmonologists before tertiary emergency transfer.
* **Low-Resource Optimization:** A lightweight reference implementation designed to run efficiently on edge hardware without demanding enterprise GPU infrastructure.

---

## 🏗️ Repository Structure

```text
CardioPulmoRAD-AI/
├── LICENSE                     # Open Source License (MIT)
├── CITI_Certificate.pdf        # Research Ethics & Compliance Certification
├── requirements.txt            # Python Dependencies
├── demo_simulation.py          # Multimodal Forward Pass & Triage Simulation Script
└── src/
    └── models.py               # PyTorch Architecture (DenseNet-121 + MLP + Fusion)
