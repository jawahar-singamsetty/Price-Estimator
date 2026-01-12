# 🧠 Price Estimator – Fine-Tuned LLaMA 3.2 (3B)

## Overview

This repository contains a **fine-tuned Large Language Model (LLM)** for **estimating product prices from unstructured item descriptions**. The model is trained to infer realistic market prices using natural language inputs such as product features, condition, and context.

The model was fine-tuned by **Jawahar S R** using **QLoRA** for efficiency and is optimized to run on limited GPU memory while maintaining strong predictive performance.

---

## Base Model

* **Base Architecture:** `meta-llama/Llama-3.2-3B`
* **Model Type:** Causal Language Model
* **Frameworks:** Hugging Face Transformers, TRL, PEFT

---

## Dataset

* **Source:** `ed-donner/items_prompts_lite`
* **Structure:** Instruction-style prompt → response pairs
* **Split Used:**

  * Training set
  * Validation set (1,000 samples)
  * Test set (held out)

The dataset consists of item descriptions paired with price-related outputs, making it suitable for supervised fine-tuning.

---

## Fine-Tuning Method

The model was fine-tuned using **Supervised Fine-Tuning (SFT)** with **QLoRA** to minimize memory usage and training cost.

### Quantization

* **Base model quantization:** 4-bit (NF4)
* **Double quantization:** Enabled
* **Compute dtype:** `bfloat16` (Ampere+ GPUs) or `float16`

### LoRA Configuration

* **LoRA rank (`r`):** 16
* **LoRA alpha:** 32
* **LoRA dropout:** 0.1
* **Target modules:**

  * Attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  * MLP projections: `gate_proj`, `up_proj`, `down_proj`

This setup allows the model to learn pricing-relevant representations efficiently without updating all base weights.

---

## Training Configuration

* **Epochs:** 2
* **Batch size:** 32
* **Gradient accumulation:** 1
* **Max sequence length:** 175
* **Optimizer:** `paged_adamw_32bit`
* **Learning rate:** 2e-4
* **LR scheduler:** Cosine
* **Warmup ratio:** 0.01
* **Weight decay:** 0.001
* **Gradient clipping:** 0.3

Training progress and evaluation metrics were tracked using **Weights & Biases (W&B)**.

---

## Output

The model generates **price-related text outputs** based on the learned prompt–response format from the dataset. These outputs can be directly parsed or post-processed to obtain numeric price estimates, depending on downstream usage.

---

## Intended Use Cases

* Marketplace price estimation
* E-commerce analytics
* Internal pricing tools
* Data science experimentation with LLM-based regression-style tasks

---

## Limitations

* Predictions are **estimates**, not authoritative valuations
* Performance depends on the quality and detail of input descriptions
* Domain-specific or rare products may require additional fine-tuning or calibration

---

## Deployment

The model supports:

* GPU inference with 4-bit quantization
* Disk offloading for low-memory environments
* Interactive demos using Gradio
* API deployment via FastAPI or similar frameworks

---

## Author

**Jawahar S R**

---

Just tell me 👍
