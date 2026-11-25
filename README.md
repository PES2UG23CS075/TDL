# Accent-Adaptive Whisper ASR Using EWC and LoRA

## Overview

Automatic Speech Recognition (ASR) systems often struggle with accented speech because their training data may not adequately represent different regional accents.

This project explores **continual learning for accent-adaptive speech recognition**, using OpenAI's Whisper-Small model. The objective is to incrementally adapt the model to new English accents while minimizing catastrophic forgetting of previously learned accents.

We compare four adaptation strategies: Naive Fine-Tuning, Elastic Weight Consolidation (EWC), Low-Rank Adaptation (LoRA), and a Hybrid LoRA + EWC approach.

The experiments use German-accented English and South Asian English from the Common Accent dataset.

## Objectives

- Improve speech recognition across different English accents.
- Study catastrophic forgetting during sequential accent adaptation.
- Compare full-model fine-tuning with parameter-efficient adaptation.
- Reduce model storage requirements while maintaining recognition performance.
- Evaluate adaptation strategies using Word Error Rate (WER).

## Tech Stack

- **Language:** Python
- **Base Model:** OpenAI Whisper-Small
- **Deep Learning:** PyTorch
- **Model Framework:** Hugging Face Transformers
- **Parameter-Efficient Fine-Tuning:** PEFT (LoRA)
- **Dataset:** Hugging Face Datasets – Common Accent Dataset
- **Evaluation:** Hugging Face Evaluate, JiWER
- **Techniques:** Continual Learning, EWC, LoRA, Fine-Tuning

## Dataset

The Common Accent dataset is used to evaluate sequential adaptation across two accent groups.

| Accent | Description | Training Samples | Evaluation Samples |
|---|---|---:|---:|
| Accent A | German-accented English | 500 | 150 |
| Accent B | South Asian English (Indian, Pakistani, Sri Lankan) | 500 | 150 |

Audio is resampled to **16 kHz** and represented using 80-bin log-Mel spectrograms.

## Methodology

We evaluate four approaches:

### 1. Naive Fine-Tuning (Baseline)

- Fine-tune the entire Whisper-Small model on Accent A.
- Continue training on Accent B.
- No explicit mechanism is used to preserve previously learned knowledge.
- This serves as the baseline for measuring catastrophic forgetting.

### 2. Elastic Weight Consolidation (EWC)

EWC helps preserve previously learned knowledge by penalizing changes to important model parameters.

- Estimate parameter importance using the diagonal Fisher Information Matrix (FIM).
- Save the learned parameters after Accent A training.
- Apply an additional regularization penalty during Accent B training.

The objective function is:

L_total = L_task + λ Σ F_i (θ_i − θ_i*)²

Where:
- F_i represents parameter importance.
- θ_i* represents the previously learned parameter values.
- λ controls the strength of regularization.

### 3. Low-Rank Adaptation (LoRA)

LoRA enables parameter-efficient fine-tuning without updating the entire pretrained model.

- Freeze the Whisper-Small backbone.
- Introduce trainable low-rank adapter matrices.
- Apply adapters to the query and value projection layers (q_proj and v_proj).
- Maintain separate adapters for individual accents.

This reduces trainable parameters and storage requirements.

### 4. Hybrid LoRA + EWC

The hybrid approach combines the benefits of LoRA and EWC.

- Keep the Whisper-Small backbone frozen.
- Train LoRA adapters for accent adaptation.
- Apply EWC regularization to the adapter parameters.
- Preserve previously learned accent information while adapting to new accents.

This approach aims to balance recognition performance, storage efficiency, and knowledge retention.

## Experimental Configuration

| Parameter | Value |
|---|---|
| Base Model | Whisper-Small |
| Model Parameters | 244M |
| Optimizer | AdamW |
| Learning Rate | 1 × 10⁻⁴ |
| Effective Batch Size | 8 |
| Training Steps per Phase | 100 |
| EWC Regularization (λ) | 500 |
| Fisher Samples | 150 |
| LoRA Rank | 8 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.1 |
| LoRA Target Modules | q_proj, v_proj |
| Evaluation Metric | Word Error Rate (WER) |

## Results

The four approaches are evaluated using Word Error Rate (WER), forgetting, model storage size, and training time.

### Performance Comparison

| Method | WER-A ↓ | WER-B ↓ | Forgetting ↓ | Storage |
|---|---:|---:|---:|---:|
| Baseline | 0.3154 | 0.3559 | +0.0148 | 966.9 MB |
| EWC | 0.3539 | 0.4046 | +0.0289 | 966.9 MB |
| LoRA | 0.1169 | 0.1619 | 0.0000 | 3.54 MB |
| Hybrid | **0.1092** | **0.1601** | **−0.0090** | 3.54 MB |

*Lower WER and lower forgetting values indicate better performance. Negative forgetting indicates an improvement in Accent A recognition after learning Accent B.*

### Key Findings

- **Best Overall Performance:** Hybrid LoRA + EWC achieved the lowest WER on both accent groups.
- **Negative Forgetting:** The hybrid approach achieved −0.0090 forgetting, indicating a marginal improvement in previously learned Accent A performance.
- **Storage Efficiency:** LoRA-based methods reduced adapter storage by approximately 99.6% compared with full-model fine-tuning.
- **Zero Forgetting:** Separate LoRA adapters achieved zero forgetting by preserving the Accent A adapter independently.
- **Parameter Efficiency:** LoRA trained only 884,736 parameters (approximately 0.36% of the 244M-parameter model).

## Conclusion

The experiments demonstrate that parameter-efficient continual learning can be an effective approach to sequential multi-accent speech recognition.

While conventional fine-tuning and EWC over the full model showed higher forgetting and WER, LoRA and Hybrid LoRA + EWC achieved substantially better recognition performance with significantly lower storage requirements.

The Hybrid approach provided the best overall WER performance and demonstrated negative forgetting, making adapter-level regularization a promising direction for multi-accent ASR systems.

## Limitations and Future Work

- Evaluate adaptation across more than two accent groups.
- Experiment with larger Whisper models.
- Explore LoRA adapter merging for multi-accent inference.
- Investigate better EWC regularization and Fisher Information estimation.
- Evaluate performance using larger training datasets and longer training runs.

## Research Contribution

This project investigates a hybrid continual learning strategy that combines Low-Rank Adaptation with Elastic Weight Consolidation at the adapter level for sequential multi-accent speech recognition.

**Research Paper:** *Accent Adaptive Whisper Automatic Speech Recognition Using Elastic Weight Consolidation and Low Rank Adaptation*
