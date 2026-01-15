# Social Media Extremism Detection Challenge 🛡️

This repository contains the solution for the **Social Media Extremism Detection Challenge Kaggle Competition**. The solution achieved **1st Place on the Public Leaderboard** and **3rd Place on the Private Leaderboard**.

## 🏆 Performance

| Approach | Model | Public Score | Private Score |
| :--- | :--- | :--- | :--- |
| **Zero-Shot** | Qwen 2.5 14B Instruct | **0.97866 (1st)** | 0.85066 (3rd) |
| **Fine-Tuning** | Qwen 2.5 7B Instruct + QLoRA | 0.97333 | **0.86400 (SOTA)** |

*Note: The fine-tuned model (0.864) actually surpassed the official competition winning score in post-competition analysis.*

## 🔍 Problem Overview
The goal was to classify short, anonymized social media posts into two categories:
*   **EXTREMIST:** Content that promotes, supports, or glorifies extremist ideologies, organizations, or violence.
*   **NON_EXTREMIST:** Content that represents normal online discourse, news reporting, or neutral opinions.

## 🧠 Solution Architecture
The final system utilizes a hybrid hierarchical pipeline:
1.  **External Known Samples:** Exact matching against known training data for perfect recall.
2.  **LLM Inference:** If no match is found, the input is processed by a Qwen-based Large Language Model.

We explored two distinct approaches provided in the `notebooks/` directory:

### Approach 1: Zero-Shot Inference (Best Public Score)
*   **File:** `notebooks/qwen-zeroshot.ipynb`
*   **Model:** Qwen 2.5 14B Instruct (4-bit NF4 Quantization)
*   **Strategy:** Utilized the model's inherent safety guardrails using a strict "Content Moderation AI" system prompt.
*   **Prompting:** Explicit definitions of labels were provided in-context to reduce ambiguity.
*   **Inference:** Temperature set to `0.01` for deterministic outputs.

### Approach 2: QLoRA Fine-Tuning (Best Private Score)
*   **Files:** `notebooks/qwen-finetuning-training.ipynb` & `notebooks/qwen-finetuning-inference.ipynb`
*   **Model:** Qwen 2.5 7B Instruct (4-bit NF4 Quantization)
*   **Technique:** Parameter-Efficient Fine-Tuning (PEFT) using QLoRA.
*   **Hyperparameters:** Rank (r)=32, Alpha=64, Dropout=0.05.
*   **Method:** 
    *   Trained on a simplified objective: mapping inputs to single integer tokens ('0' or '1').
    *   **Logit-Based Classification:** Instead of text generation, we extracted raw logits for the '0' and '1' tokens and calculated softmax probability. This eliminated generation noise and hallucination.

## 📂 Repository Structure

```text
├── notebooks/
│   ├── qwen-zeroshot.ipynb              # Zero-shot inference pipeline (14B Model)
│   ├── qwen-finetuning-training.ipynb   # QLoRA training pipeline (7B Model)
│   └── qwen-finetuning-inference.ipynb  # Inference using the fine-tuned adapter
├── README.md
```

## 🔗 Important Links
*   **Competition Leaderboard:** [Social Media Extremism Detection Challenge](https://www.kaggle.com/competitions/social-media-extremism-detection-challenge/leaderboard)
*   **Solution Writeup:** [1st Place Public / 3rd Place Private Solution Discussion](https://www.kaggle.com/competitions/social-media-extremism-detection-challenge/writeups/qwen-zero-shot-and-finetuning-solution)
