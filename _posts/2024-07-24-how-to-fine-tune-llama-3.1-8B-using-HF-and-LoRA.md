# Fine-tuning Llama3.1 8B using HF and LoRA on Custom Data

Meta just released Llama3.1 models Yesterday (23rd of July, 2024), in this blog, we will fine-tune the 8B model using Hugging Face (HF) and Low-Rank Adaptation (LoRA), to enhace its performance on particular tasks/datasets.

**Table of Contents**
- [Low-Rank Adaptation (LoRA)](##Low-Rank-Adaptation-(LoRA))
  - [Concept](###Concept)
  - [Example](###Example)
- [Setting up Environment](##Setting-up-Environment)
- [Preparing Data](##Preparing-Data)

---

## Low-Rank Adaptation (LoRA)

When fine-tuning large language models like LLaMA 3.1 8B, one of the biggest challenges is the required computational resources. This is where Low-Rank Adaptation (LoRA) comes in. LoRA is a technique designed to efficiently fine-tune large language models by reducing the number of trainable parameters while maintaining model performance.

### Concept

The main idea of LoRA is to approximate the weight updates required for fine-tuning using **low-rank** matrices. By **decomposing** the original weights, LoRA allows us to train only these smaller matrices instead of updating the full weight matrix during fine-tuning.

### Example

Let's consider a simplified example to understand how LoRA works:

Suppose we have a pre-trained weight matrix (W) of size (1000x1000) (1 million parameters). In traditional fine-tuning, we would update all of these parameters. With LoRA, using a rank (r=16):

- Matrix (B) would be (1000x16)
- Matrix (A) would be (16x1000)

Total trainable parameters: ((16x1000) x2 = 32,000) parameters.

This is a **96.8%** reduction in trainable parameters!

---

## Setting up Environment

1. Install latest version of transformers
New Llama 3.1 models have new attributes within the model config, we won't be able to load the model unless we upgrade transformers library version
```console
pip install --upgrade transformers
```

2. Request access to Llama 3.1 8B model
You will have to sign-in to HuggingFace Hub, and request access to [Llama 3.1 8B Instruct model](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
<img width="930" alt="Screenshot 2024-07-24 at 9 15 14 PM" src="https://github.com/user-attachments/assets/b52d3fd7-edb3-4950-bc34-4c48b824b29c">

---

## Preparing-Data
placeholder
