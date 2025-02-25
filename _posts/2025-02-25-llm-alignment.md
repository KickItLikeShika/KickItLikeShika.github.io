#  LLM Alignment: SFT, RL, and ORPO

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding LLM Alignment](#understanding-llm-alignment)
3. [Traditional Alignment Techniques](#traditional-alignment-techniques)
   1. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
   2. [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
4. [ORPO: A Simpler Alternative](#orpo-a-simpler-alternative)
   1. [What is ORPO?](#what-is-orpo)
   2. [How ORPO Works](#how-orpo-works)
   3. [ORPO vs Traditional Methods](#orpo-vs-traditional-methods)
5. [Implementation and Setup](#implementation-and-setup)
   1. [Dataset](#dataset)
   2. [Model Architecture](#model-architecture)
   3. [Training Configuration](#training-configuration)
6. [Experimental Results](#experimental-results)
   1. [Performance Metrics](#performance-metrics)
   2. [Analysis](#analysis)
7. [Why Choose ORPO?](#why-choose-orpo)
8. [Conclusion](#conclusion)

The code for these experiments is available in this GitHub repository: https://github.com/KickItLikeShika/llm-alignment
Model checkpoints can be found here:
1. SFT Model: https://huggingface.co/KickItLikeShika/SFTLlama-3.2-1B
2. ORPO Model: https://huggingface.co/KickItLikeShika/ORPOLlama-3.2-1B

## Introduction

Large Language Models (LLMs) have revolutionized natural language processing with their remarkable capabilities in text generation, summarization, translation, and more. However, aligning these models with human values, preferences, and intentions is a process known as **LLM alignment**, which remains a significant challenge. 

In this blog post, I'll explore some methods for aligning LLMs, with a specific focus on a relatively new technique called **Odds Ratio Preference Optimization (ORPO)**. I'll share findings from experiments comparing traditional alignment methods with ORPO, demonstrating how this approach might simplify the alignment pipeline while maintaining competitive performance.

## Understanding LLM Alignment

LLM alignment refers to the process of ensuring that language models produce outputs that are:

- **Helpful**: Providing useful information that meets user needs
- **Harmless**: Avoiding harmful, misleading, or inappropriate content

Unaligned models may produce outputs that are technically coherent but fail to meet human expectations, violate ethical standards, or generate harmful content. The goal of alignment is to narrow this gap between model capabilities and human values.

## Traditional Alignment Techniques

### Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning is typically the first step in aligning a pre-trained LLM. The process involves:

1. Curating high-quality demonstration data showing desired model behavior
2. Fine-tuning the pre-trained model on this demonstration data
3. Optimizing the model to predict the next token in these demonstrations

Here's a simplified implementation of our SFT training function:
```py
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

training_args = TrainingArguments(
    output_dir="./results/sft",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    weight_decay=0.001,
    optim="paged_adamw_8bit"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    tokenizer=tokenizer
)

trainer.train()
```

While SFT is effective, it has several limitations:
- Requires high-quality examples written by humans
- Can't easily distinguish between multiple acceptable responses
- May lead to models that are overly verbose or have other stylistic issues

### Reinforcement Learning from Human Feedback (RLHF)

To address SFT limitations, researchers developed RLHF, which involves:

1. **Reward Model Training**: Train a model to predict human preferences between different model outputs
2. **Policy Optimization**: Use reinforcement learning (typically PPO) to optimize the model against this reward

RLHF has shown impressive results in aligning models, but introduces complexity:
- Requires multiple components (reward model, policy optimization)
- Can be unstable during training
- Computationally expensive and complex to implement

## ORPO: A Simpler Alternative

### What is ORPO?

Odds Ratio Preference Optimization (ORPO) is a direct preference optimization technique that aims to simplify the alignment process. Unlike RLHF, which requires a separate reward model, ORPO works directly with preference data.

### How ORPO Works

ORPO optimizes the model by:

1. Taking pairs of responses (chosen and rejected) for a given prompt
2. Computing the odds ratio between the probabilities of these responses
3. Optimizing the model to increase this ratio while staying close to the reference model

```py
training_args = ORPOConfig(
    output_dir="./results/orpo",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    beta=0.1,
    num_train_epochs=1,
    report_to="wandb"
)

trainer = ORPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer
)

trainer.train()
```

### ORPO vs Traditional Methods

ORPO offers several advantages over the traditional SFT + RLHF pipeline:

1. **Simplicity**: Works directly with preference data without a separate reward model
2. **Efficiency**: Requires significantly less computational resources than RLHF
3. **Stability**: More stable training dynamics compared to reinforcement learning
4. **Training Time**: Faster to train than the full alignment pipeline

## Implementation and Setup

### Dataset

For our experiments, we used a subset of `mlabonne/orpo-dpo-mix-40k` dataset, which contains:
- 20,000 examples with preferred and non-preferred responses
- A diverse range of topics and instruction types
- High-quality human preference data

Loading the dataset was straightforward:

```py
def load_and_prepare_dataset():

    if os.path.exists(f"{DATASET_PATH}/train.json") and os.path.exists(f"{DATASET_PATH}/val.json"):
        print("Loading cached dataset...")
    
        with open(f"{DATASET_PATH}/train.json", 'r') as f:
            train_dataset = load_dataset('json', data_files=f"{DATASET_PATH}/train.json")['train']
        
        with open(f"{DATASET_PATH}/val.json", 'r') as f:
            val_dataset = load_dataset('json', data_files=f"{DATASET_PATH}/val.json")['train']
    else:
        print("Downloading and preparing dataset...")
        dataset = load_dataset(DATASET_NAME, split="all")
        dataset = dataset.shuffle(seed=42).select(range(20000))
        
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        
        os.makedirs(DATASET_PATH, exist_ok=True)

        train_dataset.to_json(f"{DATASET_PATH}/train.json")
        val_dataset.to_json(f"{DATASET_PATH}/val.json")
    
    return train_dataset, val_dataset
```


### Model Architecture

We used the Llama-3.2-1B as our base model with:
- 4-bit quantization for efficient training
- LoRA adapters targeting key attention layers
- Flash Attention 2 for faster training
```py
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
```

### Training Configuration

Our experiments included three training scenarios:
1. Base model evaluation (no fine-tuning)
2. SFT training on chosen responses
3. ORPO training directly on the base model


## Experimental Results

### Performance Metrics

We evaluated model performance using ROUGE scores, which measure the overlap between the model's outputs and reference texts:
```
Base Model: {'rouge1': 0.14786865275767497, 'rouge2': 0.05741813089556658, 'rougeL': 0.09560573415051726}
SFT Model: {'rouge1': 0.16640010568749722, 'rouge2': 0.06878433704191969, 'rougeL': 0.1018761352773121}
ORPO Model from Base: {'rouge1': 0.1505326146060846, 'rouge2': 0.05853340860919248, 'rougeL': 0.09578936523988472}
```

### Analysis

The results reveal interesting patterns:

1. **SFT Performance**: SFT showed the strongest performance across all ROUGE metrics, with a 12.5% improvement in ROUGE-1 over the base model. This confirms the value of supervised fine-tuning for basic alignment.

2. **ORPO Direct Improvement**: ORPO applied directly to the base model achieved a 1.8% improvement in ROUGE-1 without first requiring SFT. 

3. **ORPO vs SFT Gap**: While ORPO didn't match SFT's ROUGE scores, the performance gap isn't dramatic, especially considering the simplified training approach.

Beyond these metrics, we observed that:
- ORPO-trained models tended to produce more concise answers
- SFT models occasionally exhibited more verbose responses
- The base model sometimes strayed from the prompt instructions

## Why Choose ORPO?

Based on our experiments, ORPO offers several compelling advantages:

1. **Pipeline Simplification**: ORPO can potentially bypass the need for SFT (in some cases), streamlining the alignment process.
2. **Resource Efficiency**: Training with ORPO required less computational resources than the full SFT + RLHF pipeline.
3. **Acceptable Performance Trade-off**: While ORPO didn't match SFT in ROUGE scores, the performance gap might be acceptable in cases where resource constraints or training speed are priorities.
4. **Direct Preference Learning**: ORPO directly optimizes for human preferences rather than imitating examples, which may better capture nuanced quality differences.

The decision between ORPO and traditional methods ultimately depends on your specific use case:

- If you need maximum performance and have ample resources: SFT followed by RLHF or DPO
- If you have resource constraints but need good alignment: ORPO directly on the base model
- For rapid prototyping or experimentation: ORPO offers a simpler starting point

## Conclusion

LLM alignment remains a critical challenge in developing safe, helpful AI systems. Our experiments demonstrate that newer approaches like ORPO offer promising alternatives to traditional alignment methods, potentially simplifying the alignment pipeline while maintaining reasonable performance.

Key takeaways:

1. SFT continues to provide strong performance for basic alignment
2. ORPO offers a viable alternative that works directly with preference data
3. The choice between methods depends on your specific requirements, resources, and priorities

As the field evolves, we'll likely see further innovations in alignment techniques that balance performance, efficiency, and training complexity. The exploration of methods like ORPO represents an exciting direction in creating AI systems that are both powerful and aligned with human values.
