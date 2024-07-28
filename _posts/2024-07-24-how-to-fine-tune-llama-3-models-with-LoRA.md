# Fine-tuning Llama3 Models with LoRA on Custom Data

Meta just released Llama3.1 models yesterday (23rd of July, 2024), so I thought it would be a great time to discuss how we can fine-tune Llama 3 models. In this blog, we will fine-tune the Llama3 8B model with Low-Rank Adaptation (LoRA), to enhance its performance on particular tasks/datasets.

**Table of Contents**
- Low-Rank Adaptation (LoRA)
  - Concept
  - Example
- Setting up the Environment
- Data Preparation
- Fine-tuning
  - Seeding
  - Load and Quantize Model
  - Add Padding Token
  - Format Training Examples
  - Prepare Training Datasets
  - Use LoRA
  - Training Configurations
  - Start Training
- Loading and Merging Saved Model
- Pushing Trained Model to HF Hub
- Evaluation

---

## Low-Rank Adaptation (LoRA)

When fine-tuning large language models like LLaMA 3/3.1 8B, one of the biggest challenges is the required computational resources. This is where Low-Rank Adaptation (LoRA) comes in. LoRA is a technique designed to efficiently fine-tune large language models by reducing the number of trainable parameters while maintaining model performance.

### Concept

The main idea of LoRA is to approximate the weight updates required for fine-tuning using **low-rank** matrices. By **decomposing** the original weights, LoRA allows us to train only these smaller matrices instead of updating the full weight matrix during fine-tuning.

### Example

Let's consider a simplified example to understand how LoRA works:

Suppose we have a pre-trained weight matrix (W) of size 1000x1000 (1 million parameters). In traditional fine-tuning, we would update all of these parameters. With LoRA, using a rank r=16:

- Matrix (B) would be (1000x16)
- Matrix (A) would be (16x1000)

Total trainable parameters: ((16x1000) x2 = 32,000) parameters.

This is a **96.8%** reduction in trainable parameters!

---

## Setting up the Environment

**Note**: In this post, I will be using Llama 3 8B as an example, but you should be able to train Llama 3.1 in the exact same way. This section should be relevant only if you will train 3.1 models.

1. Install the latest version of transformers
New Llama 3.1 models have new attributes within the model config, we won't be able to load the model unless we upgrade transformers library version
```console
pip install --upgrade transformers
```

2. Request access to Llama 3.1 8B model
You will have to sign-in to HuggingFace Hub, and request access to [Llama 3.1 8B Instruct model](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
<img width="930" alt="Screenshot 2024-07-24 at 9 15 14 PM" src="https://github.com/user-attachments/assets/b52d3fd7-edb3-4950-bc34-4c48b824b29c">

---

## Data Preparation

As the main goal of this blog post is to train the model on your own custom dataset, we will be talking abstractly about how to train the model on any dataset, and how the data should be formatted.

First, let's have two main columns in the dataset:
```
question: <this is the prompt, and this is what the model will be trained on>
answer: <this is the answer to the prompt/question, this is the label>
```

It's **not** recommended to do any normalization/cleaning on your text, it's preferred to leave text as is when training LLM.

---

## Fine-tuning

**1. Seeding**

To ensure reproducibility, we will need to set seeds.
```py
import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(0)
```

**2. Load and Quantize Model**

The 8B model is still quite big to fit on average Colab GPUs (e.g T4), so It's recommended to quantize the model to a lower precision rate before starting training.

Here’s how we can load and quantize the model using BitsAndBytes to 8-bit

**Note**: This will reduce GPU utilization from **18GB to approximately 6GB.**
```py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", 
    quantization_config=quantization_config,
    device_map="auto"
)
```

**3. Add Padding Token**

Llama 3 tokenizers do not have a `padding` token by default, so, to train the model in batches, we will need to configure this ourselves, and it has also proven to show better results even when training with a batch size of one sample.
```py
PAD_TOKEN = "<|pad|>"

tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

# we added a new padding token to the tokenizer, we have to extend the embddings
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

print(tokenizer.pad_token, tokenizer.pad_token_id)
# output: ('<|pad|>', 128256)
```

**4. Format Training Examples**

We need to properly format all of our training examples, I have my custom data in `pandas` dataframe with 2 columns `question` and `answer`, and here is how we can format them
```py
from textwrap import dedent

def format_example(row: dict):
    prompt = dedent(
        f"""
        {row['question']}
        """
    )
    messages = [
        # the system prompt is very important to adjust/control the behavior of the model, make sure to use it properly accoring to your task
        {"role": "system", "content": "You're a document classifier, try to classify the given document as relevant or irrelevant"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row['answer']}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# format the training examples into a new text column
df['text'] = df.apply(format_example, axis=1)
```

**5. Prepare Training Datasets**

First, we need to create our training, validation, and test splits to evaluate the model during training and test it afterward
```py
from sklearn.model_selection import train_test_split

train, temp = train_test_split(df, test_size=0.2, random_state=1)
val, test = train_test_split(temp, test_size=0.2, random_state=1)

# save training-ready data to JSON
train.to_json("train.json", orient='records', lines=True)
val.to_json("val.json", orient='records', lines=True)
test.to_json("test.json", orient='records', lines=True)
```

Second, create HF datasets
```py
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={'train': 'train.json', 'validation': 'val.json', 'test': 'test.json'}
)

# print a training exmaple
print(dataset['train'][0]['text'])
```

Third, create the training-ready datesets
```py
from trl import DataCollatorForCompletionOnlyLM

# in order to only evaluate the generation of the model, we shouldn't consider the text that were already inputed, we will use the end header id token to get the generated text only, and mask everything else
response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
```

**6. Use LoRA**

Use LoRA to reduce the number of trainable parameters, you can print the model modules by using `print(model)`, and you can see the names of the modules being targeted here
```py
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)

# this is recommended by original lora paper: using lora, we should target the linear layers only
lora_config = LoraConfig(
    r=32,  # rank for matrix decomposition
    lora_alpha=16,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

pirnt(model.print_trainable_parameters())
# output: trainable params: 83,886,080 || all params: 8,114,212,864 || trainable%: 1.0338
```

**7. Training Configurations**

Set the training configurations
```py
from trl import SFTConfig, SFTTrainer

OUTPUT_DIR = "experiments"

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field='text',  # this is the final text example we formatted
    max_seq_length=4096,
    num_train_epochs=1,
    per_device_train_batch_size=2,  # training batch size
    per_device_eval_batch_size=2,  # eval batch size
    gradient_accumulation_steps=4,  # by using gradient accum, we updating weights every: batch_size * gradient_accum_steps = 4 * 2 = 8 steps
    optim="paged_adamw_8bit",  # paged adamw
    eval_strategy='steps',
    eval_steps=0.2,  # evalaute every 20% of the trainig steps
    save_steps=0.2,  # save every 20% of the trainig steps
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,  # also try bf16=True
    save_strategy='steps',
    warmup_ratio=0.1,  # learning rate warmup
    save_total_limit=2,
    lr_scheduler_type="cosine",  # scheduler
    save_safetensors=True,  # saving to safetensors
    dataset_kwargs={
        "add_special_tokens": False,  # we template with special tokens already
        "append_concat_token": False,  # no need to add additional sep token
    },
    seed=1
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    data_collator=collator,
)
```

**7. Start Training**

Now, we are finally ready to start training
```py
trainer.train()
```

We can see how the training is going well, and the validation loss is going down
<img width="578" alt="Screenshot 2024-07-24 at 10 21 14 PM" src="https://github.com/user-attachments/assets/e31aa6a6-5dac-4ea6-b982-a7f7a1fffe86">

---

## Loading and Merging Saved Model

Models are being saved during training, but while training with LoRA, the model will be saved with an Adapter, so we will load both the Model and the Adapter, merge them, and have a final model that we can easily push to HF Hub
```py
from peft import PeftModel

NEW_MODEL="path_to_saved_model"

# load trained/resized tokenizer
tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)

# here we are loading the raw model, if you can't load it on your GPU, you can just change device_map to cpu
# we won't need gpu here anyway
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
    device_map='auto',
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = PeftModel.from_pretrained(model, NEW_MODEL)
model = model.merge_and_unload()
```

---

## Pushing Trained Model to HF Hub

Now we have merged the Model and the Adapter, we can push the Model to HF Hub and load it from there

**1. Sign-in to HF Hub using HF-cli**

Sign-in, and make sure to create a token with write access, check [HF Docs](https://huggingface.co/docs/transformers/en/model_sharing) for more info
```console
huggingface-cli login
```

**2. Push Model and Tokenizer**
```py
username = "your_username"
repo_name = "repo_name"
model.push_to_hub(f"{username}/{repo_name}", tokenizer=tokenizer, max_shard_size="5GB", private=True)
tokenizer.push_to_hub(f"{username}/{repo_name}", private=True)
```

---

## Evaluation

In a separate notebook, we can load our trained model and tokenizer from HF hub, and use them for inference
```py
from textwrap import dedent
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

MODEL_NAME = "your_repo_name"

# this should create
df = pd.read_csv('data.csv')

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
)

# load trained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=quantization_config,
    device_map="auto"
)

pipe = pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    return_full_text=False
)


def creaet_test_prompt(row):
    prompt = dedent(
        f"""
        {row['question']}
        """
    )
    messages = [
        # the system prompt is very important to adjust the control the behavior of the model, make sure to use properly accoring to your task
        {"role": "system", "content": "You're a document classifier, try to classify the given document as relevant or irrelevant"},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


questions = df['question'].tolist()
prompt = creaet_test_prompt(questions[0])
result = pipe(prompt)[0]['generated_text']
print(result)
# output: <model's response>
```

