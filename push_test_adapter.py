import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, OFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# --- Configuration ---
BASE_MODEL_PATH = "/lustre/fast/fast/zqiu/hf_models/Qwen2.5-0.5B-Instruct"
# REPLACE THIS with your actual Hugging Face username and repo name
ADAPTER_HUB_ID = "zqiu/Qwen2.5-0.5B-Instruct-LoRA-Test" 
MAX_STEPS = 10

def main():
    # 1. Load Tokenizer & Model
    print(f"Loading model from {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # Qwen usually handles padding well, but setting pad_token is safe for training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # Use float16 if your GPU doesn't support bf16
    )

    breakpoint()

    # 2. Configure LoRA (PEFT)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16,            # Rank
        lora_alpha=32,   # Alpha scaling
        lora_dropout=0.05,
        # Target modules common for Qwen/Llama architectures
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # peft_config = OFTConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     oft_block_size=32,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # )
    
    model = get_peft_model(model, peft_config)
    print("Trainable Parameters:")
    model.print_trainable_parameters()

    # 3. Create a dummy dataset (Instruction Format)
    # In a real scenario, replace this with `load_dataset(...)`
    print("Creating dummy dataset...")
    dummy_data = [
        {"text": "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>"},
        {"text": "<|im_start|>user\nWrite a hello world in Python.<|im_end|>\n<|im_start|>assistant\nprint('Hello, World!')<|im_end|>"},
        {"text": "<|im_start|>user\nExplain LoRA.<|im_end|>\n<|im_start|>assistant\nLoRA stands for Low-Rank Adaptation. It freezes the pretrained model weights and injects trainable rank decomposition matrices.<|im_end|>"},
    ] * 10  # Duplicate data to ensure we have enough for 10 steps

    dataset = Dataset.from_list(dummy_data)

    def tokenize_function(examples):
        # Simple tokenization with padding
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 4. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir="./tmp_qwen_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=MAX_STEPS,
        logging_steps=1,
        fp16=False,   # Set to True if using older GPUs (V100, T4)
        bf16=True,    # Recommended for Ampere+ GPUs (A100, A10, 3090, 4090)
        save_strategy="no",
        report_to="none"
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Push Adapter to Hub
    print(f"Pushing adapter to {ADAPTER_HUB_ID}...")
    try:
        model.push_to_hub(ADAPTER_HUB_ID)
        print("Success! Adapter pushed.")
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        print("Make sure you are logged in with `huggingface-cli login`")

if __name__ == "__main__":
    main()