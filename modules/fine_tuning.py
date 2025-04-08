import torch
import pandas as pd
import streamlit as st
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import json
import tempfile
import os

def setup_lora_config(rank=8):
    """
    Configure LoRA adapters for efficient fine-tuning.
    
    Args:
        rank (int): Rank for LoRA adapters
    
    Returns:
        LoraConfig: Configuration for LoRA fine-tuning
    """
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    return peft_config

def prepare_dataset(data, tokenizer, max_length=512):
    """
    Prepare dataset for fine-tuning from uploaded data.
    
    Args:
        data (list): List of prompt-completion pairs
        tokenizer: Tokenizer for the model
        max_length (int): Maximum sequence length
    
    Returns:
        Dataset: HuggingFace dataset ready for training
    """
    # Format the data for instruction fine-tuning
    formatted_data = []
    for item in data:
        prompt = item["prompt"]
        completion = item["completion"]
        
        # Format into instruction format
        formatted_text = f"<s>[INST] {prompt} [/INST] {completion}</s>"
        
        formatted_data.append({
            "text": formatted_text
        })
    
    # Create dataset
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def run_fine_tuning(model, dataset, output_dir, learning_rate=3e-4, epochs=3):
    """
    Run fine-tuning process with LoRA.
    
    Args:
        model: Model to fine-tune
        dataset: Prepared dataset
        output_dir (str): Directory to save results
        learning_rate (float): Learning rate
        epochs (int): Number of training epochs
    
    Returns:
        trained_model: Fine-tuned model
    """
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        save_steps=50,
        logging_steps=10,
        save_total_limit=2,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Start training
    trainer.train()
    
    return model

def setup_fine_tuning_ui():
    """
    Set up the fine-tuning UI components in Streamlit.
    
    Returns:
        dict: Fine-tuning parameters and data
    """
    st.subheader("Custom Training Interface")
    
    ft_params = {}
    
    # Dataset upload
    uploaded_file = st.file_uploader("Upload training data (JSON)", type=["json"])
    
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            st.success(f"Loaded dataset with {len(data)} examples")
            
            # Display sample data
            if len(data) > 0:
                sample = data[0]
                st.subheader("Sample data")
                st.text(f"Prompt: {sample['prompt']}")
                st.text(f"Completion: {sample['completion']}")
            
            ft_params["data"] = data
            
            # Advanced parameters
            with st.expander("Fine-tuning Parameters"):
                ft_params["learning_rate"] = st.slider("Learning Rate", 1e-5, 1e-3, 3e-4, format="%.6f")
                ft_params["epochs"] = st.slider("Epochs", 1, 10, 3)
                ft_params["lora_rank"] = st.slider("LoRA Rank", 2, 32, 8)
            
            # Start fine-tuning button
            if st.button("Start Fine-tuning"):
                return ft_params
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    return None

def prepare_model_for_fine_tuning(model, lora_config):
    """
    Prepare model for fine-tuning with LoRA.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
    
    Returns:
        model: Model prepared for fine-tuning
    """
    # Add LoRA adapters to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model