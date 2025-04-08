import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreaming
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import modules
from modules.generation import generate_text, load_model_and_tokenizer
from modules.visualization import setup_visualization_ui, display_visualizations
from modules.evaluation import run_model_evaluation, setup_evaluation_ui
from modules.multimodal import load_vision_model, process_image_input, setup_multimodal_ui, generate_from_mockup
from modules.fine_tuning import setup_lora_config, prepare_dataset, run_fine_tuning, setup_fine_tuning_ui, prepare_model_for_fine_tuning
from modules.cot_visualization import setup_cot_visualization_ui, visualize_chain_of_thought, display_cot_results

# Page configuration
st.set_page_config(
    page_title="Advanced Gemma Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (load from file or include here)
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar for model selection and options
with st.sidebar:
    st.image("assets/logo.png", width=200)
    st.title("Advanced Gemma Demo")
    st.markdown("Interactive demonstration of Google's Gemma model capabilities with advanced features")
    
    st.subheader("Model Configuration")
    model_size = st.selectbox(
        "Select Gemma Model",
        options=["gemma-2b", "gemma-7b"],
        index=0,
    )
    
    mode = st.radio(
        "Select Primary Mode",
        options=["Text Generation", "Question Answering", "Code Completion"],
        index=0,
    )
    
    st.subheader("Generation Parameters")
    max_length = st.slider("Maximum Output Length", 50, 1000, 200)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)
    
    advanced_options = st.expander("Advanced Options")
    with advanced_options:
        top_k = st.slider("Top K", 1, 100, 50)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.05)
        streaming = st.checkbox("Enable Streaming Generation", value=True)
    
    st.markdown("---")
    st.markdown("Created for GSoC submission to Google DeepMind")

# Main content
st.title(f"Advanced Gemma {model_size.split('-')[1].upper()} Demo")

# Create tabs for different modes
main_tab, multimodal_tab, fine_tuning_tab, cot_tab, eval_tab = st.tabs([
    "Text Generation", 
    "Multimodal", 
    "Custom Training", 
    "Chain-of-Thought", 
    "Evaluation"
])

# Text Generation Tab
with main_tab:
    st.header("Text Generation")
    
    # Example prompts for the selected mode
    example_prompts = {
        "Text Generation": [
            "Write a short story about a robot learning to paint.",
            "Explain the concept of quantum computing to a high school student.",
            "Create a recipe for a healthy breakfast smoothie."
        ],
        "Question Answering": [
            "What are the main causes of climate change?",
            "How does the human immune system work?",
            "What is the significance of the Turing test in AI?"
        ],
        "Code Completion": [
            "Write a Python function to calculate the Fibonacci sequence.",
            "Create a simple React component for a to-do list.",
            "Write a SQL query to find the top 5 customers by total purchases."
        ]
    }
    
    # Display example prompts
    st.subheader("Example Prompts")
    example_cols = st.columns(3)
    selected_example = None

    for i, (col, example) in enumerate(zip(example_cols, example_prompts[mode])):
        if col.button(f"Example {i+1}", key=f"example_{i}"):
            selected_example = example

    # Input area
    if selected_example:
        user_input = st.text_area("Your prompt:", value=selected_example, height=100)
    else:
        user_input = st.text_area("Your prompt:", height=100)

    # Generate button
    if st.button("Generate with Gemma", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a prompt.")
        else:
            # Get advanced parameters from the sidebar
            streaming_enabled = advanced_options.checkbox("Enable Streaming Generation", value=True)
            
            if not streaming_enabled:
                with st.spinner(f"Generating with Gemma {model_size.split('-')[1].upper()}..."):
                    pass
                    
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(model_size)
            
            if model and tokenizer:
                # Generate response
                response, generation_time = generate_text(
                    user_input, 
                    model, 
                    tokenizer, 
                    max_length, 
                    temperature, 
                    top_p,
                    top_k,
                    repetition_penalty,
                    streaming_enabled
                )
                
                # Display generation stats
                st.info(f"Generation completed in {generation_time:.2f} seconds.")
                
                # Store the response in session state
                st.session_state['last_response'] = response
                st.session_state['last_prompt'] = user_input
                st.session_state['generation_time'] = generation_time
            else:
                st.error("Failed to load model. Please try again or select a different model.")

# Multimodal Tab
with multimodal_tab:
    st.header("Multimodal Processing")
    
    # Set up the multimodal UI
    uploaded_image, task = setup_multimodal_ui()
    
    if uploaded_image and task and st.button("Process Image", type="primary", key="process_image"):
        with st.spinner("Loading models..."):
            # Load Gemma model
            gemma_model, tokenizer = load_model_and_tokenizer(model_size)
            
            # Load vision model
            clip_model, clip_processor = load_vision_model()
            
            if gemma_model and clip_model:
                with st.spinner(f"Processing image with {task}..."):
                    if task == "Image Description":
                        result = process_image_input(
                            uploaded_image, 
                            gemma_model, 
                            tokenizer, 
                            clip_model, 
                            clip_processor
                        )
                        st.subheader("Image Description")
                        st.write(result)
                    
                    elif task == "Question Answering":
                        question = st.session_state.get("image_question", "What's in this image?")
                        # Implementation for question answering about images
                        st.subheader("Answer")
                        st.write("Feature to be implemented in the final version")
                        
                    elif task == "Code Generation from Mockup":
                        code = generate_from_mockup(
                            uploaded_image, 
                            gemma_model, 
                            tokenizer, 
                            clip_model, 
                            clip_processor
                        )
                        st.subheader("Generated Code")
                        st.code(code, language="html")
            else:
                st.error("Failed to load required models.")

# Custom Training Tab
with fine_tuning_tab:
    st.header("Custom Training Interface")
    
    # Set up the fine-tuning UI
    ft_params = setup_fine_tuning_ui()
    
    if ft_params and st.button("Start Fine-tuning", type="primary", key="start_ft"):
        with st.spinner("Loading base model..."):
            # Load base model
            base_model, tokenizer = load_model_and_tokenizer(model_size)
            
            if base_model:
                # Configure LoRA
                lora_config = setup_lora_config(rank=ft_params["lora_rank"])
                
                # Prepare model for fine-tuning
                with st.spinner("Preparing model with LoRA adapters..."):
                    peft_model = prepare_model_for_fine_tuning(base_model, lora_config)
                
                # Prepare dataset
                with st.spinner("Preparing dataset..."):
                    dataset = prepare_dataset(ft_params["data"], tokenizer)
                    st.info(f"Dataset prepared with {len(dataset)} examples")
                
                # Create a progress placeholder
                progress_container = st.empty()
                progress_container.info("Starting fine-tuning...")
                
                # Run fine-tuning in a simulated fashion for demo purposes
                # In a real implementation, this would use actual training
                epochs = ft_params["epochs"]
                steps_per_epoch = 10
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for epoch in range(1, epochs + 1):
                    status_text.text(f"Epoch {epoch}/{epochs}")
                    
                    for step in range(1, steps_per_epoch + 1):
                        # Simulate training step
                        time.sleep(0.5)
                        
                        # Update progress
                        progress = (epoch - 1) * steps_per_epoch + step
                        total_steps = epochs * steps_per_epoch
                        progress_bar.progress(progress / total_steps)
                        
                        # Display metrics
                        loss = 1.0 - (progress / total_steps) * 0.7  # Simulated decreasing loss
                        status_text.text(f"Epoch {epoch}/{epochs}, Step {step}/{steps_per_epoch}, Loss: {loss:.4f}")
                
                # Complete
                progress_container.success("Fine-tuning complete!")
                st.balloons()
                
                # Show download button (simulated)
                st.download_button(
                    "Download Fine-tuned Model Adapter",
                    data="This would be the model weights in a real implementation",
                    file_name="gemma_lora_adapter.bin",
                    mime="application/octet-stream"
                )
            else:
                st.error("Failed to load base model.")