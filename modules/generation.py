import torch
import streamlit as st
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreaming

@st.cache_resource
def load_model_and_tokenizer(model_name):
    """
    Load model and tokenizer with caching for efficiency.
    
    Args:
        model_name (str): Model identifier (e.g., "gemma-2b", "gemma-7b")
    
    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    model_id = f"google/{model_name}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_text(prompt, model, tokenizer, max_length, temperature, top_p, top_k=50, repetition_penalty=1.1, streaming=True):
    """
    Generate text using the model with support for streaming generation.
    
    Args:
        prompt (str): Input prompt
        model: Language model
        tokenizer: Tokenizer for the model
        max_length (int): Maximum generation length
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
        top_k (int): Top-k sampling parameter
        repetition_penalty (float): Penalty for token repetition
        streaming (bool): Whether to stream generation token-by-token
    
    Returns:
        tuple: (generated_text, generation_time)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    
    # Create streaming output placeholder if streaming is enabled
    if streaming:
        output_container = st.empty()
        streamer = TextStreaming(tokenizer, skip_special_tokens=True, skip_prompt=True)
        
        generation_kwargs = {
            **inputs,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # Start the generation in a separate thread
        full_response = ""
        
        # Start generation
        model.generate(**generation_kwargs)
        
        # Stream the output to the UI
        for new_text in streamer:
            full_response += new_text
            output_container.markdown(full_response)
        
        response = full_response
    else:
        # Non-streaming generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # For code responses, try to detect and format the language
    if "```" in response:
        response = format_code_response(response)
        
    return response, generation_time

def format_code_response(response):
    """
    Format code blocks in the response with proper language tags.
    
    Args:
        response (str): Text response from the model
    
    Returns:
        str: Formatted response with proper code blocks
    """
    # Extract code blocks and detect language
    code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', response, re.DOTALL)
    
    formatted_response = response
    for lang, code in code_blocks:
        if not lang:
            # Try to detect language from code
            if 'def ' in code or 'import ' in code or 'class ' in code:
                lang = 'python'
            elif '<div' in code or '<html' in code:
                lang = 'html'
            elif 'function' in code and '{' in code:
                lang = 'javascript'
            elif 'SELECT' in code.upper() and 'FROM' in code.upper():
                lang = 'sql'
            else:
                lang = 'text'
        
        # Replace the original code block with a properly labeled one
        original_block = f'```\n{code}\n```'
        new_block = f'```{lang}\n{code}\n```'
        formatted_response = formatted_response.replace(original_block, new_block)
    
    return formatted_response

def get_example_prompts(mode):
    """
    Provide example prompts based on the selected mode.
    
    Args:
        mode (str): The operation mode (Text Generation, Question Answering, or Code Completion)
    
    Returns:
        list: List of example prompts
    """
    if mode == "Text Generation":
        return [
            "Write a short story about a robot learning to paint.",
            "Explain the concept of quantum computing to a high school student.",
            "Create a recipe for a healthy breakfast smoothie."
        ]
    elif mode == "Question Answering":
        return [
            "What are the main causes of climate change?",
            "How does the human immune system work?",
            "What is the significance of the Turing test in AI?"
        ]
    else:  # Code Completion
        return [
            "Write a Python function to calculate the Fibonacci sequence.",
            "Create a simple React component for a to-do list.",
            "Write a SQL query to find the top 5 customers by total purchases."
        ]