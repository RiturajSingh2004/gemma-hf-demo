import streamlit as st
import torch
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit compatibility
import numpy as np
from io import StringIO
import time

def stream_generate_with_tracking(prompt, model, tokenizer, max_length=512, temperature=0.7, top_p=0.9):
    """
    Generate text with intermediate state tracking for visualizing chain of thought.
    
    Args:
        prompt (str): Input prompt
        model: Language model
        tokenizer: Tokenizer for the model
        max_length (int): Maximum generation length
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
    
    Returns:
        list: List of intermediate generation states
    """
    # Add chain-of-thought prompt
    cot_prompt = f"{prompt}\n\nLet's think through this step by step:"
    
    # Tokenize the prompt
    inputs = tokenizer(cot_prompt, return_tensors="pt").to(model.device)
    
    # Track intermediate states
    reasoning_steps = [cot_prompt]
    current_text = cot_prompt
    
    # Set up for token-by-token generation
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Stream generation
    with torch.no_grad():
        for _ in range(max_length):
            # Get next token predictions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get logits for the next token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create a mask for logits to keep
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            # Decode and track current state
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            current_text += token_text
            
            # Track at meaningful points (e.g., after sentences or steps)
            if token_text in ['.', '?', '!', '\n'] or "Step" in token_text:
                reasoning_steps.append(current_text)
            
            # Check for EOS
            if next_token[0].item() == tokenizer.eos_token_id:
                break
    
    # Make sure the final state is included
    if reasoning_steps[-1] != current_text:
        reasoning_steps.append(current_text)
    
    return reasoning_steps

def parse_reasoning_steps(reasoning_text):
    """
    Parse the reasoning steps from the generated text.
    
    Args:
        reasoning_text (str): Generated text with reasoning
    
    Returns:
        list: List of distinct reasoning steps
    """
    # Attempt to find clearly defined steps
    step_pattern = r"Step\s*\d+:?\s*(.*?)(?=Step\s*\d+:|$)"
    steps = re.findall(step_pattern, reasoning_text, re.DOTALL)
    
    # If no clear steps found, split by sentences or line breaks
    if not steps:
        # Try splitting by numbered lines
        numbered_pattern = r"\d+\.\s*(.*?)(?=\d+\.\s*|$)"
        steps = re.findall(numbered_pattern, reasoning_text, re.DOTALL)
    
    if not steps:
        # Fall back to sentences
        steps = [s.strip() for s in re.split(r'[.!?]\s+', reasoning_text) if s.strip()]
    
    # Clean up steps
    steps = [step.strip() for step in steps]
    steps = [step for step in steps if step]
    
    return steps

def create_reasoning_graph(reasoning_steps):
    """
    Create a graph visualization of the reasoning process.
    
    Args:
        reasoning_steps (list): List of reasoning steps
    
    Returns:
        matplotlib.figure.Figure: Figure containing the visualization
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Parse steps if it's a single string
    if isinstance(reasoning_steps, str):
        steps = parse_reasoning_steps(reasoning_steps)
    else:
        # Process list of intermediate states to extract steps
        steps = []
        for i, state in enumerate(reasoning_steps):
            if i > 0:  # Skip the prompt
                # Extract the new content from this state
                new_content = state[len(reasoning_steps[i-1]):].strip()
                if new_content:
                    steps.append(new_content)
    
    # Add nodes for each step
    for i, step in enumerate(steps):
        # Truncate long steps for display
        display_text = step[:50] + "..." if len(step) > 50 else step
        G.add_node(i, text=display_text, full_text=step)
    
    # Add edges connecting sequential steps
    for i in range(len(steps)-1):
        G.add_edge(i, i+1)
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Use a hierarchical layout
    pos = nx.spring_layout(G)
    
    # Draw nodes and edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, width=2, edge_color='gray')
    
    # Draw nodes with custom appearance
    node_colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(G)))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, labels={i: f"Step {i+1}" for i in G.nodes()})
    
    # Add step text as separate annotations
    for node, (x, y) in pos.items():
        plt.text(x, y-0.08, G.nodes[node]['text'],
                 horizontalalignment='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.title("Chain-of-Thought Reasoning Process")
    plt.axis('off')
    
    return plt.gcf()

def setup_cot_visualization_ui():
    """
    Set up the chain-of-thought visualization UI components in Streamlit.
    
    Returns:
        str: User prompt for chain-of-thought reasoning
    """
    st.subheader("Chain-of-Thought Visualization")
    
    # Example problems
    examples = [
        "What is the sum of the first 10 prime numbers?",
        "If a shirt originally costs $25 and is on sale for 20% off, and I have a coupon for an additional 10% off, how much will I pay?",
        "How many different ways can you arrange the letters in the word 'MATH'?",
        "What is the probability of rolling a sum of 7 with two fair dice?"
    ]
    
    example_selected = st.selectbox("Select an example problem", [""] + examples)
    
    if example_selected:
        user_prompt = st.text_area("Problem", value=example_selected, height=100)
    else:
        user_prompt = st.text_area("Enter a problem for step-by-step reasoning", height=100)
    
    return user_prompt if user_prompt else None

def visualize_chain_of_thought(prompt, model, tokenizer):
    """
    Create an interactive visualization of model reasoning steps.
    
    Args:
        prompt (str): Problem prompt
        model: Language model
        tokenizer: Tokenizer for the model
    
    Returns:
        tuple: Generated reasoning and visualization figure
    """
    # Generate reasoning with intermediate state tracking
    with st.spinner("Generating chain-of-thought reasoning..."):
        reasoning_steps = stream_generate_with_tracking(prompt, model, tokenizer)
    
    # Get the full reasoning text (final state)
    full_reasoning = reasoning_steps[-1]
    
    # Create the visualization
    with st.spinner("Creating reasoning visualization..."):
        fig = create_reasoning_graph(reasoning_steps)
    
    return full_reasoning, fig

def display_cot_results(reasoning, fig):
    """
    Display the chain-of-thought reasoning results in Streamlit.
    
    Args:
        reasoning (str): Generated reasoning text
        fig: Matplotlib figure with visualization
    """
    # Display the reasoning text
    st.subheader("Generated Reasoning")
    st.markdown(reasoning)
    
    # Display the visualization
    st.subheader("Reasoning Flow Visualization")
    st.pyplot(fig)
    
    # Explanation
    st.markdown("""
    This visualization shows how the model builds its reasoning step by step. 
    Each node represents a key step in the problem-solving process. 
    The flow of the arrows shows the progression of thought.
    """)