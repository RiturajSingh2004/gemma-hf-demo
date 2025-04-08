import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import streamlit as st

def load_vision_model():
    """
    Load the CLIP vision model for image processing.
    
    Returns:
        tuple: CLIP model and processor
    """
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

def process_image_input(image_file, gemma_model, tokenizer, clip_model, clip_processor):
    """
    Process image inputs to generate textual descriptions or answers using
    CLIP for image encoding and Gemma for text generation.
    
    Args:
        image_file: Uploaded image file
        gemma_model: Loaded Gemma model
        tokenizer: Gemma tokenizer
        clip_model: CLIP vision model
        clip_processor: CLIP processor
    
    Returns:
        str: Generated text based on the image
    """
    # Process the image with CLIP
    image = Image.open(image_file)
    inputs = clip_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        # Extract image features using CLIP
        image_features = clip_model.get_image_features(inputs.pixel_values)
        
        # Create a textual prompt that includes image context
        prompt = "Describe this image in detail:"
        
        # Tokenize the prompt
        text_inputs = tokenizer(prompt, return_tensors="pt").to(gemma_model.device)
        
        # Generate text based on the prompt
        outputs = gemma_model.generate(
            **text_inputs,
            max_length=200,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
    return response

def setup_multimodal_ui():
    """
    Set up the multimodal UI components in Streamlit.
    
    Returns:
        tuple: Uploaded image and selected task
    """
    st.subheader("Multimodal Processing")
    
    # Image upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Task selection
        task = st.selectbox(
            "Select task",
            options=["Image Description", "Question Answering", "Code Generation from Mockup"],
            index=0
        )
        
        if task == "Question Answering":
            st.text_input("Ask a question about the image")
        
        return uploaded_image, task
    
    return None, None

def generate_from_mockup(image_file, gemma_model, tokenizer, clip_model, clip_processor):
    """
    Generate UI code from a mockup image.
    
    Args:
        image_file: Uploaded mockup image
        gemma_model: Loaded Gemma model
        tokenizer: Gemma tokenizer
        clip_model: CLIP vision model
        clip_processor: CLIP processor
    
    Returns:
        str: Generated code based on the mockup
    """
    # Process the image with CLIP
    image = Image.open(image_file)
    inputs = clip_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        # Extract image features using CLIP
        image_features = clip_model.get_image_features(inputs.pixel_values)
        
        # Create a textual prompt for code generation
        prompt = "Generate HTML and CSS code for a webpage that looks like this UI mockup:"
        
        # Tokenize the prompt
        text_inputs = tokenizer(prompt, return_tensors="pt").to(gemma_model.device)
        
        # Generate code based on the prompt
        outputs = gemma_model.generate(
            **text_inputs,
            max_length=500,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = generated_text[len(prompt):].strip()
        
    return code