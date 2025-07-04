import torch
from model_self_attention import VisionLanguageEncoder, CaptionDecoder
from tqdm import tqdm
import itertools
from PIL import Image
from torchvision import transforms
import os
import json
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global model variables
encoder = None
decoder = None

def load_model(model_path='saved_models/self_attention/model_checkpoint.pth'):
    """Load the pre-trained model"""
    global encoder, decoder
    
    checkpoint = torch.load(model_path, map_location='cpu')
    encoder = VisionLanguageEncoder()
    decoder = CaptionDecoder()

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Move models to device
    encoder.to(device)
    decoder.to(device)
    
    # Set the model to evaluation mode
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder

def generate_caption_for_image(image_path, true_caption=None, max_tokens=30, top_k=50):
    """
    Generate a caption for a single image
    
    Args:
        image_path: Path to the image file or PIL Image object
        true_caption: Optional true caption for comparison
        max_tokens: Maximum number of tokens to generate
        top_k: Top-k sampling parameter
        
    Returns:
        tuple: (predicted_caption, matplotlib_figure)
    """
    global encoder, decoder
    
    # Load models if not already loaded
    if encoder is None or decoder is None:
        load_model()
    
    # Load and preprocess image
    if isinstance(image_path, str):
        pil_image = Image.open(image_path).convert("RGB")
    else:
        pil_image = image_path  # Assume it's already a PIL Image
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # --- Image Processing (via Encoder) ---
        combined_embeddings, _, num_patches = encoder(image_tensor, ["dummy caption"])
        
        # Extract just the image part for generation
        image_embeddings = combined_embeddings[:, :num_patches, :]
        
        # --- Autoregressive Generation ---
        generated_ids = []
        # Start with the Beginning-Of-Sequence token ID
        sos_id = decoder.tokenizer.bos_token_id if decoder.tokenizer.bos_token_id is not None else decoder.tokenizer.eos_token_id
        input_ids = torch.tensor([[sos_id]], dtype=torch.long, device=device)

        for step in range(max_tokens):
            # Get text embeddings for the current sequence
            text_embeddings = decoder.qwen_model.get_input_embeddings()(input_ids)
            
            # Add text modality embedding
            text_mod_id = torch.ones_like(input_ids, device=device)
            text_mod_emb = encoder.modality_embedding(text_mod_id)
            text_embeddings_final = text_embeddings + text_mod_emb
            
            # Combine image and text embeddings
            combined_for_generation = torch.cat([image_embeddings, text_embeddings_final], dim=1)
            
            # Add enhanced positional embeddings and layer norm
            batch_size, seq_len, hidden_size = combined_for_generation.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            pos_embeddings = decoder.enhanced_pos_embedding(position_ids)
            enhanced_embeddings = combined_for_generation + pos_embeddings
            enhanced_embeddings = decoder.input_layer_norm(enhanced_embeddings)

            # Get logits using the self-attention decoder
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                outputs = decoder.qwen_model(inputs_embeds=enhanced_embeddings)
                logits = outputs.logits
            
            # Get the logit for the last token
            next_token_logits = logits[:, -1, :]
            
            # --- Top-k Sampling ---
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
            next_token_relative_idx = torch.multinomial(probs, num_samples=1)
            next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)

            # Stop if EOS is generated
            if next_token_id.item() == decoder.tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token_id.item())
            
            # Append the new token for the next iteration
            input_ids = torch.cat([input_ids, next_token_id.to(device)], dim=1)

        predicted_caption = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Create the plot
        fig = create_caption_plot(pil_image, predicted_caption, true_caption)
        
        return predicted_caption, fig

def create_caption_plot(pil_image, predicted_caption, true_caption=None):
    """
    Create a matplotlib plot showing the image with captions
    
    Args:
        pil_image: PIL Image object
        predicted_caption: Generated caption
        true_caption: Optional true caption for comparison
        
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Convert PIL image to numpy array for display
    img_array = np.array(pil_image)
    ax.imshow(img_array)
    ax.axis('off')
    
    # Create title with captions
    if true_caption:
        title = f"True: {true_caption}\n\nPredicted: {predicted_caption}"
    else:
        title = f"Predicted: {predicted_caption}"
    
    ax.set_title(title, fontsize=12, wrap=True, pad=20)
    plt.tight_layout(pad=3.0)
    
    return fig

# Example usage and testing
if __name__ == "__main__":
    # Load models at startup for faster inference
    print("Loading models...")
    load_model()
    print("Models loaded successfully!")
    
    # Load a single image for testing
    test_image_path = "syntetic_data/667626_18933d713e.jpg"
    test_true_caption = "A child lies in the water."
    
    # Generate caption for the image (models are already loaded)
    print("Generating caption...")
    predicted_caption, fig = generate_caption_for_image(
        test_image_path, 
        true_caption=test_true_caption
    )
    
    print(f"Predicted caption: {predicted_caption}")
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/single_image_caption.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print("Caption generation complete! Result saved to results/single_image_caption.png")