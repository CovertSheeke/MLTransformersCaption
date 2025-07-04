import torch
from model_self_attention import VisionLanguageEncoder, CaptionDecoder
from tqdm import tqdm
import itertools
from PIL import Image
from torchvision import transforms
import os
import json
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## import weights from saved_models/self_attention/model_checkpoint.pth
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    encoder = VisionLanguageEncoder()
    decoder = CaptionDecoder()

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Move models to device
    encoder.to(device)
    decoder.to(device)

    return encoder, decoder

encoder, decoder = load_model('saved_models/self_attention/model_checkpoint.pth')
# Set the model to evaluation mode
encoder.eval()
decoder.eval()

# Example input
image = torch.randn(1, 3, 224, 224, device=device)  # Dummy
target_caption = 'image of random noise'

# # List of image file paths
# image_paths = [
#     "syntetic_data/667626_18933d713e.jpg",
#     "syntetic_data/3637013_c675de7705.jpg",
#     "syntetic_data/10815824_2997e03d76.jpg",
#     "syntetic_data/12830823_87d2654e31.jpg"
# ]

# true_captions = [
#     "A child lies in the water.",
#     "People stand in a park near a pond.",
#     "A horse and its young minder look at a burning tree trunk.",
#     "A crowd of adults and children stand near an artifical fish pond."
# ]

# Load captions from JSON
with open("data/disco/captions.json", "r") as f:
    captions_data = json.load(f)

# Get image paths and true captions
image_dir = "data/disco/images"
image_paths = [os.path.join(image_dir, item["image"]) for item in captions_data]
true_captions = [item["caption"] for item in captions_data]

# Load images as PIL Images, convert to tensors, and move to device
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
pil_images = [Image.open(path).convert("RGB") for path in image_paths]
tensor_images = [preprocess(img).unsqueeze(0).to(device) for img in pil_images]

# Structure as list of (image, caption) pairs
val_data = list(zip(tensor_images, true_captions))

# Create batches (here, batch size 1 for simplicity)
val_batches = [[(img, cap)] for img, cap in val_data]

# Store predicted captions for plotting
predicted_captions = []

with torch.no_grad():
    print("Starting validation batches...")
    for i, batch in tqdm(enumerate(val_batches), desc="Batches"):
        # Unpack batch (batch size 1)
        first_image, true_caption = batch[0]
        
        # --- Image Processing (via Encoder) ---
        print(f"\nProcessing image {i+1}...")
        combined_embeddings, _, num_patches = encoder(first_image, ["dummy caption"])
        print("Combined embeddings shape:", combined_embeddings.shape)
        print("Num patches:", num_patches)
        
        # Extract just the image part for generation
        image_embeddings = combined_embeddings[:, :num_patches, :]
        print("Image embeddings shape:", image_embeddings.shape)
        
        # --- Autoregressive Generation ---
        generated_ids = []
        # Start with the Beginning-Of-Sequence token ID
        sos_id = decoder.tokenizer.bos_token_id if decoder.tokenizer.bos_token_id is not None else decoder.tokenizer.eos_token_id
        input_ids = torch.tensor([[sos_id]], dtype=torch.long, device=device)
        print("Starting autoregressive generation...")

        for step in tqdm(range(30), desc="Generating tokens", leave=False):
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
            
            # --- Top-k Sampling (same as explained model) ---
            k = 50
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
            next_token_relative_idx = torch.multinomial(probs, num_samples=1)
            next_token_id = torch.gather(top_k_indices, -1, next_token_relative_idx)
            
            # Debug prints for generation
            print(f"Step {step+1}: Next token id: {next_token_id.item()}")

            # Stop if EOS is generated
            if next_token_id.item() == decoder.tokenizer.eos_token_id:
                print("EOS token generated, stopping.")
                break
            
            generated_ids.append(next_token_id.item())
            
            # Append the new token for the next iteration
            input_ids = torch.cat([input_ids, next_token_id.to(device)], dim=1)

        predicted_caption = decoder.tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_captions.append(predicted_caption)
        
        print(f"\nImage {i+1}:")
        print(f"  True: {true_captions[i]}")
        print(f"  Pred: {predicted_caption}")
        
print("\nDone!")


# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Show images with predicted and true captions, and save the plot
fig, axes = plt.subplots(1, len(val_data), figsize=(20, 8))
for idx, (img_tensor, true_caption) in enumerate(val_data):
    img = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    axes[idx].imshow(img)
    axes[idx].axis('off')
    
    # Include both true and predicted captions in the title
    pred_caption = predicted_captions[idx] if idx < len(predicted_captions) else "No prediction"
    axes[idx].set_title(f"True: {true_caption}\n\nPred: {pred_caption}", fontsize=8, wrap=True)
plt.tight_layout(pad=3.0)
plt.savefig("results/captions_results.png", bbox_inches='tight', dpi=300)
plt.close()