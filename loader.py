import os, json, torch, urllib.request as url, zipfile, random
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets import load_dataset
from tqdm import tqdm

def get_flickr_data(max_samples=100000, val_split=0.2, batch_size=16):
    # Download images (Flickr8k)
    if not os.path.exists("data/images"):
        print("📸 Downloading Flickr8k images (1GB)...")
        os.makedirs("data", exist_ok=True)
        url.urlretrieve("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", "data/images.zip")
        with zipfile.ZipFile("data/images.zip") as z:
            z.extractall("data") # Extracts to data/Flicker8k_Dataset
        os.rename("data/Flicker8k_Dataset", "data/images")
        os.remove("data/images.zip")

    # Download and process captions (Flickr8k)
    if not os.path.exists("data/captions.json"):
        print("📂 Downloading Flickr8k captions...")
        url.urlretrieve("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip", "data/captions.zip")
        with zipfile.ZipFile("data/captions.zip") as z:
            z.extractall("data")
        
        captions_file = "data/Flickr8k.token.txt"
        captions_data = []
        with open(captions_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_id, caption = parts[0][:-2], parts[1] # "image.jpg#0" -> "image.jpg"
                    captions_data.append({'image': image_id, 'caption': caption})
        
        json.dump(captions_data, open("data/captions.json", "w"))
        # Clean up
        os.remove("data/captions.zip")
        os.remove(captions_file)
        for f in ["Flickr_8k.trainImages.txt", "Flickr_8k.devImages.txt", "Flickr_8k.testImages.txt", "readme.txt"]:
            if os.path.exists(f"data/{f}"): os.remove(f"data/{f}")

    # Load and split
    data = json.load(open("data/captions.json"))[:max_samples]
    random.shuffle(data)
    split = int(len(data) * (1 - val_split))
    train_data, val_data = data[:split], data[split:]
    
    # Just resize PIL images - let each model handle its own preprocessing
    resize_transform = Resize((224, 224))
    
    # Batch generators
    def batches(data):
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            # Load and resize PIL images once
            pil_images = []
            for item in batch:
                if os.path.exists(f"data/images/{item['image']}"):
                    img = Image.open(f"data/images/{item['image']}").convert('RGB')
                    pil_images.append(resize_transform(img))
                else:
                    pil_images.append(Image.new('RGB', (224, 224)))  # Empty image fallback
            
            captions = [item['caption'] for item in batch]
            yield pil_images, captions
    
    return lambda: batches(train_data), lambda: batches(val_data)

def get_disco_data(max_samples=100000, val_split=1, batch_size=16):
    # Download and load the dataset from HuggingFace (e.g., "ntkuhn/mlx_dropouts_images")
    ds = load_dataset("ntkuhn/mlx_dropouts_images")
    data = []
    disco_dir = "data/disco"
    os.makedirs(disco_dir, exist_ok=True)
    images_dir = os.path.join(disco_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    captions_file = os.path.join(disco_dir, "captions.json")
    # Use tqdm for download/saving progress
    dataset = ds['train']
    for idx, item in enumerate(tqdm(dataset, desc="Downloading & saving images", total=min(len(dataset), max_samples))):
        if "image" in item and "description" in item:
            img = item['image']
            img_filename = f"{idx}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            if isinstance(img, Image.Image):
                img.save(img_path)
            else:
                Image.open(img).convert('RGB').save(img_path)
            data.append({'image': img_filename, 'caption': item['description']})
        if len(data) >= max_samples:
            break

    # Save captions to JSON
    with tqdm(total=1, desc="Saving captions") as pbar:
        with open(captions_file, "w") as f:
            json.dump(data, f)
        pbar.update(1)

    random.shuffle(data)
    split = int(len(data) * (1 - val_split))
    train_data, val_data = data[:split], data[split:]

    resize_transform = Resize((224, 224))

    def batches(data):
        for i in tqdm(range(0, len(data), batch_size), desc="Batching"):
            batch = data[i:i+batch_size]
            pil_images = []
            for item in batch:
                img_path = os.path.join(images_dir, item['image'])
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    pil_images.append(resize_transform(img))
                else:
                    pil_images.append(Image.new('RGB', (224, 224)))
            captions = [item['caption'] for item in batch]
            yield pil_images, captions

    return lambda: batches(train_data), lambda: batches(val_data)

get_disco_data = get_disco_data(max_samples=100000, val_split=1, batch_size=16)