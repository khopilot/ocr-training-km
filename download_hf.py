
import os
os.system('pip install datasets pillow -q')

from datasets import load_dataset
from pathlib import Path
from PIL import Image

try:
    dataset = load_dataset(
        "SaladTechnologies/khmer-ocr-dataset",
        split="train[:5000]"
    )
    
    train_dir = Path("data/hf_train")
    train_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(dataset):
        if i >= 5000:
            break
        
        # Save image
        img = item.get('image')
        if img:
            img_path = train_dir / f"hf_{i:06d}.png"
            img.save(img_path)
            
            # Save text
            text = item.get('text', '')
            txt_path = img_path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
    
    print(f"Downloaded {i} samples to data/hf_train/")
    
except Exception as e:
    print(f"Could not download HF data: {e}")
