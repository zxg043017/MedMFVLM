import os
import clip
import torch


## PAOT
ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Arota', 'Postcava', 'Portal Vein and Splenic Vein',
                'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
                'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 
                'Colon Tumor', 'Kidney Cyst']

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load('ViT-B/32', device)


text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in ORGAN_NAME]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(f"Generated text features with shape: {text_features.shape} and dtype: {text_features.dtype}")
    
    # Ensure the output directory exists
    output_dir = 'Text-emmbedding-gen'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the embeddings to the correct path for the main script
    output_path = os.path.join(output_dir, 'TAO_clip_txt_encoding.pth')
    torch.save(text_features, output_path)
    print(f"Saved text embeddings to {output_path}")
