import os
import clip
import torch

# Define the class names for the CT-RATE dataset
CT_RATE_CLASSES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening"
]

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load('ViT-B/32', device)

# Create text prompts
text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in CT_RATE_CLASSES]).to(device)

# Calculate text embedding features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(f"Generated text features with shape: {text_features.shape} and dtype: {text_features.dtype}")
    
    # Ensure the output directory exists
    output_dir = 'Text-emmbedding-gen'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the embeddings
    output_path = os.path.join(output_dir, 'CT_RATE_clip_txt_encoding.pth')
    torch.save(text_features, output_path)
    print(f"Saved CT-RATE text embeddings to {output_path}")
