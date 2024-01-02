from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, DataLoader
import os

# Load the pretrained Vision Transformer model and its feature extractor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Define a custom dataset class for image classification
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, feature_extractor):
        self.img_dir = img_dir
        self.feature_extractor = feature_extractor

        # Collect image paths and their corresponding labels
        self.img_paths = []
        self.labels = []
        for label, class_dir in enumerate(['cat', 'dog']):
            class_img_paths = [os.path.join(img_dir, class_dir, filename) 
                               for filename in os.listdir(os.path.join(img_dir, class_dir))]
            self.img_paths.extend(class_img_paths)
            self.labels.extend([label] * len(class_img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Open and convert the image to RGB
        features = self.feature_extractor(images=image, return_tensors="pt")  # Apply preprocessing
        return features['pixel_values'].squeeze(), self.labels[idx]  # Return processed pixel values and labels

# Instantiate the dataset and dataloader
img_dir = 'path_to/real_test_data'  # Path to your dataset
dataset = CustomImageDataset(img_dir=img_dir, feature_extractor=feature_extractor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Define a mapping from class indices to human-readable labels
label_map = {0: "cat", 1: "dog"}

# Switch the model to evaluation mode
model.eval()

# Disable gradient calculations for inference
with torch.no_grad():  
    for inputs, label in dataloader:
        outputs = model(inputs)  # Obtain model predictions
        logits = outputs.logits  # Extract logits
        predicted_class_idx = logits.argmax(-1).item()  # Determine the most likely class
        print(f"Actual label: {label_map[label.item()]}, Predicted label: {label_map[predicted_class_idx]}")
