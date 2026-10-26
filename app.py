# Name: Ibrahim Muhammad

from shiny.express import ui, render, input
from shiny import reactive
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

#---
#MODEL 
#----


#architecture used to train the model so PyTorch can load the weights
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), 
        nn.BatchNorm2d(out_channels), 
        nn.ReLU(inplace=True)
    ]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class SatelliteResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.prep = conv_block(in_channels, 64)                         
        self.layer1 = conv_block(64, 128, pool=True)                    
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))                                                               
        self.layer2 = conv_block(128, 256, pool=True)                   
        self.layer3 = conv_block(256, 512, pool=True)                   
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))                                                               
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                                    
            nn.Flatten(), 
            nn.Dropout(0.2),                                            
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = out + self.res1(out) 
        out = self.layer2(out)
        out = self.layer3(out)     
        out = out + self.res2(out) 
        out = self.classifier(out)
        return out

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]

# Validation transforms ONLY (we should not augment images for inference)
img_transforms = transforms.Compose([
    transforms.Resize((64, 64)), # Ensure any uploaded image is resized to 64x64
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEVICE = torch.device("cpu") #Not everyone has CUDA:)
MODEL_PATH = "best_satellite_cnn.pth"

# Initialize model and load weights
model = SatelliteResNet9(in_channels=3, num_classes=10).to(DEVICE)


if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    _ = model.load_state_dict(state_dict)
    _ = model.eval()
    model_loaded = True
else:
    model_loaded = False


#----
# 2. SHINY EXPRESS UI 
#-----

ui.page_opts(title="Satellite Image Classifier", fillable=True)

with ui.sidebar():
    ui.h4("Upload Image")
    ui.input_file("img_upload", "Select satellite image", accept=[".jpg", ".jpeg", ".png"])
    
    if not model_loaded:
        ui.p("⚠️ Error: 'best_satellite_cnn.pth' not found in the app directory!", style="color: red; font-weight: bold;")

with ui.layout_columns():
    
    # Left Column: Display Image
    with ui.card():
        ui.card_header("Uploaded Image")
        
        @render.image
        def display_image():
            file_info = input.img_upload()
            if not file_info:
                return None
            return {"src": file_info[0]["datapath"], "width": "100%", "style": "max-width: 400px; border-radius: 8px;"}

    # Right Column: Top Prediction
    with ui.card(class_="bg-primary text-white"):
        ui.card_header("Prediction Results")
        
        @render.ui
        def display_prediction():
            results = get_predictions()
            if not results:
                return ui.p("Please upload an image to see the prediction.")
            
            return ui.div(
                ui.h2(f"{results['top_class']}", style="margin-bottom: 5px;"),
                ui.h5(f"Confidence: {results['top_confidence']:.2f}%", style="opacity: 0.9;"),
                style="text-align: center; padding: 20px;"
            )

# Bottom Panel: Probability Table
with ui.card():
    ui.card_header("Class Probabilities")
    
    @render.table
    def probability_table():
        results = get_predictions()
        if not results:
            return pd.DataFrame({"Class": [], "Probability": []})
        return results["prob_df"]



@reactive.calc
def get_predictions():
    file_info = input.img_upload()
    if not file_info or not model_loaded:
        return None
    
    img_path = file_info[0]["datapath"]
    
    try:
        # 1. Open and Transform
        img = Image.open(img_path).convert("RGB")
        img_tensor = img_transforms(img).unsqueeze(0).to(DEVICE) 
        
        # 2. Inference
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
        
        # 3. Process Results
        prob_dict = {CLASS_NAMES[i]: float(probabilities[i] * 100) for i in range(10)}
        
        # Sort by highest probability
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_class, top_prob = sorted_probs[0]
        
        # Create DataFrame for the table
        df = pd.DataFrame(sorted_probs, columns=["Class", "Probability (%)"])
        df["Probability (%)"] = df["Probability (%)"].apply(lambda x: f"{x:.2f}%")
        
        return {
            "top_class": top_class,
            "top_confidence": top_prob,
            "prob_df": df
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None