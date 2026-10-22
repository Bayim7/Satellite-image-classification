# Satellite Image Classification with ResNet-9

This repository contains an end-to-end deep learning pipeline that classifies satellite terrain images into 10 distinct categories (Forest, Highway, Industrial, etc.) using a custom ResNet-9 architecture built in PyTorch. 

**Achieved 96% accuracy on a blind test set.**

## Project Highlights
* **Exploratory Data Analysis (EDA):** Analyzed pixel distributions and brightness variance to inform data augmentation strategies.
* **Custom Dataloader:** Implemented a RAM-caching PyTorch Dataset to eliminate I/O bottlenecks during training.
* **Model Architecture:** Built a ResNet-9 CNN from scratch with custom residual blocks and batch normalization.
* **Training Strategy:** Utilized `AdamW` optimizer, `OneCycleLR` scheduler, and Early Stopping.
* **Deployment:** Packaged the inference model into an interactive web application using Shiny for Python.

## How to Run the Web App Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/Bayim7/Satellite-image-classification] (https://github.com/Bayim7/Satellite-image-classification)
   cd satellite-image-classification