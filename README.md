# Satellite Image Classification with ResNet-9

[🚀 Launch Live Web Application](https://01a179d9-c8f7-3e81-c784-6d3bca47eaa6.share.connect.posit.cloud/)

An end-to-end deep learning pipeline that classifies satellite terrain images into 10 distinct categories (Forest, Highway, Industrial, etc.) using a custom PyTorch ResNet-9 architecture. The model achieves **96% accuracy** on a blind test set and is deployed as a low-latency, interactive web application via Posit Connect Cloud.

## Technical Highlights

- **Cloud Deployment:** Packaged the inference model into an interactive web UI using Shiny for Python, with automated deployment and CPU-optimized inference.
- **Model Architecture:** Built a ResNet-9 CNN from scratch with custom residual blocks and batch normalization.
- **Training Strategy:** Used the AdamW optimizer, OneCycleLR scheduler, and Early Stopping for stable, efficient convergence.
- **Data Engineering:** Implemented a RAM-caching PyTorch Dataset to eliminate I/O bottlenecks during training.
- **Exploratory Data Analysis (EDA):** Analyzed pixel distributions and brightness variance to inform targeted data augmentation strategies.

## How to Run the Web App Locally

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/Bayim7/Satellite-image-classification
   cd Satellite-image-classification
   ```

2. Create a virtual environment and install the required dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Launch the Shiny application:

   ```bash
   shiny run --reload --launch-browser --port=0 app.py
   ```
