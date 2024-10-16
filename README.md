# Brain Tumor Diagnosis and Treatment with Patient Assistance Chatbot

This repository offers an AI-powered healthcare platform for **brain tumor classification**, **multi-modal tumor segmentation**, and **patient support** through an interactive chatbot. 

## Problem Statement

Brain tumors are life-threatening and require accurate diagnosis, but current solutions lack a unified platform for tumor classification, segmentation, and patient support. This results in fragmented workflows and delayed treatments. Our solution integrates all three components into a single AI-powered platform for faster, more reliable diagnostics and real-time patient assistance.
## Objective

The main goal of this project is to build an AI-powered platform that integrates:

- **Brain Tumor Classification**: Using a fine-tuned ResNet50 model to classify brain tumors into different types (e.g., gliomas, meningiomas) based on MRI scans.
- **Tumor Segmentation**: Employing a 3D U-Net model to segment brain tumors from MRI images, giving detailed insights into tumor size and shape.
- **Patient Assistance Chatbot**: A chatbot built on Retrieval-Augmented Generation (RAG) that helps patients with common queries, provides information on diagnosis and treatment, and offers 24/7 support.

## Methodology

### 1. Data Collection and Preprocessing
- **Data Source**: MRI brain scan datasets from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/).
- **Preprocessing**: MRI images undergo normalization, noise reduction, and data augmentation techniques to improve model robustness and performance.

### 2. Brain Tumor Classification
- **Model**: A pre-trained **ResNet50** model is fine-tuned using labeled brain MRI data.
- **Objective**: The classification helps medical professionals quickly identify tumor types (e.g., gliomas, meningiomas) from MRI scans.

### 3. Tumor Segmentation
- **Model**: A **3D U-Net** is used to segment tumor sub-regions across multiple views (axial, sagittal, coronal).
- **Outcome**: The segmented MRI scans provide a detailed 3D map of the tumor, allowing doctors to plan appropriate treatments.

### 4. Patient Assistance Chatbot
- **Model**: A **Retrieval-Augmented Generation (RAG)** chatbot is implemented to provide real-time responses to patient queries. It uses medical resources such as those from the **National Cancer Institute**.
- **Outcome**: The chatbot offers 24/7 patient support, providing valuable information on symptoms, diagnosis, treatments, and offering empathetic guidance.

## Installation

To run the project locally:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/prabhaM07/BRATS.git
    cd BRATS
    ```

2. **Install dependencies**:
    ```bash
    pip install opencv-python
    pip install numpy
    pip install matplotlib
    pip install seaborn
    pip install pandas
    pip install scikit-learn
    pip install transformers
    ```

3. **Run the application**:
    ```bash
    python app.py
    ```

## Demo

Watch a demo of the platform [here](https://drive.google.com/file/d/1APR4f2L3jOqr8wXgwP74jlxLPUVSNSSO/view?usp=sharing](https://drive.google.com/file/d/1DPjMdNO-ddxtxqQKfx6Lyx_0rTJgopVE/view?usp=sharing).

## Files Description

- **Analyser.py**: Script for analyzing input MRI data or model predictions.
- **app.py**: The main application file that serves the platform.
- **brain_tumor_pdf.pdf**: Documentation related to brain tumor analysis.
- **chatbot.py**: Code implementing the RAG-based chatbot for patient support.
- **classifier.py**: The classification logic based on the ResNet50 model.
- **main.py**: Core script for model initialization and system setup.
- **Demo_Video.mp4**: A demonstration video of the platform.

## Technologies Used

- **ResNet50**: For brain tumor classification.
- **3D U-Net**: For brain tumor segmentation.
- **RAG Chatbot**: For real-time patient support (Gemini 1.5).
- **Flask**: Backend framework for serving the web app.
- **PyTorch**: Deep learning framework for model development.
- **OpenCV**: Image processing.
- **Numpy, Pandas**: Data manipulation and analysis.

## Contributing

Contributions are welcome! If you'd like to improve the project, feel free to submit a pull request or raise an issue.
