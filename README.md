# Thangka Image Segmentation
This is the official code repository for the paper "Research on the Semantic Segmentation of Thangka Images via an Improved PIDNet".

## 📌 Data & Model Weights Availability
The minimal anonymized dataset and pre-trained model weights required to replicate the study findings are openly available in Zenodo:
**DOI: 10.5281/zenodo.19496600**
https://zenodo.org/doi/10.5281/zenodo.19496600

## 🛠️ Environment
- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- opencv-python
- pillow
- tqdm
- scipy
- matplotlib
- mmcv-full
- openmim

## 🚀 Installation
```bash
conda create -n thangka-seg python=3.8
conda activate thangka-seg
pip install torch torchvision
pip install numpy opencv-python pillow tqdm scipy matplotlib
pip install -U openmim
mim install mmcv-full
