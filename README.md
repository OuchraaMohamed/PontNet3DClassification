
# PointNet 3D Point Cloud Classification

This repository contains a Jupyter Notebook implementing 3D point cloud classification using PointNet. PointNet is a deep learning framework designed for tasks like object classification and segmentation directly on point clouds, which are widely used in 3D computer vision.

## Project Overview

The notebook demonstrates:
- Preprocessing of 3D point cloud data.
- Implementation of the PointNet architecture for classification.
- Training and evaluation of the model on a 3D dataset.

Key features include:
- A step-by-step explanation of the PointNet framework.
- Customization options for model parameters and training configurations.
- Visualization of results and 3D point cloud data.

## Requirements

To run the notebook, you need the following:
- Python 3.7+
- Jupyter Notebook or Jupyter Lab
- Required Python libraries:
  - `torch` (PyTorch)
  - `numpy`
  - `matplotlib`
  - `open3d` (optional, for advanced visualization)

Install the required dependencies using:

```bash
pip install torch numpy matplotlib open3d
```

## Dataset

The notebook uses a 3D point cloud dataset for training and testing. You can use public datasets like [ModelNet](https://modelnet.cs.princeton.edu/) or your own custom dataset.

Ensure the dataset is formatted as follows:
- Point clouds should be represented as `.off`, `.h5`, or similar formats with XYZ coordinates.
- Label files should match the point cloud files for supervised learning.

## Usage

1. Clone the repository and navigate to the project directory.
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook PointNetClassification_3D.ipynb
   ```
3. Follow the instructions in the notebook to preprocess data, configure the model, and run the training.

## Results

The notebook provides metrics like accuracy and loss curves for model evaluation. Visualization of the classified point clouds is also included.

## References

- **PointNet**: Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. [Paper](https://arxiv.org/abs/1612.00593)
- **PyTorch Documentation**: [PyTorch](https://pytorch.org/docs/stable/index.html)

## License

This project is for educational purposes and is provided "as-is". Please ensure compliance with dataset licenses when using external data.
