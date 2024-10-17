# SAM2 Image Segmentation Tool

A GUI application for image segmentation using the **SAM2 (Segment Anything Model 2)** model, implemented in Python using PyQt. This tool allows you to upload images, annotate them with points and lines as prompts, and generate masks using the SAM2 model.

## Features

- **Upload Images**: Supports common image formats (PNG, JPG, JPEG, BMP).
- **Interactive Annotation**:
  - Add **foreground points** (`Ctrl + Click`).
  - Add **background points** (`Shift + Click`).
  - **Draw lines** as prompts.
- **Mask Generation**: Generate segmentation masks using the SAM2 model.
- **Real-time Mask Updates**: Masks update in real-time as you add annotations.
- **Undo/Redo Functionality**: Supports undoing and redoing of actions.
- **Multiple Model Sizes**: Choose between Tiny, Small, Base+, and Large SAM2 models.
- **Zooming and Panning**: Use the mouse wheel to zoom in/out.

## Requirements

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/) (with CUDA support for GPU acceleration, optional)
- PyQt5 or PyQt6
- NumPy
- Pillow (PIL)
- scikit-image
- **SAM2 modules** (from the `sam2` package)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sam2-gui.git
cd sam2-gui
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can install the dependencies manually:

```bash
pip install numpy pillow scikit-image
pip install torch torchvision
pip install PyQt5  # or PyQt6 if you prefer
```

**Note**: If you intend to use GPU acceleration, ensure that PyTorch is installed with the appropriate CUDA version. Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for installation instructions tailored to your system.

### 4. Install PyQt

Depending on your preference, install PyQt5 or PyQt6:

```bash
# For PyQt5
pip install PyQt5

# For PyQt6
pip install PyQt6
```

### 5. Download SAM2 Model Checkpoints and Configs

Download the SAM2 model checkpoints and configuration files:

- **Tiny Model**:
  - Checkpoint: `sam2.1_hiera_tiny.pt`
  - Config: `sam2.1_hiera_t.yaml`
- **Small Model**:
  - Checkpoint: `sam2.1_hiera_small.pt`
  - Config: `sam2.1_hiera_s.yaml`
- **Base+ Model** (Default):
  - Checkpoint: `sam2.1_hiera_base_plus.pt`
  - Config: `sam2.1_hiera_b+.yaml`
- **Large Model**:
  - Checkpoint: `sam2.1_hiera_large.pt`
  - Config: `sam2.1_hiera_l.yaml`

**Directory Structure**:

- Place the checkpoint files in the `checkpoints/` directory.
- Place the config files in the `sam2/configs/sam2.1/` directory.

Ensure that the directory structure matches the paths used in the code. If these directories don't exist, create them.

### 6. Ensure the `sam2` Package is Available

The application depends on the `sam2` package. Ensure that:

- The `sam2` package is installed in your Python environment.
- **OR** the `sam2` directory (containing `__init__.py`, `build_sam.py`, etc.) is located in the same directory as your script or is accessible via your Python path.

### 7. Verify All Dependencies

Ensure all required packages are installed:

- `numpy`
- `torch`
- `Pillow` (PIL)
- `PyQt5` or `PyQt6`
- `scikit-image`

## Usage

### Run the Application

```bash
python app.py
```

### Interface Overview

- **Upload Image**: Click to select and upload an image.
- **Draw Line**: Toggle line drawing mode to draw lines as prompts.
- **Remove Point**: Remove selected points from the image.
- **Remove Line**: Remove selected lines from the image.
- **Generate Mask**: Generate segmentation masks based on the current annotations.
- **Remove Mask**: Remove generated masks from the image.
- **Undo/Redo**: Undo or redo the last action.
- **Clear Annotations**: Remove all annotations (points, lines, masks) from the image.
- **Model Type**: Select the SAM2 model type (Tiny, Small, Base+, Large).
- **Batch Size**: Adjust the batch size for mask generation.
- **Enable Multimask Output**: Toggle between single and multiple mask outputs.
- **Instructions**: Displays helpful tips for using the application.

### Annotation Instructions

- **Add Foreground Point**: Hold `Ctrl` and click on the desired location in the image.
- **Add Background Point**: Hold `Shift` and click on the desired location in the image.
- **Draw Line**:
  - Click the `Draw Line` button to enable line drawing mode.
  - Click on the image to set the start point of the line.
  - Click again to set the end point of the line.
- **Zoom In/Out**: Use the mouse wheel to zoom in and out.
- **Right-click on a Point**: Opens a context menu to:
  - **Delete Point**: Remove the point from the image.
  - **Toggle Point Label**: Change the point from foreground to background or vice versa.

### Generating Masks

After adding annotations (points and/or lines):

1. Click on the `Generate Mask` button.
2. The application will process the image and display the generated mask(s) over the image.
3. Masks can be removed using the `Remove Mask` button.

### Real-time Mask Updates

- The application supports real-time mask updates as you add or modify annotations.
- This feature is automatically triggered when you add or remove points and lines.

## Troubleshooting

- **Model Files Not Found**: Ensure that the SAM2 model checkpoint and config files are placed in the correct directories (`checkpoints/` and `sam2/configs/sam2.1/`).
- **Dependencies Errors**: Verify that all required Python packages are installed in your environment.
- **GPU Support**:
  - If you have a CUDA-compatible GPU, ensure that PyTorch is installed with CUDA support.
  - On macOS with M1/M2 chips, the application will attempt to use the Metal Performance Shaders (MPS) backend.
  - If no GPU is available, the application will default to CPU mode, which may be slower.
- **Pillow Version**:
  - Ensure you have a recent version of Pillow installed.
  - The application uses `Image.Resampling.LANCZOS` for image resizing.
  - If you encounter issues, try updating Pillow:
    ```bash
    pip install --upgrade Pillow
    ```

## Acknowledgments

- **SAM2**: This application utilizes the [Segment Anything Model 2 (SAM2)](https://github.com/yourusername/sam2) for image segmentation.
- **PyQt**: The GUI is built using [PyQt](https://www.riverbankcomputing.com/software/pyqt/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: Replace `https://github.com/yourusername/sam2-gui.git` and other placeholder URLs with the actual URLs relevant to your project. Ensure that the `LICENSE` file is included if you are distributing the project.