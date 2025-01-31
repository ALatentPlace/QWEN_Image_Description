# QWEN Image Description

A graphical user interface (GUI) application designed to generate descriptions of images using pre-trained AI models (Qwen2-VL-7B-Instruct).

## Project Overview

This project provides a GUI tool that allows users to:
- Select an image folder to be processed
- Generate detailed text descriptions of the contents

## Dependencies

To install this project, you will need Python 3.7 or higher and the following packages:

### Required Libraries:
```bash
pip install -r requirements.txt
```

## Usage

1. **Installation**:
   - Ensure Python is installed on your system.
   - Clone this repository.
   - Create a virtual environment and activate it.
   - Install torch:
      ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - Install dependencies using pip:
     ```bash
     pip install -r requirements.txt
     ```

1. **Running the Application**:
   - Run `start.bat` (Windows) or execute `app_gui.py`.

2. **Selecting an Image Folder**:
   - Open the application.
   - Select the folder containing your images.

3. **Generating Descriptions**:
   - Enter a trigger word, which will be placed in front of the description (for LoRA training purposes).
   - Enter a prompt like "Please describe the image in every detail". You last prompt will be save in a file for later usage.
   - The application will process each image and generate a detailed text description. Processed images and their descriptions will be moved to a "processed" folder. This allows you to easily continue with unhandled files if you stop the process.

##Note: First generation can take up several minutes. After that it will be faster. ##


## Features

- Basic image upload and processing interface.
- Text description generation using a pre-trained model.
- Batch processing of multiple images.

## Note

This project requires an internet connection to download the base AI model (Hugging Face Spaces model). The application itself is lightweight but depends on external models being available. It is a fun project, so support is not guaranteed.

## Contributing

Contributions are welcome! You can fork the repository and submit a pull request with improvements or new features. Please ensure you reference relevant files using their full path in the filenames.