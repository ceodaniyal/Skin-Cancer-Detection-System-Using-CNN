# Skin Detection System using Convolutional Neural Networks (CNN)
This repository contains the code for a skin detection system developed using Convolutional Neural Networks (CNN). The project is divided into two main parts: training the model and using a GUI to make predictions with the trained model.

## Table of Contents
- Overview
- Features
- Prerequisites
- Installation
- Dataset
- Training the Model
- Running the GUI for Prediction
- Contributing
- License

## Overview
This project is designed to detect skin conditions using image data. The system is built using a Convolutional Neural Network (CNN) that is trained on labeled skin image data. The trained model can then be used to make predictions on new images through a graphical user interface (GUI).

## Features
- Training Script: A script to train the CNN model on a dataset of skin images and save the trained model.
- Prediction GUI: A GUI application that loads the saved model and allows users to make predictions on new images.
- Model Persistence: The trained model is saved to disk for reuse, eliminating the need to retrain the model every time the GUI is run.

## Prerequisites
Before you can run the code, ensure you have the following software and libraries installed:

- Python 3.7+
- TensorFlow 2.5+
- Keras 2.5+
- NumPy
- OpenCV
- Tkinter (for the GUI)
- Matplotlib (for visualization)
- Scikit-learn

## Installation
Follow these steps to set up your environment and run the code:

1. Clone the repository:

bash
`git clone https://github.com/your-username`
`skin-detection-cnn.git`
`cd skin-detection-cnn`


2. Create a virtual environment (optional but recommended):

bash
`python -m venv venv`
`source venv/bin/activate  # On Windows use` `venv\Scripts\activate`

3. Install the required packages:

bash
`pip install -r requirements.txt`

4. Download and prepare the dataset:

- Download the dataset from Kaggle: ![Melanoma Skin Cancer Dataset of 10000 Images.](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)
- Extract the dataset and place it in a directory named `data/` within the project folder.
- The dataset should be structured as follows:
bash
```
data/
├── train/
│   ├── benign/
│   ├── malignant/
│   └── ...
└── test/
    ├── benign/
    ├── malignant/
    └── ...
```
Note: Modify the dataset path in the code if your dataset structure is different.

## Training the Model
To train the model, run the train.py script. This script will train the CNN on your dataset and save the trained model to a file.

bash
`python train.py`

### Customizing Training
You can adjust the training parameters such as batch size, number of epochs, learning rate, and others by editing the train.py script.

## Running the GUI for Prediction
Once the model is trained and saved, you can use the GUI to make predictions on new images.

1. Run the GUI:

bash
`python gui_predict.py`

2. Using the GUI:

- Load an image using the "Load Image" button.
- Click "Predict" to see the model’s prediction.
- The prediction result will be displayed on the GUI.

## Contributing
Contributions are welcome! Please create an issue or submit a pull request for any features, enhancements, or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
