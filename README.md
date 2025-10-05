-----

# Handwritten Digit & Letter Recognizer ✍️

This project is an interactive web application that recognizes handwritten digits (0-9) and lowercase letters (a-z). It is built with a PyTorch-powered Convolutional Neural Network (CNN) and a Flask backend.

The standout feature is its ability to perform **online learning**: users can correct the model's predictions in real-time, and the model immediately fine-tunes itself on the new data, continuously improving its accuracy.

-----

## Features

  * **Real-time Recognition**: Draw a character on an HTML canvas and get an instant prediction.
  * **36 Classes**: Recognizes all 10 digits (0-9) and 26 lowercase letters (a-z).
  * **Interactive Online Learning**: Correct wrong predictions to fine-tune the model on the fly.
  * **Simple UI**: Clean and intuitive web interface for drawing, predicting, and training.
  * **Experience Replay**: When fine-tuning, the model retrains on the new sample plus a small batch of previous user-submitted samples to prevent catastrophic forgetting.

-----

## Tech Stack

  * **Backend**: Python, Flask
  * **Machine Learning**: PyTorch, Torchvision
  * **Image Processing**: Pillow, NumPy
  * **Frontend**: HTML5, CSS, vanilla JavaScript

-----

## Project Structure

```
handwritten_recognizer/
│
├── model.py             # CNN model definition
├── dataset_utils.py     # Data loading and preprocessing utilities
├── utils.py             # Image conversion utilities (base64 -> PIL)
├── train.py             # Script to pre-train the model
├── app.py               # Flask application backend
├── requirements.txt     # Project dependencies
└── static/
    └── index.html       # Frontend HTML and JavaScript
```

-----

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1\. Clone the Repository

```bash
git clone https://github.com/mithuhn-27/Handwritten_Recognizer.git
cd handwritten-recognizer
```

### 2\. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# Create the environment
python -m venv .venv
```
```bash
# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate
```

### 3\. Install Dependencies

Install all the required Python libraries.

```bash
pip install -r requirements.txt
```

-----

## Usage

Running the application is a two-step process. First, you need to train the initial model. Then, you can run the web server to start using it.

### 1\. Pre-train the Model

The `train.py` script downloads the MNIST (digits) and EMNIST (letters) datasets, trains the CNN, and saves the best model weights to `model.pt`.

In your terminal, run:

```bash
python train.py
```

This will take a few minutes. Once it's finished, you will see `model.pt` and `class_map.json` in your project folder.

### 2\. Run the Web Application

Now, start the Flask web server.

```bash
python app.py
```

Open your web browser and navigate to:

**`http://localhost:5000`**

You can now draw on the canvas, click **Predict** to see the result, or enter the correct label and click **Train with this** to help the model learn.
