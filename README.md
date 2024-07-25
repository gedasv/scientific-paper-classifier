# Scientific Paper Classifier

## Setup

You can set up the environment using either venv or Conda.

### Option 1: venv

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install pandas numpy matplotlib seaborn torch torchvision torchaudio scikit-learn transformers tqdm pyyaml flask
   ```

### Option 2: Conda (it works on my machine)

1. Clone the repository and navigate to the project directory.

2. Create a Conda environment and activate it:
   ```
   conda create --name paper_classifier python=3.8
   conda activate paper_classifier
   ```

3. Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

4. Ensure you have the following files in your project directory:
   - `config.yaml`: Configuration file
   - `cc_data.parquet`: Training data
   - `cc_test.parquet`: Test data
   - `requirements.txt`: List of required packages

## Pre-trained Model

A pre-trained model is available for immediate use. You can download it from the following link:

[Pre-trained Model](https://drive.google.com/file/d/1Wmrvl4A-P2uhnAlayCRLeuUneYqZYdNQ/view?usp=sharing)

After downloading, place the model file in the appropriate directory as specified in your `config.yaml` file.

## Training

1. Run all the cells in the noteboook.

2. The trained model will be saved as specified in the `config.yaml` file (`bert_classifier.pth` by default)

## Running the Prediction Server

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open a web browser and go to `http://127.0.0.1:5000` (you can Ctrl+click this link in most consoles).

3. Upload a `.parquet` file containing scientific paper data.

4. The server will process the file and return a `predictions.parquet` file with the classification results (straight to downloads).
