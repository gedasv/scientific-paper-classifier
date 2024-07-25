from flask import Flask, request, send_file
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import re
import io
import yaml

app = Flask(__name__)

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Typically this would be in a separate file
# However, for the sake of keeping everything in one jupiter notebook file, I did not write the model and dataset classes in a separate file
# This is the same code as in the training notebook

# Define the model
class BERTClassifier(nn.Module):
    def __init__(self, num_labels, pretrained_model):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Define the dataset
class ScientificPaperDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

# Initialize model and tokenizer
model = BERTClassifier(config['model']['num_labels'], config['model']['pretrained_model']).to(device)
model.load_state_dict(torch.load(config['model']['save_path'], map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model'])
subject_names = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

# Preprocess text utility function would also be in a separate file ...
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b(?!\w)', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Basically, I am keeping everything in one file for the sake of simplicity
# However, in a real-world scenario, we would adhere to best coding practices
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and file.filename.endswith('.parquet'):
            # Read the parquet file and preprocess the abstracts
            df = pd.read_parquet(io.BytesIO(file.read()))
            df['processed_text'] = df['TITLE'] + ' ' + df['ABSTRACT']
            df['processed_text'] = df['processed_text'].apply(preprocess_text)
            
            # Create dataset and dataloader
            dataset = ScientificPaperDataset(df['processed_text'].tolist(), tokenizer, config['data']['max_length'])
            dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'])
            
            # Predict
            all_preds = []
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    preds = torch.sigmoid(outputs).cpu().numpy()
                    all_preds.extend(preds)
            
            predictions_df = pd.DataFrame(all_preds, columns=subject_names)
            
            # Add additional columns
            predictions_df['ID'] = range(len(predictions_df))
            predictions_df['Abstract'] = df['ABSTRACT']
            
            predictions_df['Subjects_Threshold'] = predictions_df[subject_names].apply(
                lambda row: [subject for subject, value in row.items() if value > 0.5], axis=1
            )
            predictions_df['Subjects_Top2'] = predictions_df[subject_names].apply(
                lambda row: [subject for subject, value in sorted(row.items(), key=lambda x: x[1], reverse=True)[:2]], axis=1
            )
            
            predictions_df = predictions_df[['ID', 'Abstract', 'Subjects_Threshold', 'Subjects_Top2'] + subject_names]
            
            # Save to parquet
            output = io.BytesIO()
            predictions_df.to_parquet(output)
            output.seek(0)
            
            return send_file(output, as_attachment=True, download_name='predictions.parquet', mimetype='application/octet-stream')
    
    # Very simple HTML form
    return '''
    <!doctype html>
    <title>Upload Parquet File</title>
    <h1>Upload Parquet File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)