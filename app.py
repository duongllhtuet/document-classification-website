import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import PyPDF2
from docx import Document

from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = load_model('models/best_model.keras')

tokenizer = KerasTokenizer(num_words=5000)
tokenizer.fit_on_texts([])  

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


LABEL_MAP = {
    0: 'Business',
    1: 'Sci_Tech',
    2: 'Sports',
    3: 'World'
}

def predict_category(text):
    """
    Phân loại văn bản bằng mô hình LSTM đã huấn luyện.
    """
    cleaned_text = preprocess_text(text)
    
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=150, padding='post')

    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)[0]
    predicted_category = LABEL_MAP.get(predicted_label, 'Unknown')
    
    return f"Category: {predicted_category}"

def extract_text(file):
    """
    Trích xuất văn bản từ file PDF hoặc DOCX
    """
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext == 'pdf':
        return extract_text_from_pdf(file)
    elif file_ext == 'docx':
        return extract_text_from_docx(file)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

def extract_text_from_pdf(file):
    """
    Trích xuất văn bản từ file PDF
    """
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file):
    """
    Trích xuất văn bản từ file DOCX
    """
    doc = Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text.strip()

@app.route('/')
def home():
    return redirect(url_for('choose'))

@app.route('/choose', methods=['GET', 'POST'])
def choose():
    if request.method == 'POST':
        option = request.form.get('option')
        if option == 'text':
            return redirect(url_for('text_input'))
        elif option == 'file':
            return redirect(url_for('file_upload'))
    return render_template('choose.html')

@app.route('/text_input', methods=['GET', 'POST'])
def text_input():
    if request.method == 'POST':
        text = request.form.get('text_input')
        if text:
            prediction = predict_category(text)
            return render_template('result.html', result=prediction, input_text=text)
    return render_template('text_input.html')

@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                content = extract_text(file)  
                
                prediction = predict_category(content)
                return render_template('result.html', result=prediction, input_text=content)
            except ValueError as e:
                flash(str(e), 'error')
    return render_template('file_upload.html')

if __name__ == '__main__':
    app.run(debug=True)
