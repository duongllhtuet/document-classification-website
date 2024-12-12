from flask import Flask, render_template, request, redirect, url_for, flash
import PyPDF2
from docx import Document

# Khởi tạo Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Mock model (thay thế bằng model thật của bạn)
class MockModel:
    def predict(self, texts):
        return ["Category: Example"]  # Trả về một loại ví dụ

model = MockModel()

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
            prediction = model.predict([text])[0]  # Dùng mô hình của bạn
            return render_template('result.html', result=prediction, input_text=text)
    return render_template('text_input.html')

@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                content = extract_text(file)  # Trích xuất văn bản từ file
                prediction = model.predict([content])[0]  # Dùng mô hình phân loại
                return render_template('result.html', result=prediction, input_text=content)
            except ValueError as e:
                flash(str(e), 'error')
    return render_template('file_upload.html')

if __name__ == '__main__':
    app.run(debug=True)
