Dưới đây là phiên bản README cho dự án của bạn:

---

# Document Classifier 📄🔍

## Mô Tả Dự Án
Một ứng dụng web thông minh cho phép người dùng phân loại tài liệu một cách dễ dàng thông qua giao diện thân thiện. Người dùng có thể nhập văn bản trực tiếp hoặc tải lên các file PDF, DOCX để phân loại.

## Tính Năng ✨
- Phân loại văn bản từ văn bản trực tiếp
- Tải lên và phân loại file PDF, DOCX
- Giao diện người dùng đơn giản và thân thiện
- Hỗ trợ nhiều định dạng file

## Yêu Cầu Hệ Thống 🖥️
- Python 3.7+
- Flask
- PyPDF2
- python-docx

## Cài Đặt 🔧

### Clone repository
```bash
git clone https://github.com/username/document-classifier.git
cd document-classifier
```

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Chạy ứng dụng
```bash
python app.py
```

## Cách Sử Dụng 🚀
1. Truy cập `http://localhost:5000` từ trình duyệt.
2. Chọn phương thức nhập liệu:
   - Nhập văn bản trực tiếp hoặc tải lên file.
3. Nhấn "Classify" để nhận kết quả phân loại tài liệu.

## Cấu Trúc Dự Án 📂
```
/document-classifier
│
├── app.py              # File chính Flask
├── requirements.txt    # Danh sách dependencies
│
├── templates/
│   ├── choose.html     # Giao diện chọn phương thức nhập liệu
│   ├── text_input.html # Giao diện nhập văn bản
│   └── file_upload.html # Giao diện tải lên file
│
└── static/
    └── style.css       # Tập tin CSS cho giao diện
```

## Công Nghệ Sử Dụng 🛠️
- Flask
- PyPDF2
- python-docx
- HTML/CSS

## Lưu Ý ⚠️
- Đây là phiên bản demo, cần thay thế mô hình phân loại thực tế.
- Chưa phù hợp cho môi trường production.

## Đóng Góp 🤝
Mọi đóng góp và pull request đều được chào đón!

## Giấy Phép 📄
MIT License

--- 

Bạn có thể thay thế tên repo trong URL clone với tên repo của mình nếu cần.