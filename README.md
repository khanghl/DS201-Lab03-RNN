## Lab 03: Mạng Neural Hồi Quy (Recurrent Neural Networks - RNN, LSTM, GRU và BiLSTM)

### Mục Tiêu
- Nắm vững cơ chế truyền trạng thái ẩn qua thời gian, cấu trúc cổng của LSTM/GRU, cách xử lý chuỗi độ dài thay đổi bằng `pack_padded_sequence` / `pad_packed_sequence` và chặn chuẩn gradient (`clip_grad_norm_=1.0`)

- Áp dụng cấu hình chung thống nhất (`emb_dim=300` học từ đầu, `hidden_size=256`, `dropout=0.3`, `batch_size=32`, `Adam lr=1e-3`, `epochs=10`, `SEED=42`) cho các bài toán phân loại cảm xúc và gán nhãn chuỗi tiếng Việt:
  - **Bài 1:** LSTM 5 lớp phân loại cảm xúc trên UIT-VSFC (3.138.007 tham số, đối chiếu với baseline Majority và TF-IDF + Logistic Regression theo độ đo chính macro-F1, phân tích các ca dự đoán sai trên tập dev)
  
  - **Bài 2:** GRU 5 lớp trên UIT-VSFC (2.468.823 tham số — khối hồi quy bằng đúng 3/4 số tham số của LSTM, so sánh có kiểm soát chất lượng macro-F1 và chi phí thời gian/epoch với Bài 1)
  
  - **Bài 3:** Encoder BiLSTM 5 lớp + Linear(512, 21) cho nhận diện thực thể có tên trên PhoNER_COVID19 bản syllable (8.305.297 tham số, `CrossEntropyLoss(ignore_index=-100)`, đánh giá entity micro-F1 theo chuẩn `seqeval` strict IOB2 và phân tích các kiểu lỗi biên/nhãn)
  
  - **Bài 4\*:** Khảo sát ảnh hưởng độ sâu `num_layers` $\in \{1, 2, 5\}$ của LSTM ở `hidden_size=256` trên UIT-VSFC (1.032.663 vs 1.558.999 vs 3.138.007 tham số, tái sử dụng mô hình 5 lớp ở Bài 1)

---

### Cấu Trúc Folder
- data.py: Tải trực tiếp và kiểm chứng SHA-256 cho 6 tệp dữ liệu, xây từ điển chỉ từ tập train (`<pad>=0`, `<unk>=1`), hàm `collate_cls`, `collate_ner` (`PAD_TAG=-100`) và `DataLoader` đồng bộ theo `seed=42`

- models.py: Định nghĩa `TextClassifier` (hỗ trợ RNN/LSTM/GRU nhiều lớp kèm `pack_padded_sequence`), `TokenTagger` (Encoder BiLSTM 5 lớp kèm `pad_packed_sequence`) và các hàm đối chiếu tham số `n_params`, `inspect_layers`

- engine.py: Hàm `fit_tfidf_baseline`, `train_one_epoch_cls`, `evaluate_cls`, `fit_cls` (chọn `best.pt` theo dev macro-F1 cao nhất) và `train_one_epoch_ner`, `evaluate_ner`, `fit_ner` (chọn `best.pt` theo dev entity micro-F1 strict IOB2)

- utils.py: Các hàm kiểm chứng bất biến padding (đệm 8 vs 64), vẽ biểu đồ phân bố dữ liệu, đường cong Loss/F1, Confusion Matrix, đánh giá tập test, phân tích lỗi trên dev và in bảng tổng hợp chuẩn nộp bài

- assignments.ipynb: Báo cáo thực hành hoàn chỉnh cho cả 4 bài

---

### Lưu Ý Bộ Dữ Liệu
- **UIT-VSFC** (11.426 train / 1.583 dev / 3.166 test — 3 nhãn sentiment): [Google Drive Link](https://drive.google.com/drive/folders/1rdcXNGt_3-QUvV8EtSvVsLMVeHmk6Yqk?usp=drive_link)

- **PhoNER_COVID19** bản syllable (5.027 train / 2.000 dev / 3.000 test — 21 nhãn BIO thuộc 10 loại thực thể): [VinAI Research GitHub](https://github.com/VinAIResearch/PhoNER_COVID19)

- Tuân thủ quy định bản quyền của PhoNER: không đính kèm thư mục `data/` trong repository công khai; hàm `ensure_datasets()` trong `data.py` sẽ tự động tải trực tiếp từ nguồn gốc và đối chiếu mã băm SHA-256 khi chạy trên máy cá nhân hoặc Google Colab
