## Lab 03: Recurrent Neural Networks - RNN, LSTM, GRU và BiLSTM

#### Bài 1. LSTM 5 lớp cho phân loại cảm xúc
- **Yêu cầu:** Xây LSTM num_layers=5, hidden_size=256 cho sentiment UIT-VSFC; tối ưu bằng Adam. Dùng packing, chọn epoch theo macro-F1 dev và đánh giá checkpoint đã chọn trên test. Chạy baseline TF-IDF với cùng chia tập để có mốc đối chiếu.
- **Dữ liệu:** Toàn bộ UIT-VSFC: 11.426 train, 1.583 dev, 3.166 test. Chỉ dùng sentence làm đầu vào và sentiment làm nhãn; không dùng topic làm đặc trưng.
- **Định dạng kết quả:** Bảng accuracy, precision-macro, recall-macro, F1-macro trên test; F1 từng lớp; confusion matrix; đường loss/F1; số tham số và thời gian/epoch. Phân tích ít nhất 3 trường hợp dự đoán sai trên dev.
- **Gợi ý:** giữ layers=5 và hidden=256. Dùng lại TextClassifier và fit_cls; không thêm Softmax trước CrossEntropyLoss.

#### Bài 2. GRU 5 lớp và so sánh với LSTM
- **Yêu cầu:** Xây GRU num_layers=5, hidden_size=256 trên cùng nhiệm vụ sentiment; dùng Adam. Giữ cấu hình và quy trình đánh giá của Bài 1. Lập bảng so sánh hai mô hình và baseline, giải thích chất lượng cùng chi phí huấn luyện.
- **Dữ liệu:** Dùng đúng các tệp train/dev/test, từ điển, cách tách token và ánh xạ sentiment của Bài 1.
- **Định dạng kết quả:** Cùng các độ đo của Bài 1, thêm bảng LSTM/GRU với F1-macro, số tham số, thời gian/epoch và epoch tốt nhất trên dev. Không giả định GRU phải thắng hoặc LSTM phải thắng.
- **Gợi ý:** TextClassifier(kind='GRU') dùng h_n trực tiếp. Khởi tạo lại seed và mô hình, không tái sử dụng trạng thái optimizer của Bài 1.

#### Bài 3. Encoder BiLSTM 5 lớp cho NER
- **Yêu cầu:** Xây encoder gồm 5 lớp BiLSTM, hidden_size=256 cho mỗi chiều, nối Linear(512, 21) để gán nhãn BIO. Dùng packing và CrossEntropyLoss(ignore_index=-100). Chọn checkpoint theo entity-F1 dev và chấm test bằng seqeval strict IOB2.
- **Dữ liệu:** Toàn bộ PhoNER_COVID19 bản syllable: 5.027 train, 2.000 dev, 3.000 test. Giữ nguyên words/tags; không tách từ lại và không đổi bộ chia tập.
- **Định dạng kết quả:** Precision, recall, micro-F1 mức thực thể trên test; F1 từng loại thực thể; số tham số, thời gian/epoch, epoch tốt nhất. Phân tích ít nhất 3 kiểu lỗi trên dev; dùng ví dụ tự tạo khi minh họa trong repo công khai.
- **Gợi ý:** Cắt predicted và labels theo length trước khi đổi id sang nhãn chuỗi. Không tính PAD vào F1 và không dùng token accuracy làm kết quả NER chính.

#### Bài 4*. Ảnh hưởng của độ sâu hoặc CRF
- **Yêu cầu:** Chọn một hướng: 
	- (a) so sánh num_layers={1,2,5} của LSTM hoặc GRU ở hidden_size=256, giữ các yếu tố còn lại; 
	- (b) bổ sung CRF sau BiLSTM và đối chiếu với Bài 3. Giữ đúng chia tập, quy trình chọn checkpoint và độ đo của bài tương ứng.
- **Dữ liệu:** Dữ liệu và tiền xử lý của bài được chọn; không bổ sung dữ liệu ngoài vào phép so sánh chính.
- **Định dạng kết quả:** Bảng kết quả, số tham số, thời gian/epoch và giải thích dựa trên số đo. Phân biệt thay đổi kiến trúc với thay đổi số tham số; không suy rộng từ một seed thành kết luận tuyệt đối.
- **Gợi ý:** Đây là phần mở rộng. Yêu cầu 5 lớp của ba bài bắt buộc vẫn giữ nguyên theo assignments.ipynb gốc.


---

### Cấu Trúc Folder
- **data.py:** Tải trực tiếp và kiểm chứng SHA-256 cho 6 tệp dữ liệu, xây từ điển chỉ từ tập train (<pad>=0, <unk>=1), hàm collate_cls, collate_ner (PAD_TAG=-100) và DataLoader đồng bộ theo seed=42
- **models.py:** Định nghĩa TextClassifier (hỗ trợ RNN/LSTM/GRU nhiều lớp kèm pack_padded_sequence), TokenTagger (Encoder BiLSTM 5 lớp kèm pad_packed_sequence) và các hàm đối chiếu tham số n_params, inspect_layers
- **engine.py:** Hàm fit_tfidf_baseline, train_one_epoch_cls, evaluate_cls, fit_cls (chọn best.pt theo dev macro-F1 cao nhất) và train_one_epoch_ner, evaluate_ner, fit_ner (chọn best.pt theo dev entity micro-F1 strict IOB2)
- **utils.py:** Các hàm kiểm chứng bất biến padding (đệm 8 vs 64), vẽ biểu đồ phân bố dữ liệu, đường cong Loss/F1, Confusion Matrix, đánh giá tập test, phân tích lỗi trên dev và in bảng tổng hợp chuẩn nộp bài
- **assignments.ipynb:** Báo cáo thực hành hoàn chỉnh cho cả 4 bài


---

### Lưu Ý Bộ Dữ Liệu
- **UIT-VSFC** (11.426 train / 1.583 dev / 3.166 test:3 nhãn sentiment): [Google Drive Link](https://drive.google.com/drive/folders/1rdcXNGt_3-QUvV8EtSvVsLMVeHmk6Yqk?usp=drive_link)

- **PhoNER_COVID19** bản syllable (5.027 train / 2.000 dev / 3.000 test: 21 nhãn BIO thuộc 10 loại thực thể): [VinAI Research GitHub](https://github.com/VinAIResearch/PhoNER_COVID19)

- Tuân thủ quy định bản quyền của PhoNER: không đính kèm thư mục `data/` trong repository công khai; hàm ensure_datasets() trong data.py sẽ tự động tải trực tiếp từ nguồn gốc và đối chiếu mã băm SHA-256 khi chạy 
