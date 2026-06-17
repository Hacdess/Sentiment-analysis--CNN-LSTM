# Sentiment Analysis - CNN + LSTM

**Họ tên:** Lê Trung Kiên  
**MSSV:** 23127075  
**Tên bài tập:** Sentiment analysis - CNN + LSTM

## 1. Mục tiêu

Xây dựng mô hình phân loại cảm xúc văn bản tiếng Việt dựa trên kiến trúc trong `VNSen.pdf`: kết hợp nhiều kênh CNN với LSTM để học đồng thời đặc trưng cục bộ và đặc trưng theo thứ tự từ.

Bài toán được thực hiện dưới dạng phân loại 3 lớp:

- Positive
- Negative
- Neutral

## 2. Dataset

Sử dụng bộ dữ liệu Vietnamese Sentiment trong thư mục:

```text
MultiChannel/data/VS/
```

Bộ dữ liệu có 5 fold:

```text
Fold_1
Fold_2
Fold_3
Fold_4
Fold_5
```

Mỗi fold gồm các file train/test theo nhãn:

```text
train_nhan_1.txt: positive train
train_nhan_0.txt: negative train
train_nhan_2.txt: neutral train

test_nhan_1.txt: positive test
test_nhan_0.txt: negative test
test_nhan_2.txt: neutral test
```

Theo `VNSen.pdf`, thống kê của VS dataset:

| Dataset | Positive | Neutral | Negative | Total | Vocabulary |
| --- | ---: | ---: | ---: | ---: | ---: |
| VS | 5,988 | 5,573 | 5,939 | 17,500 | 21,540 |

## 3. Tiền xử lý dữ liệu

Các bước thực hiện:

1. Đọc dữ liệu từ các file `.txt`.
2. Gán nhãn:
   - `1` hoặc `[1, 0, 0]`: positive
   - `0` hoặc `[0, 1, 0]`: negative
   - `2` hoặc `[0, 0, 1]`: neutral
3. Chuẩn hóa văn bản cơ bản:
   - đưa về chữ thường
   - bỏ dòng rỗng
   - tách train/validation từ tập train
4. Vector hóa văn bản:
   - chuyển từ thành chỉ số số nguyên
   - padding/truncating về cùng độ dài
5. Đưa dữ liệu vào embedding layer.

## 4. Kiến trúc mô hình

Kiến trúc dựa trên paper `Multi-channel LSTM-CNN model for Vietnamese sentiment analysis`.

Mô hình gồm:

1. Input text
2. Text vectorization
3. Embedding layer
4. Nhánh CNN đa kênh:
   - Conv1D kernel size 3, 150 filters
   - Conv1D kernel size 5, 150 filters
   - Conv1D kernel size 7, 150 filters
   - Global max pooling cho từng kênh
5. Nhánh LSTM:
   - LSTM 128 units học đặc trưng theo thứ tự từ trong câu
6. Concatenate:
   - nối đặc trưng CNN và LSTM
7. Dense 100 + Dropout
8. Softmax output 3 lớp

Hyperparameter chính theo paper:

| Thành phần | Giá trị |
| --- | --- |
| CNN region sizes | 3, 5, 7 |
| Filters mỗi region size | 150 |
| Embedding dimension | 200 |
| LSTM units | 128 |
| Penultimate NN layer | 100 |
| Optimizer | AdaMax |

## 5. Huấn luyện

Thực hiện huấn luyện trên 5 fold.

Với mỗi fold:

1. Load train/test theo fold.
2. Tách validation từ train.
3. Fit vectorizer trên train.
4. Huấn luyện mô hình CNN + LSTM.
5. Lưu model tốt nhất theo validation accuracy.
6. Đánh giá trên test set.

## 6. Đánh giá

Các metric cần report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Kết quả nên trình bày dạng bảng:

| Fold | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |
| Average |  |  |  |  |

Kết quả tham khảo trong `VNSen.pdf` trên VS dataset:

| Method | With token | Without token |
| --- | ---: | ---: |
| SVM | 77.87 | 87.21 |
| LSTM | 78.98 | 85.83 |
| CNN | 81.49 | 86.66 |
| Proposed CNN + LSTM | 81.86 | 87.72 |

Performance theo từng lớp của mô hình đề xuất trên VS without token:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Positive | 0.92 | 0.90 | 0.91 |
| Neutral | 0.81 | 0.89 | 0.85 |
| Negative | 0.90 | 0.828 | 0.864 |

## 7. Nội dung notebook `23127075.ipynb`

Notebook cần có các phần:

1. Thông tin sinh viên và bài tập.
2. Mô tả bài toán sentiment analysis tiếng Việt.
3. Mô tả dataset.
4. Load và kiểm tra dữ liệu.
5. Tiền xử lý/vector hóa văn bản.
6. Xây dựng mô hình CNN + LSTM.
7. Huấn luyện theo từng fold.
8. Đánh giá kết quả.
9. Bảng tổng hợp performance.
10. Nhận xét kết quả và hướng cải thiện.

## 8. Ghi chú

Code gốc trong thư mục `MultiChannel` dùng Python 2 và Keras API cũ. Notebook `23127075.ipynb` nên dùng TensorFlow/Keras hiện đại để dễ chạy lại trên môi trường Python 3.
