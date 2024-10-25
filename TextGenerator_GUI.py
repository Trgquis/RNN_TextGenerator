import tkinter as tk
from tkinter import messagebox
import numpy as np
import collections
import tensorflow as tf
import pickle

# Tải kiến trúc mô hình từ pickle
with open('E:/CTU/DeepLearning_Projects/RNN_TextGenerator/RNN_TextGenerator/my_model.pkl', 'rb') as f:
    model_json = pickle.load(f)
    model = tf.keras.models.model_from_json(model_json)

# Tải trọng số mô hình
model.load_weights('E:/CTU/DeepLearning_Projects/RNN_TextGenerator/RNN_TextGenerator/my_model_weights.h5')


def build_dataset(words):
    count = collections.Counter(words).most_common()
    word2id = {}
    for word, freq in count:
        word2id[word] = len(word2id)
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return word2id, id2word
def read_TextData(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    words = []
    for line in content:
        words.extend(line.split())
    
    return np.array(words)
data = read_TextData('E:\CTU\DeepLearning_Projects\RNN_TextGenerator\RNN_TextGenerator\TextData.txt')
w2i, i2w = build_dataset(data)

# Hàm để mã hóa dữ liệu đầu vào
def encode(sent):
    # Chuyển đổi câu thành các chỉ số tương ứng (giả sử bạn có từ điển w2i và i2w)
    return np.array([[w2i[w] for w in sent.split() if w in w2i]])

# Hàm để thực hiện dự đoán
def predict(input_text):
    if len(input_text.split()) < 2:
        return "Vui lòng nhập ít nhất 2 từ."
    
    encoded_input = encode(input_text)
    predictions = model.predict(encoded_input)
    predicted_word_index = np.argmax(predictions, axis=-1)[0]  # Lấy chỉ số từ dự đoán
    return i2w[predicted_word_index]  # Trả về từ tương ứng

# Hàm xử lý khi nhấn nút
def on_predict():
    try:
        input_data = entry.get()
        prediction = predict(input_data)
        messagebox.showinfo("Kết quả dự đoán", f"Từ dự đoán: {prediction}")
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Dự đoán với mô hình Keras")

# Trường nhập liệu duy nhất
entry = tk.Entry(root, width=50)
entry.pack(pady=20)

# Nút dự đoán
predict_button = tk.Button(root, text="Dự đoán", command=on_predict)
predict_button.pack()

# Thông tin hướng dẫn
label = tk.Label(root, text="Nhập ít nhất 2 từ để dự đoán từ tiếp theo:")
label.pack()

root.mainloop()
