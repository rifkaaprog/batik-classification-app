import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Judul dan deskripsi aplikasi
st.set_page_config(page_title="Klasifikasi Motif Batik Indonesia", page_icon="🧵")
st.title("🧵 Klasifikasi Motif Batik Indonesia")
st.markdown("""
Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** untuk mengenali berbagai **motif batik tradisional Indonesia**.  
Upload gambar batik untuk melihat hasil prediksi.
""")

# Load model
MODEL_PATH = "batik_classification_model.h5"  # ganti nama sesuai file model kamu
model = load_model(MODEL_PATH)

# Label kelas
class_names = ['Kawung', 'Mega Mendung', 'Parang', 'Truntum', 'Sekar Jagad', 'Tambal']

# Upload gambar
uploaded_file = st.file_uploader("📸 Upload gambar batik (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    st.image(img, caption='Gambar yang diunggah', use_column_width=True)

    # Preprocessing gambar
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediksi
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.markdown("### 🧾 Hasil Prediksi:")
    st.success(f"**Motif Batik:** {predicted_class}")
    st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")

    # Probabilitas tiap kelas
    st.subheader("📊 Probabilitas Tiap Kelas:")
    prob_dict = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
    st.bar_chart(prob_dict)
else:
    st.warning("Silakan upload gambar batik terlebih dahulu untuk melihat hasil prediksi.")

st.markdown("---")
st.caption("© 2025 | Aplikasi Klasifikasi Motif Batik Indonesia - CNN + Streamlit")
