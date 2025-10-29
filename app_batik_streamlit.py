import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import gdown
import os

# =========================
# 🧵 SETTING APLIKASI
# =========================
st.set_page_config(page_title="Klasifikasi Motif Batik Indonesia", page_icon="🧵", layout="centered")
st.title("🧵 Klasifikasi Motif Batik Indonesia")
st.markdown("""
Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** untuk mengenali berbagai **motif batik tradisional Indonesia**.  
Upload gambar batik untuk melihat hasil prediksi.
""")

# =========================
# 🔽 DOWNLOAD MODEL (DARI GOOGLE DRIVE)
# =========================
MODEL_PATH = "batik_classification_model.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1ygoQg36k0WD7YkNNY0seTm7iq8Og5QnW" 
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# =========================
# 🧠 LOAD MODEL
# =========================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =========================
# 📚 LABEL KELAS DAN DESKRIPSI
# =========================
class_names = ['Kawung', 'Mega Mendung', 'Parang', 'Truntum', 'Sekar Jagad', 'Tambal']
class_descriptions = {
    'Kawung': 'Motif dengan pola lingkaran seperti buah aren — melambangkan keadilan dan kesucian.',
    'Mega Mendung': 'Berasal dari Cirebon, menggambarkan awan mendung sebagai lambang kesabaran.',
    'Parang': 'Motif berbentuk diagonal panjang, melambangkan kekuatan dan keberanian.',
    'Truntum': 'Motif dari Solo yang berarti cinta tumbuh kembali, sering digunakan dalam pernikahan.',
    'Sekar Jagad': 'Melambangkan keindahan dan keberagaman budaya Indonesia.',
    'Tambal': 'Motif dengan potongan kain kecil, melambangkan perbaikan dan harapan baru.'
}

# =========================
# 📤 UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader("📸 Upload gambar batik (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    st.image(img, caption='🖼️ Gambar yang diunggah', use_container_width=True)

    # =========================
    # 🔍 PREPROCESSING GAMBAR
    # =========================
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # =========================
    # 🤖 PREDIKSI
    # =========================
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.markdown("### 🧾 Hasil Prediksi:")
    st.success(f"**Motif Batik:** {predicted_class}")
    st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")
    st.write(f"📖 *{class_descriptions[predicted_class]}*")

    # =========================
    # 📊 PROBABILITAS
    # =========================
    st.subheader("📊 Probabilitas Tiap Kelas:")
    prob_dict = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
    st.bar_chart(prob_dict)

else:
    st.warning("Silakan upload gambar batik terlebih dahulu untuk melihat hasil prediksi.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("© 2025 | Aplikasi Klasifikasi Motif Batik Indonesia - CNN + Streamlit 🧵")
