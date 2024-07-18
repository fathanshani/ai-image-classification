import streamlit as st
import streamlit.components.v1 as components
import json
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

import numpy as np
import pandas as pd
from PIL import Image

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import gridfs
import io
import os
from pymongo import MongoClient, errors
from bson.objectid import ObjectId
from dotenv import load_dotenv
import collections


if not hasattr(collections, 'MutableMapping'):
    import collections.abc
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping

# MongoDB setup
load_dotenv()
MONGO_URI = st.secrets["MONGO_URI"]
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["image_classification"]
    collection = db["history"]
    fs = gridfs.GridFS(db)
    client.admin.command('ping')  # Test the connection
    mongo_connected = True
except errors.ServerSelectionTimeoutError:
    mongo_connected = False

# Load model
model_path = "model/model_ai_vs_real.h5"
model = load_model(model_path)

# Kelas
class_names = ["AI", "Human"]

# Fungsi untuk load dan proses gambar
def load_and_preprocess_image(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

selected = option_menu(
    menu_title=None,
    options=["Home", "Deteksi", "Histori", "Tentang"],
    icons=["house", "robot", "clock-history", "person-circle"],
    default_index=0,
    orientation="horizontal"
)

def load_lottiefile(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
lottie_human = load_lottiefile("assets/Animation - human.json")
lottie_ai = load_lottiefile("assets/Animation - AI.json")

# Welcome Page
if selected == "Home":
    st.title("Pendeteksi Gambar Buatan AI")
    st.write("""
            Tahun-tahun belakangan ini, artificial intelligence sangatlah populer digunakan orang-orang.
            Serta muncul teknologi artificial intelligence untuk membuat gambar dengan model text-to-image,
            seperti DALL-E, Midjourney, Stable Diffusion dan lainnya. Orang-orang dapat dengan mudah membuat
            gambar dengan teknologi tersebut tanpa butuh keterampilan menggambar yang memadai.
            Beberapa orang memanfaatkan teknologi tersebut untuk hal yang tidak baik, seperti menjual gambar tersebut
            dengan harga yang sama dengan gambar buatan tangan tanpa memberitahu gambar tersebut merupakan buatan AI.
            Gunakan sistem deteksi ini untuk melihat gambar tersebut merupakan hasil buatan AI atau manusia!
            """)

    # Contoh gambar
    st.header("Contoh Gambar")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Buatan Manusia")
        st.image("assets/human1.jpg", width=320)  # Ubah path sesuai contoh gambar Human

    with col2:
        st.subheader("Buatan AI")
        st.image("assets/woman-8161029_1920.jpg", width=320)  # Ubah path sesuai contoh gambar AI

# Classification Page
if selected == "Deteksi":
    st.title("Deteksi Gambar atau Foto")
    uploaded_file = st.file_uploader("Pilih gambar yang ingin dideteksi...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load gambar
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess gambar
        img_array = load_and_preprocess_image(img)

        # Lakukan prediksi
        prediction = model.predict(img_array)
        predicted_class = class_names[int(np.round(prediction)[0][0])]

        colResult, colGif = st.columns(3, vertical_alignment='center')

        if predicted_class == 'Human':
            with colResult:
                st.subheader(f"Result: :green[{predicted_class}]")
            with colGif:
                st_lottie(lottie_human,
                        speed=1,
                        reverse=False,
                        loop=True,
                        quality='medium',
                        height=224,
                        width=224)
        else:
            with colResult:
                st.subheader(f"Result: :blue[{predicted_class}]")
            with colGif:
                st_lottie(lottie_ai,
                        speed=1,
                        reverse=False,
                        loop=True,
                        quality='medium',
                        height=224,
                        width=224)

        if mongo_connected:
            try:
                # Simpan gambar ke MongoDB
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_id = fs.put(img_byte_arr.getvalue(), filename=uploaded_file.name)

                # Simpan hasil ke MongoDB
                collection.insert_one({"image_id": img_id, "filename": uploaded_file.name, "class": predicted_class})
            except errors.PyMongoError as e:
                st.error(f"Terjadi kesalahan saat menyimpan hasil ke MongoDB: {e}")
        else:
            st.warning("Tidak dapat menyimpan hasil ke MongoDB karena database tidak tersedia.")

# History Page
if selected == "Histori":
    st.title("Histori Hasil Deteksi")
    st.write("Berikut ini adalah gambar dengan hasil deteksinya")
    
    if mongo_connected:
        try:
            # Sort by the newest first
            history = list(collection.find().sort([("_id", -1)]))
        except errors.PyMongoError as e:
            st.error(f"Terjadi kesalahan saat mengambil data dari MongoDB: {e}")
            history = []
    else:
        st.warning("Tidak dapat mengambil data dari MongoDB karena database tidak tersedia.")
        history = []
    
    if not history:
        st.write("Belum ada gambar yang dideteksi.")
    else:
        for record in history:
            img_id = record["image_id"]
            img_data = fs.get(ObjectId(img_id)).read()
            img = Image.open(io.BytesIO(img_data))
            
            with st.container(border=True):
                if record['class'] == 'Human':
                    st.image(img, width=224)
                    st.caption(f"Result: :green[{record['class']}] | {record['filename']}")
                else:
                    st.image(img, width=224)
                    st.caption(f"Result: :blue[{record['class']}] | {record['filename']}")

if selected == "Tentang":
    colPhoto, colDesc = st.columns([1, 2], vertical_alignment='center')

    with colPhoto:
        st.image('assets/profile.jpg', caption='Fathan Shani Putra Aliadi')

    with colDesc:
        st.write("""Seorang mahasiswa yang sedang menjalani program studi Informatika di Universitas Gunadarma .
                 Memiliki ketertarikan di dunia _artificial intelligence_ dan data.
                 """)
        st.page_link("https://github.com/fathanshani", label="GitHub", icon="🔗")
        st.page_link("https://www.linkedin.com/in/fathanshani/", label="LinkedIn", icon="💼")

ft = """
<style>
a:link , a:visited{
color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: #0283C3; /* theme's primary color*/
background-color: transparent;
text-decoration: underline;
}

#page-container {
  position: relative;
  min-height: 10vh;
}

footer{
    visibility:hidden;
}

.footer {
position: relative;
left: 0;
top:230px;
bottom: 0;
width: 100%;
background-color: transparent;
color: #808080;
text-align: center;
}
</style>

<div id="page-container">

<div class="footer">
<p style='font-size: 0.875em;'>Made with <a style='display: inline; text-align: left;' href="https://streamlit.io/" target="_blank">Streamlit</a><br 'style= top:3px;'>
with <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><a style='display: inline; text-align: left;' href="https://github.com/fathanshani" target="_blank"> by Fathan Shani</a></p>
</div>

</div>
"""
st.write(ft, unsafe_allow_html=True)
