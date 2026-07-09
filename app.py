import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
from googletrans import Translator

from predict_disease import predict_image


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide"
)

# =====================================================
# TRANSLATOR
# =====================================================

translator = Translator()

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.main-title {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
    margin-top: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #DDDDDD;
    font-size: 22px;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background-color: rgba(255,255,255,0.05);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0px 0px 20px rgba(0,255,0,0.2);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Input */
.stTextInput input {
    border-radius: 15px;
    border: 2px solid #4CAF50;
    background-color: #1E1E1E;
    color: white;
    padding: 12px;
}

/* Button */
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
}

/* File uploader */
.stFileUploader {
    background-color: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<p class="main-title">🌾 Smart Agriculture AI Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI Farming Assistant • Disease Detection • Farmer Support • Multilingual</p>',
    unsafe_allow_html=True
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌱 Navigation")

# LANGUAGE SELECTOR
language = st.sidebar.selectbox(
    "🌐 Select Language",
    ["English", "Hindi", "Marathi"]
)

menu = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🤖 Farming Assistant",
        "🧪 Disease Detection",
        "📞 Farmer Help"
    ]
)

# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("agriculture_dataset.csv", on_bad_lines="skip")

questions = data["question"].astype(str).tolist()

if "answer" in data.columns:
    answers = data["answer"].astype(str).tolist()

elif "answers" in data.columns:
    answers = data["answers"].astype(str).tolist()

elif "response" in data.columns:
    answers = data["response"].astype(str).tolist()

else:
    st.error("No valid answer column found!")
    st.stop()

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(questions)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# =====================================================
# HOME PAGE
# =====================================================

if menu == "🏠 Home":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1500937386664-56d1dfef3854",
        use_column_width=True
    )

    home_text = """
# 🌱 Welcome to Smart Agriculture AI

This platform helps farmers using Artificial Intelligence.

## 🚀 Features

✅ Smart Farming Assistant  
✅ Crop Disease Detection  
✅ Treatment Suggestions  
✅ Farmer Government Support  
✅ AI-Based Farming Guidance  

---

## 🌾 Benefits for Farmers

- Better crop management
- Fast disease detection
- Government support information
- Modern farming guidance
"""

    if language == "Hindi":
        home_text = translator.translate(home_text, dest='hi').text

    elif language == "Marathi":
        home_text = translator.translate(home_text, dest='mr').text

    st.markdown(home_text)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FARMING ASSISTANT
# =====================================================

elif menu == "🤖 Farming Assistant":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🤖 Smart Farming Assistant")

    query = st.text_input("Ask farming question")

    if query:

        query_lower = query.lower()

        # =================================================
        # RICE GUIDE
        # =================================================

        if "rice" in query_lower:

            answer = """
🌾 Complete Rice Farming Guide

1️⃣ Land Preparation
- Plough field properly
- Level the soil
- Maintain water retention

2️⃣ Seed Selection
- Use certified seeds
- Soak seeds for 24 hours

3️⃣ Nursery Preparation
- Prepare nursery bed
- Grow seedlings for 20–25 days

4️⃣ Transplanting
- Maintain spacing between plants

5️⃣ Irrigation
- Maintain proper water level

6️⃣ Fertilizer Use
- Apply NPK fertilizers
- Add organic compost

7️⃣ Pest Control
- Monitor crop regularly
- Use pesticides carefully

8️⃣ Harvesting
- Harvest when grains become golden

✅ Tip:
Good irrigation improves yield.
"""

        # =================================================
        # WHEAT GUIDE
        # =================================================

        elif "wheat" in query_lower:

            answer = """
🌾 Wheat Farming Guide

- Prepare fine soil
- Use quality seeds
- Sow in rows
- Apply fertilizers
- Irrigate properly
- Remove weeds
- Harvest at golden stage

✅ Tip:
Avoid overwatering.
"""

        # =================================================
        # FERTILIZER GUIDE
        # =================================================

        elif "fertilizer" in query_lower:

            answer = """
🌱 Fertilizer Usage Guide

- Use NPK fertilizers
- Use organic compost
- Avoid overuse
- Perform soil testing

✅ Healthy soil improves crop growth.
"""

        else:

            query_vector = model.encode([query])

            query_vector = np.array(query_vector).astype("float32")

            distances, indices = index.search(query_vector, 1)

            answer = answers[indices[0][0]]

        # =================================================
        # TRANSLATION
        # =================================================

        if language == "Hindi":
            answer = translator.translate(answer, dest='hi').text

        elif language == "Marathi":
            answer = translator.translate(answer, dest='mr').text

        st.success(answer)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# DISEASE DETECTION
# =====================================================

elif menu == "🧪 Disease Detection":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧪 Detect Crop Disease from Image")

    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Image", width=350)

        result = predict_image(uploaded_file)

        disease_text = f"🌿 Detected Disease: {result}"

        if language == "Hindi":
            disease_text = translator.translate(disease_text, dest='hi').text

        elif language == "Marathi":
            disease_text = translator.translate(disease_text, dest='mr').text

        st.success(disease_text)

        # =================================================
        # TREATMENT
        # =================================================

        if result == "Leaf_Spot":

            treatment = """
🦠 Treatment for Leaf Spot

- Remove infected leaves
- Spray Mancozeb fungicide
- Avoid excess watering
- Improve air circulation
"""

        elif result == "Blight":

            treatment = """
🦠 Treatment for Blight

- Use copper fungicide
- Remove infected areas
- Avoid leaf moisture
"""

        elif result == "Virus":

            treatment = """
🦠 Treatment for Virus

- Remove infected plants
- Control aphids/insects
- Use resistant seeds
"""

        else:

            treatment = """
✅ Plant is Healthy

- Proper watering
- Balanced fertilizer
- Regular monitoring
"""

        if language == "Hindi":
            treatment = translator.translate(treatment, dest='hi').text

        elif language == "Marathi":
            treatment = translator.translate(treatment, dest='mr').text

        st.warning(treatment)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FARMER HELP
# =====================================================

elif menu == "📞 Farmer Help":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    help_text = """
# 🇮🇳 Important Farmer Helplines

📞 Kisan Call Center: 1800-180-1551

📞 PM-Kisan Helpline: 155261

📞 Agriculture Helpline: 1800-180-1551

📞 Crop Insurance Help: 14447

---

# 🌱 Government Support

✅ Fertilizer guidance  
✅ Seed support  
✅ Pest control help  
✅ Crop insurance  
✅ Soil testing support  
✅ Irrigation schemes
"""

    if language == "Hindi":
        help_text = translator.translate(help_text, dest='hi').text

    elif language == "Marathi":
        help_text = translator.translate(help_text, dest='mr').text

    st.info(help_text)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

footer = "🌱 Made with AI for Farmers"

if language == "Hindi":
    footer = translator.translate(footer, dest='hi').text

elif language == "Marathi":
    footer = translator.translate(footer, dest='mr').text

st.markdown(
    f"<center><h4>{footer}</h4></center>",
    unsafe_allow_html=True
)