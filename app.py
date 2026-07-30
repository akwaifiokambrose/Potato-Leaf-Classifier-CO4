import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Potato Leaf Classifier", page_icon="🥔", layout="centered")
st.title("🥔 Potato Leaf Disease Classifier")
st.write("Upload an image of a potato leaf to detect if it is **Healthy** or showing signs of **Early Blight**.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/potato_blight_model.keras")

uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", width=400)
    
    with st.spinner("Waking up the AI and analyzing patterns... (This takes a few seconds on the cloud)"):
        try:
            model = load_model()
        except Exception as e:
            st.error(f"Error loading model. Please ensure your .keras file is in the 'models/' directory. \n\n {e}")
            st.stop()
            
        img_resized = img.convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
        
        prob_class_1 = float(model.predict(img_array, verbose=0)[0][0])
        prob_class_0 = 1.0 - prob_class_1
        
        if prob_class_1 >= 0.5:
            label = "Potato Early Blight ⚠️"
            confidence = prob_class_1 * 100
        else:
            label = "Healthy Potato Leaf 🌿"
            confidence = prob_class_0 * 100
        
    if confidence < 85.0:
        st.warning(f"⚠️ **Low Confidence Detected ({confidence:.1f}%)**")
        st.write("This doesn't look like a clear potato leaf. Please ensure you uploaded the correct subject item.")
    else:
        st.success(f"**Diagnosis: {label}**")
        st.progress(int(confidence), text=f"Confidence Score: {confidence:.1f}%")
        
        if "Blight" in label:
            st.warning("Recommendation: Isolate affected plants and consider applying an appropriate fungicide to prevent fungal spore spread.")
        else:
            st.info("Recommendation: Continue standard monitoring and irrigation practices.")