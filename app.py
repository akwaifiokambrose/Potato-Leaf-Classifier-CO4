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

def validate_image_data(img):
    img_array = np.array(img.convert("RGB"))
    image_variance = np.std(img_array)
    
    if image_variance < 10.0:
        return False, "Too uniform. Looks like a solid block of color or computer-generated."
    
    R = img_array[:, :, 0].astype(int)
    G = img_array[:, :, 1].astype(int)
    B = img_array[:, :, 2].astype(int)
    
    total_intensity = R + G + B + 1 
    green_ratio = G / total_intensity
    
    green_pixels = np.sum(green_ratio > 0.35)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100
    
    if green_percentage < 5.0:
        return False, f"Only {green_percentage:.1f}% plant color detected. Please upload a clear leaf."
        
    if green_percentage > 90.0:
        return False, f"{green_percentage:.1f}% green is abnormally high. Looks artificial or heavily filtered."
        
    return True, f"Passed data check ({green_percentage:.1f}% natural green detected)."

uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", width=400)
    
    is_valid, message = validate_image_data(img)
    
    if not is_valid:
        st.error(f"🛑 **Validation Failed:** {message}")
        st.stop() 
    
    st.success(f"✅ **Validation Passed:** {message}")
    
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
        
    if confidence < 93.0:
        st.warning(f"⚠️ **Low Confidence Detected ({confidence:.1f}%)**")
        st.write("This passed the color check, but the AI is unsure. Please ensure it's a clear potato leaf.")
    else:
        st.success(f"**Diagnosis: {label}**")
        st.progress(int(confidence), text=f"Confidence Score: {confidence:.1f}%")
        
        if "Blight" in label:
            st.warning("Recommendation: Isolate affected plants and consider applying an appropriate fungicide to prevent fungal spore spread.")
        else:
            st.info("Recommendation: Continue standard monitoring and irrigation practices.")