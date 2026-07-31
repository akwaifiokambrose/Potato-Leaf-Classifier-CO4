# 🥔 Potato Leaf Disease Classifier

A robust, web-based diagnostic tool built with TensorFlow and Streamlit. This application utilizes a Convolutional Neural Network (CNN) to accurately classify potato foliage as either **Healthy** or showing signs of **Early Blight**.

##  Key Features

* **Deep Learning Inference:** Powered by a custom-trained Keras model optimized for agricultural image classification.
* **Defensive Engineering (RGB Data Filter):** Features a pre-prediction validation pipeline that calculates pixel variance and green-channel ratios. This mathematically blocks Out-of-Distribution (OOD) uploads—such as documents, human subjects, or digital green screens—before the AI processes them.
* **Lazy Loading Optimization:** The heavy AI model is deferred until an image is explicitly uploaded, preventing server timeouts and ensuring instant frontend boot times.
* **Diagnostic Safety Net:** Enforces a strict 93% confidence threshold to flag uncertain predictions and prevent misdiagnosis on edge-case images which makes the model more resilient than otherwise.

##  Live Demo
The application is deployed and currently live on Streamlit Community Cloud: 
 **https://potato-leaf-classifier-co4.streamlit.app/**

---

##  Local Installation & Setup (Linux / WSL)

Alternatively, if you wish to run this application locally on your own machine, follow these steps. 

### Prerequisites
* Python 3.11 (Recommended to match deployment environment)
* Git

### Step-by-Step Guide

**1. Clone the repository**
```bash
git clone [https://github.com/akwaifiokambrose/Potato-Leaf-Classifier-CO4.git](https://github.com/akwaifiokambrose/Potato-Leaf-Classifier-CO4.git)
cd Potato-Leaf-Classifier-CO4
```

**2. Create and activate a virtual environment**
It is highly recommended to isolate the dependencies to prevent conflicts with your system Python.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the required dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**
```bash
streamlit run app.py
```

**5. View the App**
Once the server starts, the terminal will output a local network URL. Open your web browser and navigate to:
```text
http://localhost:8501
```

##  Repository Structure

* `app.py`: The main Streamlit web application and RGB validation logic.
* `models/potato_blight_model.keras`: The serialized TensorFlow CNN model.
* `requirements.txt`: The pinned dependencies required to build the environment.
* `README.md`: Project documentation.
