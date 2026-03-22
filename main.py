import os

# 🚀 FIX 1: TensorFlow ला जुन्या (Keras 2) मॉडेलला सपोर्ट करायला सांगणे
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# 🚀 FIX 2: TensorFlow च्या फालतू वॉर्निंग्ज बंद करून RAM वाचवणे
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import tensorflow as tf
import keras  # 🚀 <--- बदल १: थेट keras इम्पोर्ट केले
import numpy as np
from keras.preprocessing import image # 🚀 <--- बदल २: tensorflow.keras ऐवजी फक्त keras
import json
import shutil
import requests
from datetime import datetime

app = FastAPI()

# 🚀 FIX 3: जर सर्व्हरवर फोल्डर्स नसतील, तर ते आपोआप बनवणे
os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = 'crop_model.h5'
LABELS_PATH = 'class_indices.json'
HISTORY_FILE = 'history.json'

print("Loading Model... Please wait.")
model = keras.models.load_model(MODEL_PATH) # 🚀 <--- बदल ३: tf.keras ऐवजी फक्त keras.models

# 🚀 SMART LABEL LOADER: Fixes Label Mismatch Issues!
with open(LABELS_PATH, 'r') as f:
    raw_labels = json.load(f)
    # Check if keys are string-numbers (e.g., "0": "Apple") or names (e.g., "Apple": 0)
    if list(raw_labels.keys())[0].isdigit():
        class_names = {int(k): v for k, v in raw_labels.items()}
    else:
        class_names = {v: k for k, v in raw_labels.items()}

print("Model and Labels Loaded Successfully!")

# Dictionary for Solutions
solutions = {
    "Pepper,_bell___Bacterial_spot": "Use Copper-based fungicides. Ensure proper spacing between plants.",
    "Tomato___Early_blight": "Apply Chlorothalonil or Copper fungicide. Remove infected leaves.",
    "Potato___Late_blight": "Apply Mancozeb fungicide. Do not overwater the crops.",
    "Tomato___Tomato_mosaic_virus": "Remove and destroy infected plants. Wash hands and tools.",
    "Healthy": "Crop is healthy! Continue regular watering and proper fertilization."
}

def save_to_history(data):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try: history = json.load(f)
            except: pass
    
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    history.append(data)
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# 🔐 SECURITY HELPER FUNCTION (युजर लॉगिन आहे की नाही ते चेक करते)
def is_logged_in(request: Request):
    return request.cookies.get("auth_session") == "admin_logged_in"

# --- 🔐 Login & Logout Routes ---
@app.get("/")
async def login_page(request: Request):
    # जर आधीच लॉगिन असेल तर डायरेक्ट home ला पाठवा
    if is_logged_in(request):
        return RedirectResponse(url="/home", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    # Static Username & Password
    if username == "admin" and password == "admin123":
        response = RedirectResponse(url="/home", status_code=303)
        # ✅ लॉगिन झाल्यावर ब्राउझरमध्ये 'Cookie' सेव्ह करणे
        response.set_cookie(key="auth_session", value="admin_logged_in", httponly=True)
        return response
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid Username or Password!"})

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    # ❌ लॉगआउट केल्यावर 'Cookie' डिलीट करणे
    response.delete_cookie("auth_session")
    return response

# --- 🏡 Protected Web Pages (इथे फक्त लॉगिन झाल्यावरच येता येईल) ---

@app.get("/home")
async def home(request: Request):
    if not is_logged_in(request): return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/info")
async def info(request: Request):
    if not is_logged_in(request): return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("info.html", {"request": request})

@app.get("/about")
async def about_page(request: Request):
    if not is_logged_in(request): return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/predict")
async def predict_page(request: Request):
    if not is_logged_in(request): return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("predict.html", {"request": request, "result": None})

@app.get("/history")
async def show_history(request: Request):
    if not is_logged_in(request): return RedirectResponse(url="/", status_code=303)
    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try: history_data = json.load(f)
            except: pass
    history_data = history_data[::-1] # Reverse list to show latest first
    return templates.TemplateResponse("history.html", {"request": request, "history": history_data})

@app.get("/delete_history/{timestamp}")
async def delete_history(request: Request, timestamp: str):
    if not is_logged_in(request): return RedirectResponse(url="/", status_code=303)
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try: history = json.load(f)
            except: history = []
        history = [item for item in history if item.get("timestamp") != timestamp]
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    return RedirectResponse(url="/history", status_code=303)

@app.post("/predict")
async def predict_disease(
    request: Request, 
    latitude: str = Form(None),
    longitude: str = Form(None),
    file: UploadFile = File(...)
):
    if not is_logged_in(request): return RedirectResponse(url="/", status_code=303)

    os.makedirs("static/uploads", exist_ok=True)
    file_path = f"static/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🚀 Image Preprocessing matches the Custom CNN / MobileNetV2 setup
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # ✅ Keeps it exactly matched with training!

    # Prediction
    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    
    # 🚀 Safe Name Retrieval & Formatting
    raw_disease_name = class_names.get(predicted_index, "Unknown_Disease")
    clean_disease_name = raw_disease_name.replace("___", " - ").replace("_", " ")

    # Recommendation Logic
    reco = solutions.get(raw_disease_name, "Consult a local agricultural expert for exact pesticide measurements.")
    if "healthy" in raw_disease_name.lower():
        reco = solutions.get("Healthy", "Crop is healthy! Continue regular watering.")

    # Weather API Logic
    temp, humidity = "N/A", "N/A"
    if latitude and longitude:
        try:
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m"
            response = requests.get(weather_url).json()
            temp = response["current"]["temperature_2m"]
            humidity = response["current"]["relative_humidity_2m"]
        except Exception:
            pass

    result_data = {
        "disease": clean_disease_name, 
        "confidence": round(confidence, 2),
        "recommendation": reco,
        "temperature": temp,      
        "humidity": humidity,     
        "image_url": f"/{file_path}"
    }

    save_to_history(result_data)
    return templates.TemplateResponse("predict.html", {"request": request, "result": result_data})
