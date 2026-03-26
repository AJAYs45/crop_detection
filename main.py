from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os
import shutil
import requests
from datetime import datetime
import urllib.parse

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = 'crop_model.h5'
LABELS_PATH = 'class_indices.json'
HISTORY_FILE = 'history.json'
USERS_FILE = 'users.json'  

print("Loading Model... Please wait.")
model = tf.keras.models.load_model(MODEL_PATH)

# SMART LABEL LOADER
with open(LABELS_PATH, 'r') as f:
    raw_labels = json.load(f)
    if list(raw_labels.keys())[0].isdigit():
        class_names = {int(k): v for k, v in raw_labels.items()}
    else:
        class_names = {v: k for k, v in raw_labels.items()}

print("Model and Labels Loaded Successfully!")

solutions = {
    "Pepper,_bell___Bacterial_spot": "Use Copper-based fungicides. Ensure proper spacing between plants.",
    "Tomato___Early_blight": "Apply Chlorothalonil or Copper fungicide. Remove infected leaves.",
    "Potato___Late_blight": "Apply Mancozeb fungicide. Do not overwater the crops.",
    "Tomato___Tomato_mosaic_virus": "Remove and destroy infected plants. Wash hands and tools.",
    "Healthy": "Crop is healthy! Continue regular watering and proper fertilization."
}

# 🚀 USER MANAGEMENT LOGIC
def get_users():
    if not os.path.exists(USERS_FILE):
        return {"admin": {"password": "admin123", "fullname": "Admin", "role": "admin"}}
    with open(USERS_FILE, "r") as f:
        try: return json.load(f)
        except: return {}

def is_logged_in(request: Request):
    return request.cookies.get("auth_session") == "admin_logged_in"

@app.get("/")
async def login_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse(url="/home", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": None, "success": None})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    users = get_users()
    user_data = users.get(username)
    
    is_valid = False
    # Handle new dict format or legacy string format
    if isinstance(user_data, dict) and user_data.get("password") == password:
        is_valid = True
    elif isinstance(user_data, str) and user_data == password:
        is_valid = True

    if is_valid:
        response = RedirectResponse(url="/home", status_code=303)
        response.set_cookie(key="auth_session", value="admin_logged_in", httponly=True) 
        return response
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid Username or Password!", "success": None})

@app.get("/register")
async def register_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse(url="/home", status_code=303)
    return templates.TemplateResponse("register.html", {"request": request, "error": None})

# 🚀 UPDATED: Register route with new Agri-specific fields
@app.post("/register")
async def register(
    request: Request, 
    fullname: str = Form(...),
    age: int = Form(...),
    city: str = Form(...),
    soil_type: str = Form(...),
    username: str = Form(...), 
    password: str = Form(...), 
    confirm_password: str = Form(...)
):
    users = get_users()
    
    if username in users:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already exists!"})
    
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Passwords do not match!"})
    
    # Save user with complete profile data
    users[username] = {
        "password": password,
        "fullname": fullname,
        "age": age,
        "city": city,
        "soil_type": soil_type,
        "joined_date": datetime.now().strftime("%Y-%m-%d")
    }
    
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)
        
    return templates.TemplateResponse("login.html", {"request": request, "error": None, "success": "Account created successfully! Please login."})

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("auth_session")
    return response

# --- YOUR EXISTING APP ROUTES ---
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
    return templates.TemplateResponse("predict.html", {"request": request, "result": None, "error": None})

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

@app.get("/history")
async def show_history(request: Request):
    if not is_logged_in(request): return RedirectResponse(url="/", status_code=303)
    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try: history_data = json.load(f)
            except: pass
    history_data = history_data[::-1]
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

    os.makedirs(os.path.join("static", "uploads"), exist_ok=True)
    file_path = f"static/uploads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    
    if confidence < 50.0:
        os.remove(file_path) 
        error_msg = "Invalid Image! Please upload a clear photo of a crop leaf or fruit."
        return templates.TemplateResponse(request=request, name="predict.html", context={"request": request, "result": None, "error": error_msg})

    raw_disease_name = class_names.get(predicted_index, "Unknown_Disease")
    clean_disease_name = raw_disease_name.replace("___", " - ").replace("_", " ")

    reco = solutions.get(raw_disease_name, "Consult a local agricultural expert for exact pesticide measurements.")
    if "healthy" in raw_disease_name.lower():
        reco = solutions.get("Healthy", "Crop is healthy! Continue regular watering.")

    temp, humidity = "N/A", "N/A"
    if latitude and longitude:
        try:
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m"
            response = requests.get(weather_url).json()
            temp = response["current"]["temperature_2m"]
            humidity = response["current"]["relative_humidity_2m"]
        except Exception:
            pass

    if "healthy" in raw_disease_name.lower():
        search_term = raw_disease_name.split("___")[0].replace("_", " ") + " plant"
    else:
        search_term = raw_disease_name.replace("___", " ").replace("_", " ")
        
    wiki_url = f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(search_term)}"

    result_data = {
        "disease": clean_disease_name, 
        "confidence": round(confidence, 2),
        "recommendation": reco,
        "temperature": temp,      
        "humidity": humidity,     
        "image_url": f"/{file_path}",
        "wiki_url": wiki_url
    }

    save_to_history(result_data)
<<<<<<< HEAD
    return templates.TemplateResponse("predict.html", {"request": request, "result": result_data})
=======
    return templates.TemplateResponse(request=request, name="predict.html", context={"request": request, "result": result_data, "error": None})
>>>>>>> b70cfa073 (updated code)
