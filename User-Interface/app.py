from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import cv2
import numpy as np
import tensorflow as tf  # or import torch if using PyTorch

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessary for flashing messages
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Folder to save uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  # Valid file types

# Define the class names and get_leaf_details function
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Add a dictionary with remedies for each disease
disease_remedies = {
    'Apple___Apple_scab': "Apply fungicides and remove infected leaves to prevent further spread.",
    'Apple___Black_rot': "Prune infected branches and use fungicides as recommended for apple black rot.",
    'Apple___Cedar_apple_rust': "Use rust-resistant varieties and apply fungicides during wet weather.",
    'Apple___healthy': "No treatment needed; your plant is healthy.",
    'Blueberry___healthy': "No treatment needed; your plant is healthy.",
    'Cherry_(including_sour)___Powdery_mildew': "Apply fungicides and remove infected leaves. Ensure proper airflow.",
    'Cherry_(including_sour)___healthy': "No treatment needed; your plant is healthy.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use resistant varieties and apply fungicides to control leaf spot.",
    'Corn_(maize)___Common_rust_': "Use resistant varieties and apply fungicides to manage common rust.",
    'Corn_(maize)___Northern_Leaf_Blight': "Rotate crops and use resistant varieties. Apply fungicides if necessary.",
    'Corn_(maize)___healthy': "No treatment needed; your plant is healthy.",
    'Grape___Black_rot': "Prune infected areas and apply fungicides as directed for grape black rot.",
    'Grape___Esca_(Black_Measles)': "Remove infected vines and apply proper irrigation practices.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use fungicides and improve air circulation around vines.",
    'Grape___healthy': "No treatment needed; your plant is healthy.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove infected trees to prevent spread. Use pest control to manage psyllids.",
    'Peach___Bacterial_spot': "Use disease-free stock, prune infected branches, and apply copper-based sprays.",
    'Peach___healthy': "No treatment needed; your plant is healthy.",
    'Pepper,_bell___Bacterial_spot': "Remove infected plants and use copper-based fungicides to prevent spread.",
    'Pepper,_bell___healthy': "No treatment needed; your plant is healthy.",
    'Potato___Early_blight': "Apply fungicides and practice crop rotation to manage early blight.",
    'Potato___Late_blight': "Use certified seed potatoes and fungicides, and practice crop rotation.",
    'Potato___healthy': "No treatment needed; your plant is healthy.",
    'Raspberry___healthy': "No treatment needed; your plant is healthy.",
    'Soybean___healthy': "No treatment needed; your plant is healthy.",
    'Squash___Powdery_mildew': "Apply fungicides and remove infected leaves. Ensure adequate air circulation.",
    'Strawberry___Leaf_scorch': "Remove infected leaves and apply fungicides. Ensure proper watering practices.",
    'Strawberry___healthy': "No treatment needed; your plant is healthy.",
    'Tomato___Bacterial_spot': "Remove infected plants, and use copper-based sprays to manage bacterial spot.",
    'Tomato___Early_blight': "Apply fungicides and practice crop rotation to manage early blight.",
    'Tomato___Late_blight': "Remove infected plants and apply fungicides. Avoid overhead watering.",
    'Tomato___Leaf_Mold': "Use resistant varieties and maintain proper air circulation.",
    'Tomato___Septoria_leaf_spot': "Prune infected leaves and apply fungicides as necessary.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use insecticidal soaps and encourage natural predators.",
    'Tomato___Target_Spot': "Apply fungicides and remove infected foliage. Ensure proper spacing.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants, control whiteflies, and plant resistant varieties.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants and disinfect tools to prevent spreading.",
    'Tomato___healthy': "No treatment needed; your plant is healthy."
}



def get_leaf_details(class_name):
    parts = class_name.split("___")
    plant_name = parts[0].replace("_", " ")
    disease_name = parts[1].replace("_", " ")
    is_healthy = 'healthy' in disease_name.lower()
    return plant_name, disease_name, is_healthy

# Try loading the model and log any errors
try:
    model = tf.keras.models.load_model('PlantAI_model.keras')
    app.logger.info("Model loaded successfully.")
except Exception as e:
    model = None
    app.logger.error(f"Error loading model: {e}")

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(filepath):
    try:
        image = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image_rgb, (224, 224))
        normalized_image = resized_image / 255.0
        return np.expand_dims(normalized_image, axis=0)
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

# Modify the upload_file function to include the remedy
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part. Please choose a file to upload.')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file. Please choose a file to upload.')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image and make a prediction
        processed_image = process_image(filepath)
        
        try:
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            prediction_class = class_names[predicted_class_index]

            # Extract details using the get_leaf_details function
            plant_name, disease_name, is_healthy = get_leaf_details(prediction_class)
            remedy = disease_remedies.get(prediction_class, "No remedy information available.")

            flash('File successfully uploaded and processed!')
            return render_template(
                'index.html',
                original_image=filename,
                plant_name=plant_name,
                disease_name=disease_name,
                is_healthy=is_healthy,
                remedy=remedy
            )
        
        except Exception as e:
            print("Prediction error:", str(e))
            flash('An error occurred during prediction.')
            return redirect(url_for('index'))
    else:
        flash('Invalid file format. Please upload a PNG, JPG, or JPEG file.')
        return redirect(url_for('index'))



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

