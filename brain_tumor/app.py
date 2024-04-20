from flask import Flask, render_template, request
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes
import joblib
import cv2  
import os  
from werkzeug.utils import secure_filename

# Load the trained SVM model
model = joblib.load('svm.pkl')

# Function to calculate convexity value for a given slice
def calculate_convexity(slice_data):
    # Your convexity calculation logic here
    # This could involve image processing techniques like edge detection and contour analysis
    
    # For demonstration, let's assume a simple calculation
    # Here, we'll just count the number of connected components
    filled_slice = binary_fill_holes(slice_data)
    num_connected_components = np.max(filled_slice)
    convexity = num_connected_components

    return convexity

def calculate_contour_area(slice_data):
    contours, _ = cv2.findContours(slice_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the area of the largest contour
    contour_area = 0
    if contours:
        contour_area = cv2.contourArea(max(contours, key=cv2.contourArea))
    
    return contour_area

def calculate_convex_hull_area(slice_data):
    contours, _ = cv2.findContours(slice_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the convex hull area of the largest contour
    convex_hull_area = 0
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        convex_hull_area = cv2.contourArea(hull)
    
    return convex_hull_area

def preprocess_image(slice_data):
    resized_image = cv2.resize(slice_data, (128, 128))
    
    # Normalize the pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/convexity.html')
def convexity():
    return render_template('convexity.html')

@app.route('/solidity.html')
def solidity():
    return render_template('solidity.html')

@app.route('/', methods=['POST'])
def analyze_slice():
    if request.method == 'POST':
        # Get the uploaded file and slice number from the form
        seg_file = request.files['seg_file']
        slice_number = int(request.form['slice_number'])
        
        # Save the uploaded file to a temporary location
        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        seg_file_path = os.path.join(upload_dir, secure_filename(seg_file.filename))
        seg_file.save(seg_file_path)
        
        # Load the NIfTI file using its path
        nifti_image = nib.load(seg_file_path)
        seg_data = nifti_image.get_fdata()
        
        # Extract the specified slice
        slice_data = seg_data[:, :, slice_number]
        
        # Calculate convexity for the slice
        convexity_value = calculate_convexity(slice_data)
        contour_area = calculate_contour_area(slice_data)
        convex_hull_area = calculate_convex_hull_area(slice_data)
        
        # Combine features into a single array
        features = np.array([[convexity_value, contour_area, convex_hull_area]])
        
        # Apply the trained model to classify the features
        predicted_value = model.predict(features)[0]
        
        # Classify based on the predicted value and threshold
        if predicted_value > 0.5:
            classification = 1
        else:
            classification = 0
        
        # Determine the shape based on classification
        if classification == 1:
            shape = "More Circular"
        else:
            shape = "Irregular Boundary"
        
        # Pass data to the template for rendering
        return render_template('result.html', slice_number=slice_number, classification=classification, shape=shape)
    
    return render_template('convexity.html')



if __name__ == "__main__":
    app.run(debug=True)
