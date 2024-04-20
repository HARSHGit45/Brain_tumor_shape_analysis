from flask import Flask, render_template, request
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes
import joblib
import cv2  
import os  
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

# Load the trained SVM model
model = joblib.load('svm.pkl')

# Function to calculate convexity value for a given slice


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
    def calculate_shape(slice_data):
        contours, _ = cv2.findContours((slice_data > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        convexity_values = []

        if len(contours) > 0:
            convex_hull = cv2.convexHull(contours[0])
            contour_area = cv2.contourArea(contours[0])
            convex_hull_area = cv2.contourArea(convex_hull)

            if convex_hull_area > 0:
                convexity = contour_area / convex_hull_area
                convexity_values.append(convexity)
            else:
                convexity_values.append(0)
        else:
            convexity_values.append(0)
        
        return convexity_values

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
        
        # Calculate shape parameters
        convexity_values = calculate_shape(slice_data)

        # Plotting code
        contours, _ = cv2.findContours((slice_data > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            convex_hull = cv2.convexHull(contours[0])

            plt.figure()
            plt.imshow(slice_data, cmap='gray')
            plt.contour(slice_data, levels=[0.5], colors='r') 
            plt.plot(convex_hull[:, 0, 0], convex_hull[:, 0, 1], 'g', linewidth=2) 
            plt.title(f"Slice {slice_number}, Convexity: {convexity_values[-1]}")
            # Save the plot as PNG
            plot_filename = os.path.join('static', 'plot.png')
            plt.savefig(plot_filename)

            # Pass data to the template for rendering
            return render_template('result.html', slice_number=slice_number, convexity=convexity_values[-1], plot_filename=plot_filename)
    
    
    return render_template('convexity.html')



if __name__ == "__main__":
    app.run(debug=True)
