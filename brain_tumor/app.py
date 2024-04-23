from flask import Flask, render_template, request
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes
import joblib
import cv2  
import uuid
import os  
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import matplotlib
matplotlib.use('Agg')
import base64
from werkzeug.utils import secure_filename




convex = load_model('models/convexity.h5')
solid = load_model('models/solidity.h5')
rectangle = load_model('models/rectangularity.h5')
with open('models/eccentricity.pkl', 'rb') as model_file:
    eccent = pickle.load(model_file)
circle = load_model('models/Circular.h5')










def preprocess_image(slice_data):
    resized_image = cv2.resize(slice_data, (128, 128))
    normalized_image = resized_image / 255.0

app = Flask(__name__)





@app.route('/')
def home():
    return render_template("home.html")






@app.route('/about.html')
def about():
    return render_template('about.html')






@app.route('/model_info.html')
def model_info():
    # Here you can add any dynamic data to pass to the template
    return render_template('model_info.html')



@app.route('/info.html')
def info():
    return render_template('info.html')







@app.route('/convexity.html')
def convexity():
    return render_template('convexity.html')






@app.route('/solidity.html')
def solidity():
    return render_template('solidity.html')






@app.route('/rectangularity.html')
def rectangularity():
    return render_template('rectangularity.html')






@app.route('/eccentricity.html')
def eccentricity():
    return render_template('eccentricity.html')







@app.route('/circularity.html')
def circularity():
    return render_template('circularity.html')







@app.route('/analyze_convexity', methods=['POST'])
def analyze_convexity():
    def calculate_shape(slice_data):
        contours, _ = cv2.findContours((slice_data > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, 0  # Return zero values if no contours found

        largest_contour = max(contours, key=cv2.contourArea)
        convex_hull = cv2.convexHull(largest_contour)
        contour_area = cv2.contourArea(largest_contour)
        convex_hull_area = cv2.contourArea(convex_hull)

        if convex_hull_area > 0:
            convexity = contour_area / convex_hull_area
        else:
            convexity = 0

        return contour_area, convex_hull_area, convexity, convex_hull

    if request.method == 'POST':
        seg_file = request.files['seg_file']
        slice_number = int(request.form['slice_number'])

        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        seg_file_path = os.path.join(upload_dir, secure_filename(seg_file.filename))
        seg_file.save(seg_file_path)
        
        nifti_image = nib.load(seg_file_path)
        seg_data = nifti_image.get_fdata()
        slice_data = seg_data[:, :, slice_number]

        contour_area, convex_hull_area, convexity, convex_hull = calculate_shape(slice_data)

        if convex_hull is not None:
            # Plotting the results
            plt.figure()
            plt.imshow(slice_data, cmap='gray')
            plt.contour(slice_data, levels=[0.5], colors='r') 
            plt.plot([pt[0] for pt in convex_hull[:, 0]], [pt[1] for pt in convex_hull[:, 0]], 'g', linewidth=2) 
            plt.title(f"Slice {slice_number}, Convexity: {convexity}")
            plot_filename = os.path.join('static', 'plot.png')
            plt.savefig(plot_filename)
            plt.close()

            input_features = np.array([[contour_area, convex_hull_area, convexity]])
            prediction = convex.predict(input_features)
            prediction_result = "Concave (irregular boundaries)" if prediction[0][0] == 0 else "Convex (regular boundaries)"

            return render_template('result.html', slice_number=slice_number, convexity=convexity, prediction=prediction_result, plot_filename=plot_filename)
        else:
            # Handle case where no valid contour or convex hull could be calculated
            error_message = "No valid contours found. Unable to analyze convexity."
            return render_template('error.html', error_message=error_message)

    return render_template('convexity.html')








@app.route('/analyze_solidity', methods=['POST'])
def analyze_solidity():
    def calculate_solidity(slice_data):
        contours, _ = cv2.findContours((slice_data > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            convex_hull = cv2.convexHull(largest_contour)
            convex_hull_area = cv2.contourArea(convex_hull)
            if convex_hull_area > 0:
                solidity = contour_area / convex_hull_area
                return solidity, convex_hull_area, convex_hull
        return 0, 0, None

    if request.method == 'POST':
        seg_file = request.files['seg_file']
        slice_number = int(request.form['slice_number'])

        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        seg_file_path = os.path.join(upload_dir, secure_filename(seg_file.filename))
        seg_file.save(seg_file_path)

        nifti_image = nib.load(seg_file_path)
        seg_data = nifti_image.get_fdata()
        slice_data = seg_data[:, :, slice_number]

        solidity, convex_hull_area, convex_hull = calculate_solidity(slice_data)

        if convex_hull is not None:
            plt.figure()
            plt.imshow(slice_data, cmap='gray')
            plt.contour(slice_data, levels=[0.5], colors='r')
            plt.plot([pt[0] for pt in convex_hull[:, 0]], [pt[1] for pt in convex_hull[:, 0]], 'g', linewidth=2)
            plt.title(f"Slice {slice_number}, Solidity: {solidity}")
            plot_filename = os.path.join('static', f'plot_{uuid.uuid4()}.png')
            plt.savefig(plot_filename)
            plt.close()

            input_features = np.array([[solidity, convex_hull_area]])
            prediction = solid.predict(input_features)
            prediction_result = "Hollow" if prediction[0][0] == 0 else "Dense"

            return render_template('result_s.html', slice_number=slice_number, solidity=solidity, prediction=prediction_result, plot_filename=plot_filename)
        else:
            error_message = "No valid contours found. Unable to analyze solidity."
            return render_template('error.html', error_message=error_message)

    return render_template('solidity.html')






@app.route('/analyze_rectangularity', methods=['POST'])
def analyze_rectangularity():
    def calculate_rectangularity(slice_data):
        contours, _ = cv2.findContours((slice_data > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            contour_area = cv2.contourArea(contours[0])
            bounding_box_area = w * h
            if bounding_box_area > 0:
                rectangularity = contour_area / bounding_box_area
                return rectangularity, x, y, w, h
        return 0, 0, 0, 0, 0

    if request.method == 'POST':
        seg_file = request.files['seg_file']
        slice_number = int(request.form['slice_number'])
        
        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        seg_file_path = os.path.join(upload_dir, secure_filename(seg_file.filename))
        seg_file.save(seg_file_path)
        
        nifti_image = nib.load(seg_file_path)
        seg_data = nifti_image.get_fdata()
        slice_data = seg_data[:, :, slice_number]
        
        rectangularity, x, y, w, h = calculate_rectangularity(slice_data)

        if w > 0 and h > 0:  # Ensure we have valid bounding box dimensions
            plt.figure()
            plt.imshow(slice_data, cmap='gray')
            plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], 'b', linewidth=2)  # Draw bounding box
            plt.title(f"Slice {slice_number}, Rectangularity: {rectangularity}")
            plot_filename = os.path.join('static', f'plot_{uuid.uuid4()}.png')
            plt.savefig(plot_filename)
            plt.close()

            input_features = np.array([[rectangularity, x, y, w, h]])
            prediction = rectangle.predict(input_features)
            prediction_result = "Rectangular and regular boundaries" if prediction[0][0] == 1 else "Not rectangular and irregular boundaries"

            return render_template('result_r.html', slice_number=slice_number, rectangularity=rectangularity, prediction=prediction_result, plot_filename=plot_filename)
        else:
            error_message = "No valid bounding box found. Unable to analyze rectangularity."
            return render_template('error.html', error_message=error_message)

    return render_template('rectangularity.html')








@app.route('/analyze_eccentricity', methods=['POST'])
d+ef analyze_eccentricity():
    def calculate_eccentricity(slice_data):
        contours, _ = cv2.findContours((slice_data > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0 and len(contours[0]) >= 5:
            ellipse = cv2.fitEllipse(contours[0])
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            if major_axis != 0:  # Prevent division by zero
                eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
                return eccentricity, major_axis, minor_axis
        return 0, 0, 0

    if request.method == 'POST':
        seg_file = request.files['seg_file']
        slice_number = int(request.form['slice_number'])

        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        seg_file_path = os.path.join(upload_dir, secure_filename(seg_file.filename))
        seg_file.save(seg_file_path)

        nifti_image = nib.load(seg_file_path)
        seg_data = nifti_image.get_fdata()
        slice_data = seg_data[:, :, slice_number]

        eccentricity, major_axis, minor_axis = calculate_eccentricity(slice_data)

        if major_axis > 0 and minor_axis > 0:  # Ensure valid dimensions
            input_features = np.array([[eccentricity, major_axis, minor_axis]])
            prediction = eccent.predict(input_features)
            prediction_result = "Regular boundaries(Less Spreaded)" if prediction[0] == 1 else "Irregular boundaries(More Spreaded)"

            plt.figure(figsize=(8, 6))
            plt.imshow(slice_data, cmap='gray')
            plt.title(f"Slice {slice_number}, Eccentricity: {eccentricity}")
            plt.colorbar(label='Intensity')
            plt.axis('off')
            plot_filename = os.path.join('static', f'plot_{uuid.uuid4()}.png')
            plt.savefig(plot_filename)

            return render_template('result_e.html', slice_number=slice_number, eccentricity=eccentricity, prediction=prediction_result, plot_filename=plot_filename)
        else:
            error_message = "Not enough data to calculate eccentricity."
            return render_template('error.html', error_message=error_message)

    return render_template('eccentricity.html')












@app.route('/analyze_circularity', methods=['POST'])
def analyze_circularity():
    def calculate_circularity(slice_data):
        contours, _ = cv2.findContours((slice_data > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            return circularity, area, perimeter
        return 0, 0, 0

    if request.method == 'POST':
        seg_file = request.files['seg_file']
        slice_number = int(request.form['slice_number'])

        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        seg_file_path = os.path.join(upload_dir, secure_filename(seg_file.filename))
        seg_file.save(seg_file_path)
        
        nifti_image = nib.load(seg_file_path)
        seg_data = nifti_image.get_fdata()
        slice_data = seg_data[:, :, slice_number]
        
        circularity, area, perimeter = calculate_circularity(slice_data)

        if circularity > 0 and area > 0 and perimeter > 0:
            input_features = np.array([[circularity, area, perimeter]])
            prediction = circle.predict(input_features)
            prediction_result = "Circular and  has regular boundaries" if prediction[0][0] == 1 else "Not circular and  has irregular boundaries"

            plt.figure(figsize=(8, 6))
            plt.imshow(slice_data, cmap='gray')
            plt.title(f"Slice {slice_number}, Circularity: {circularity}")
            plt.colorbar(label='Intensity')
            plt.axis('off')
            plot_filename = os.path.join('static', f'plot_{uuid.uuid4()}.png')
            plt.savefig(plot_filename)

            return render_template('result_c.html', slice_number=slice_number, circularity=circularity, prediction=prediction_result, plot_filename=plot_filename)
        else:
            error_message = "Not enough data to calculate circularity."
            return render_template('error.html', error_message=error_message)

    return render_template('circularity.html')












if __name__ == "__main__":
    app.run(debug=False)
