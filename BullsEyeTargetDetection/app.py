from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# ... (your existing code)

# Rectify Sparse Softmax Cross Entropy Warning
tf.compat.v1.losses.sparse_softmax_cross_entropy

# Rectify Executing Eagerly Outside Functions Warning
tf.compat.v1.executing_eagerly_outside_functions

# Rectify Max Pooling Warning
tf.nn.max_pool2d

# ... (continue with your existing code)


app = Flask(__name__)

# Load the pre-trained model
model_path = 'C:\\Users\\Pavan\\BullsEyeTargetDetection\\best_model1.h5'  # Update with the correct path to your best model
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_array):
    # Assuming you need to resize the image to match the input size of the model (128x128)
    resized_image = cv2.resize(image_array, (128, 128))

    # Normalize the pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0

    # Expand dimensions to create a batch of size 1
    preprocessed_image = np.expand_dims(normalized_image, axis=0)

    return preprocessed_image

def main(image_array):
    if image_array is None:
        print('Error opening image!!')
        return -1

    # Preprocess the image
    preprocessed_image = preprocess_image(image_array)

    # Perform Bulls Eye detection using the loaded model
    predictions = model.predict(preprocessed_image)

    # Process the predictions as needed
    # For example, draw circles or annotate the image based on the predictions
    # Process the predictions as needed
    if isinstance(predictions, np.ndarray):
        detected_coordinates = predictions[0]

    # Draw circles on the original image based on detected coordinates
        for coord in detected_coordinates:
            center = (int(coord[0]), int(coord[1]))
            radius = int(coord[2])
            cv2.circle(image_array, center, radius, (0, 255, 0), 2)

    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get image file from the request
            file = request.files['file']

            # Convert image to numpy array
            image = Image.open(file)
            image_array = np.array(image)

            # Call the main function to process the image with Bulls Eye detection model
            processed_image = main(image_array)

            # Save the processed image to a temporary file
            temp_file_path = '/content/temp_processed_image.jpg'
            cv2.imwrite(temp_file_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

            # Return the processed image path to display in predict.html
            return render_template('predict.html', image_path=temp_file_path)

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('predict.html', image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
