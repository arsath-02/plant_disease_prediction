from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load the saved model
loaded_model = load_model("finetuned_model.h5")

# Path to training data directory
training_data_path = "D:\\KEC PROJECTS\\2 YEAR\\Hack Sphere\\New folder\\New Plant Diseases Dataset(Augmented)\\New Plant Diseases Dataset(Augmented)\\train"

# Get the class labels
class_labels = sorted(os.listdir(training_data_path))

# Define a threshold for similarity percentage
threshold = 70  # Adjust as needed

# Load and preprocess the training images for similarity comparison
training_images = []
for label in class_labels:
    img_path = os.path.join(training_data_path, label)  
    images = os.listdir(img_path)
    for image_name in images:
        image_path = os.path.join(img_path, image_name)
        img = image.load_img(image_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0
        training_images.append(img_array)
training_images = np.array(training_images)

@app.route('/')
def home():
    return render_template('finalindex.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        # Save the file to ./static/uploads
        img_filename = file.filename
        file_path = os.path.join('static', 'uploads', img_filename)
        file.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = loaded_model.predict(img_array)

        # Decode the predictions
        class_index = np.argmax(predictions)
        predicted_class = class_labels[class_index]

        # Calculate similarity between user input and training data
        similarities = cosine_similarity(img_array.reshape(1, -1), training_images.reshape(len(class_labels) * len(os.listdir(training_data_path)), -1))
        similarity_percentage = similarities[0, class_index] * 100

        if similarity_percentage >= threshold:
            recommendation = "Recommendation for " + predicted_class
        else:
            recommendation = "Low similarity with training data"

        return render_template('finalresult.html', prediction=predicted_class, recommendation=recommendation, 
                               similarity_percentage=similarity_percentage, img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
