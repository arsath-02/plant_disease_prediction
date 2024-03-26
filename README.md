# Plant Disease Prediction Using CNN Model

This project aims to predict plant diseases using a Convolutional Neural Network (CNN) model. The CNN model is trained on a dataset of images of healthy and diseased plants. By leveraging deep learning techniques, the model can accurately identify the presence of diseases in plants based on input images.

## Technologies Used

- Python: The backend of the application is developed using Python programming language.
- Flask: Flask is used as the web framework to serve the prediction model as a RESTful API.
- HTML: The frontend interface is developed using HTML for user interaction.
- Convolutional Neural Network (CNN): CNN is a deep learning architecture used for image recognition tasks.

## How It Works

1. **Training the CNN Model**: The CNN model is trained on a dataset consisting of images of healthy plants and plants affected by various diseases. During training, the model learns to extract relevant features from the images and classify them into different disease categories.

2. **Integration with Flask**: The trained CNN model is integrated into a Flask application, which serves as the backend. Flask provides endpoints to receive image data, preprocess it, and pass it to the CNN model for prediction.

3. **User Interface**: The frontend of the application is implemented using HTML, allowing users to interact with the system. Users can upload images of plants through the web interface, and the application provides predictions regarding the presence of diseases in the uploaded images.

## Running the Application

To run the application locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required Python dependencies using `pip install -r requirements.txt`.
3. Navigate to the project directory and run `python app.py`.
4. Once the Flask server is running, open your web browser and go to `http://localhost:5000` to access the application.

## Contributing

Contributions to this project are welcome. If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
