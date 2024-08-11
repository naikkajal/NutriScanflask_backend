from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import logging

app = Flask(__name__)


model_path = r"C:\Users\KAJAL NAIK\Downloads\MobileNet_V2.h5"

model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'})
    
    try:
        app.logger.info('File received: %s', file.filename)
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        result = predictions.tolist()
        app.logger.info('Prediction result: %s', result)
        
        return jsonify({'predictions': result})
    except Exception as e:
        app.logger.error('Error processing image: %s', str(e))
        return jsonify({'error': 'Error processing image'})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
