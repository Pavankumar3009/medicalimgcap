import os
import uuid
# Dl imports
from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore

# llm1 imports 
import json
from keras.models import Model, load_model # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
from keras.preprocessing.image import img_to_array # type: ignore
from keras.applications.inception_v3 import InceptionV3, preprocess_input  # type: ignore

app = Flask(__name__)
CORS(app)

# Deep Learning

loaded_model = tf.keras.models.load_model(r"C:\Users\91879\Desktop\project\Medicalimgcap\server\accumodelpp84.h5")

def preprocess_image(image_path):
    original_img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return original_img, img_array

# Define predict_image function
def predict_image(image_path, model):
    original_img, img_array = preprocess_image(image_path)
    predicted_probs = model.predict(img_array)
    predicted_label = "Pneumothorax" if predicted_probs[0][0] > 0.5 else "No Pneumothorax"
    return predicted_label, predicted_probs


# Define a route to handle image uploads and predictions
@app.route("/predictdl", methods=["POST","GET"])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # If file exists and is allowed extension
    if file:
        # Define a temporary directory
        temp_dir = r"Medicalimgcap/server/tempf"
        
        # Ensure the directory exists
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate a unique filename
        temp_filename = os.path.join(temp_dir, str(uuid.uuid4()) + "_" + file.filename)
        
        # Save the file
        file.save(temp_filename)
        
        # Make prediction
        prediction, probabilities = predict_image(temp_filename, loaded_model)
        
        # Clean up the temporary file if necessary
        os.remove(temp_filename)
        
        # Return prediction result
        return jsonify({
            'prediction': prediction,
            'probabilities': probabilities.tolist()
        })
#LLM

model = load_model(r'C:\Users\91879\Desktop\project\Medicalimgcap\server\indianamodelTrue.keras')

# Load necessary pre-processing objects (wordtoix, ixtoword, max_length)
with open('ixtoword.json', 'r') as f:
    ixtoword = json.load(f)

with open('wordtoix.json', 'r') as f:
    wordtoix = json.load(f)

ixtoword = {int(k): v for k, v in ixtoword.items()}
wordtoix = {k: int(v) for k, v in wordtoix.items()}

max_length = 40
vocab_size = 408

base_model = InceptionV3(weights='imagenet') 
model1 = Model(base_model.input, base_model.layers[-2].output)


def preprocess_img(image): 
    # inception v3 excepts img in 299 * 299 * 3 
    image = image.convert('RGB')
    img = image.resize((299, 299))
    x = img_to_array(img) 
    # Add one more dimension 
    x = np.expand_dims(x, axis=0) 
    x = preprocess_input(x) 
    return x 

def encode(image): 
    image = preprocess_img(image) 
    vec = model1.predict(image) 
    vec = np.reshape(vec, (vec.shape[1])) 
    return vec 

@app.route('/predictllm', methods=['POST',"GET"])
def predictllm():
    try:
        file = request.files['file']
        img = Image.open(file)
        pic = encode(img).reshape(1,2048)
        start = 'startseq'
        for i in range(max_length):
            seq = [wordtoix[word] for word in start.split() if word in wordtoix]
            seq = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([pic, seq])
            yhat = np.argmax(yhat, axis=-1)
            word = ixtoword[yhat[0]]
            start += ' ' + word
            if word == 'endseq':
                break
        final = start.split()
        final = final[1:-1]
        final = ' '.join(final)

        return jsonify({'caption': final})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True )