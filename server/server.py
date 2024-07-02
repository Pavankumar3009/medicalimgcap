# Dl imports
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image


# llm1 imports 
import json
from keras.models import Model, load_model # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
from keras.preprocessing.image import img_to_array # type: ignore
from keras.applications.inception_v3 import InceptionV3, preprocess_input  # type: ignore


#llm2 imports
from flask import Flask, request, jsonify# type: ignore
from werkzeug.utils import secure_filename # type: ignore
import os
from tensorflow.keras.preprocessing import image as keras_image # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input # type: ignore
from tensorflow.keras.applications import DenseNet121 # type: ignore
from tensorflow.keras.models import Model # type: ignore
import pandas as pd # type: ignore


app = Flask(__name__)
CORS(app)

# Deep Learning

loaded_model = tf.keras.models.load_model("accumodelpp84.h5")

# Define a route to handle image uploads and predictions
@app.route("/predictdl", methods=["POST","GET"])
def predict():
    print("got request for dl")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and file.filename:
        # Preprocess the image
        img = Image.open(file).convert('L')
        img = img.resize((224, 224))  # Resize to match model input size
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0  # Normalize pixel values

        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predicted_probs = loaded_model.predict(img_array)
        predicted_label = "Pneumothorax" if predicted_probs[0][0] > 0.5 else "No Pneumothorax"

        # Return prediction results
        return jsonify({"prediction": predicted_label, "probabilities": predicted_probs.tolist()})
    else:
        return jsonify({"error": "Invalid file format"})

#LLM

model = load_model('indianamodelTrue.keras')

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


# Load the DenseNet model architecture
chexnet = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3), pooling='avg')

# Load the weights separately
chex_weights = tf.keras.models.load_model('chexweights.h5')
chexnet.load_weights(chex_weights)

model = Model(chexnet.input, chexnet.layers[-1].output)

# Load the custom model
checkpoint_filepath = load_model('model_checkpoint.keras')
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

model_1_loaded = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'loss_function': loss_function})

# Load the word index mappings
t2 = pd.read_pickle('t2.pickle')
imp1 = {value: key for key, value in t2.word_index.items()}
imp2 = {key: value for key, value in t2.word_index.items()}

def load_and_preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

def beam(sentence):
    initial_state = model_1_loaded.layers[0].initialize_states(1)
    encoder_output = model_1_loaded.layers[0](sentence)
    result = ''
    sequences = [['<start>', initial_state, 0]]
    decoder_hidden_state = initial_state
    finished_seq = []
    beam_width = 3
    for i in range(76):
        all_candidates = []
        new_seq = []
        for s in sequences:
            cur_vec = np.reshape(imp2[s[0].split(" ")[-1]], (1, 1))
            decoder_hidden_state = s[1]
            op, hs, attention_weights, context_vector = model_1_loaded.layers[1].onestep(cur_vec, encoder_output, decoder_hidden_state)
            op = tf.nn.softmax(op)
            top3 = np.argsort(op).flatten()[-beam_width:]
            for t in top3:
                candidates = [s[0] + ' ' + imp1[t], hs, s[2] - np.log(np.array(op).flatten()[t])]
                all_candidates.append(candidates)
        sequences = sorted(all_candidates, key=lambda l: l[2])[:beam_width]
        count = 0
        for s1 in sequences:
            if s1[0].split(" ")[-1] == '<end>':
                s1[2] = s1[2] / len(s1[0])  # normalized
                finished_seq.append([s1[0], s1[1], s1[2]])
                count += 1
            else:
                new_seq.append([s1[0], s1[1], s1[2]])
        beam_width -= count
        sequences = new_seq
        if not sequences:
            break
        else:
            continue
    if len(finished_seq) > 0:
        sequences = finished_seq[-1]
        return sequences[0]
    else:
        return new_seq[-1][0]

@app.route('/predict', methods=['POST'])
def predict_caption_for_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)

        img_features = load_and_preprocess_image(filepath)
        prediction = beam(img_features)
        os.remove(filepath)
        return jsonify({'predicted_caption': prediction})

if __name__ == "__main__":
    app.run(debug=True )
