from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load VGG model for image preprocessing
vgg_model_path = r"./base/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
vgg_model = VGG16(weights=vgg_model_path, include_top=True, input_shape=(224, 224, 3))

# Remove the last layer of the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# loading tokenizer
with open('./working/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

vocab_size = len(tokenizer.word_index) + 1

# maximum length of captions
MAX_LENGTH = 35


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text


def predict_real(image_url, model):
    # Download the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Resize the image to match VGG input size
    image = image.resize((224, 224))

    # Convert the image to RGB if it's not already in that mode
    image = image.convert("RGB")

    # Convert image pixels to numpy array and normalize
    image = img_to_array(image) / 255.0

    # Reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Preprocess image for VGG
    image = preprocess_input(image)

    # Extract features
    feature = vgg_model.predict(image, verbose=0)

    # Predict from the trained model
    y_pred = predict_caption(model, feature, tokenizer, MAX_LENGTH)

    return y_pred




# Load the English model
model_eng_path = r"./working/models/best_model_LSTM_1.h5"  # Update with your model path
model_eng = load_model(model_eng_path)

# Load the Arabic model
model_ar_path = r"./working/models/best_model_LSTM_2.h5"  # Update with your model path
model_ar = load_model(model_eng_path)


# API endpoint for English caption generation
@app.route("/api/generate_caption_eng", methods=["POST"])
def generate_caption_api_eng():
    data = request.get_json()
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    # Generate caption
    caption = predict_real(image_url, model_eng)

    return jsonify({"caption": caption})


# API endpoint for Arabic caption generation
@app.route("/api/generate_caption_ar", methods=["POST"])
def generate_caption_api_ar():
    data = request.get_json()
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    # Generate caption
    caption = predict_real(image_url, model_ar)

    return jsonify({"caption": caption})


if __name__ == "__main__":
    app.run()
