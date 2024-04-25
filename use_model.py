import numpy as np
import tensorflow as tf
from pickle import load

load_model = tf.keras.models.load_model
preprocess_input = tf.keras.applications.inception_v3.preprocess_input
image = tf.keras.preprocessing.image
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

class ImageCaptioningModel:
    def __init__(self, model_path, feature_extraction_path):
        self.model_path = model_path
        self.feature_extraction_path = feature_extraction_path


    def load_model_and_tokenizer(self):
        # Load the pre-trained model
        self.model = load_model(self.model_path)

        # Load the tokenizer
        with open(self.feature_extraction_path, 'rb') as f:
            self.tokenizer = load(f)

        # Maximum sequence length (from training)
        self.max_length = self.model.layers[0].output_shape[1]

    def generate_caption(self, image_url):
        # Load and preprocess the image
        img = image.load_img(image_url, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Extract features
        features = self.model.layers[1](img)
        features = self.model.layers[2](features)
        features = self.model.layers[3](features)

        # Generate caption
        in_text = 'startseq'
        for i in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = self.model.predict([features, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.tokenizer.index_word[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break
        caption = in_text.split()
        caption = ' '.join(caption[1:-1])  # Remove 'startseq' and 'endseq'
        return caption