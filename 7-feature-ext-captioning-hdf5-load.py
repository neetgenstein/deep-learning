# pip install tensorflow==2.12.0

import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D

# Load the tokenizer
with open("/kaggle/input/tokenizer-updated/tensorflow2/default/1/tokenizer_caption_inceptionv3_oct.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the full trained caption model (.hdf5)
captionModel = load_model("/kaggle/input/captioning-inceptionv3-full/tensorflow2/default/1/caption_inceptionv3_oct.hdf5")
print("Full caption model (.hdf5) loaded successfully!")

# Recreate the InceptionV3-based feature extractor
base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(base.output)
feature_extractor = Model(inputs=base.input, outputs=x)
print("Feature extractor ready!")

# Helper: encode image into 2048-dim feature vector
def extract_features(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)
    return features  # shape (1, 2048)

# Generate caption
def generate_caption(photo_features, max_length=7):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = captionModel.predict([photo_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = next((w for w, idx in tokenizer.word_index.items() if idx == yhat), None)
        if word is None:
            break
        in_text += ' ' + word
        if word == '<end>':
            break
    return in_text

# Example: test image
img_path = "/kaggle/input/oct2017/dataset/val/NORMAL/NORMAL-1016042-63.jpeg"
photo_features = extract_features(img_path)
caption = generate_caption(photo_features)
print("Generated Caption:", caption)