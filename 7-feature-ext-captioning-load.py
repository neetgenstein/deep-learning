import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the tokenizer
with open("/kaggle/input/feature-extraction-captioning/tensorflow2/default/1/tokenizer_caption_inceptionv3_oct.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = 7  # same as during training

# Rebuild the caption model architecture
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = LSTM(256, use_cudnn=False)(se1)

decoder1 = Concatenate()([fe2, se2])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

captionModel = Model(inputs=[inputs1, inputs2], outputs=outputs)

# Load trained weights
captionModel.load_weights("/kaggle/input/feature-extraction-captioning/tensorflow2/default/1/caption_inceptionv3_oct.weights.h5")
print("Caption model weights loaded successfully!")

# Recreate feature extractor for image encoding
base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
x = GlobalAveragePooling2D()(base.output)
feature_extractor = Model(inputs=base.input, outputs=x)
print("Feature extractor ready!")

# Helper: encode image into 2048-dim features
def extract_features(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)
    return features  # shape (1, 2048)

# Generate caption
def generate_caption(photo_features):
    in_text = '<start>'  # whatever you used as start token
    for i in range(max_length):
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

# Run on a test image
img_path = "/kaggle/input/oct2017/dataset/train/DME/DME-1169820-4.jpeg"  # replace with your image path
photo_features = extract_features(img_path)
caption = generate_caption(photo_features)
print("Generated Caption:", caption)
