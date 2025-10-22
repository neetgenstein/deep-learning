import os
import numpy as np
import pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.metrics import classification_report

base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(base.output)
feature_extractor = Model(inputs=base.input, outputs=x)

clf = Sequential([
    Input(shape=(2048,)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

clf.load_weights("/kaggle/input/feature-extraction-classification-weights/tensorflow2/default/1/inceptionv3_oct_feat_final.weights.h5")
print("Classifier weights loaded successfully!")

# Label mapping
label_map = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}
idx_to_class = {v: k for k, v in label_map.items()}
print("Loaded label mapping:", idx_to_class)

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

img_path = "/kaggle/input/oct2017/dataset/val/DME/DME-119840-10.jpeg"
img_array = preprocess_image(img_path)

features = feature_extractor.predict(img_array)
pred = clf.predict(features)
predicted_class = idx_to_class[np.argmax(pred)]
confidence = np.max(pred)
print(f"Predicted class: {predicted_class} ({confidence:.2f} confidence)")

val_dir = "/kaggle/input/oct2017/dataset/val"

# Use ImageDataGenerator for efficient loading
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Step A: Extract features for all validation images
print("\nExtracting features from validation set...")
val_features = feature_extractor.predict(val_gen, verbose=1)

# Step B: Predict using the classifier
print("Predicting classes from extracted features...")
val_preds = clf.predict(val_features, verbose=1)

# Step C: Convert predictions and true labels
y_pred = np.argmax(val_preds, axis=1)
y_true = val_gen.classes

# Map indices to class names
class_names = list(val_gen.class_indices.keys())

# Step D: Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
