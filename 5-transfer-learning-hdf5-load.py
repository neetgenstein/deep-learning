import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import classification_report

# 1. Load the full trained model (.hdf5)
model_path = "/kaggle/input/initial-transfer-full/tensorflow2/default/1/inceptionv3_oct_transfer_initial.hdf5"
model = load_model(model_path, compile=False)  # compile=False to avoid legacy arg errors
print("Full InceptionV3 transfer model (.hdf5) loaded successfully!")

# 2. Compile (optional but needed for evaluation/prediction consistency)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Preprocess a single image for prediction
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))  # resize to InceptionV3 input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Example: predict on one image
img_path = "/kaggle/input/oct2017/dataset/train/DRUSEN/DRUSEN-1001666-8.jpeg"
img_array = preprocess_image(img_path)
pred = model.predict(img_array)
class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
predicted_class = class_labels[np.argmax(pred)]
print(f"Predicted class: {predicted_class}")

# 4. Evaluate on validation dataset
val_dir = "/kaggle/input/oct2017/dataset/val"

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # use InceptionV3 preprocessing
val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get predictions
print("\nRunning predictions on validation set...")
val_preds = model.predict(val_gen, verbose=1)

# Convert predictions and true labels
y_pred = np.argmax(val_preds, axis=1)
y_true = val_gen.classes

# Class label mapping
idx_to_class = {v: k for k, v in val_gen.class_indices.items()}

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(idx_to_class.values())))