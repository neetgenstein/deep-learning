from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from sklearn.metrics import classification_report

# 1. Recreate the exact same architecture
base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
preds = Dense(4, activation='softmax')(x)  # same number of classes as training
model = Model(inputs=base.input, outputs=preds)

# 2. Load your trained weights
model.load_weights("/kaggle/input/transfer-learning-weights/tensorflow2/default/1/inceptionv3_oct_transfer.weights.h5")
print("Weights loaded successfully!")

# 3. Compile model (needed for evaluation)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Preprocess a single image for prediction
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))  # resize to InceptionV3 input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    img_array = preprocess_input(img_array)
    return img_array

# Example: predict on one image
img_path = "/kaggle/input/oct2017/dataset/train/DRUSEN/DRUSEN-1001666-8.jpeg"
img_array = preprocess_image(img_path)
pred = model.predict(img_array)
class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
predicted_class = class_labels[np.argmax(pred)]
print(f"Predicted class: {predicted_class}")

# 5. Evaluate on validation dataset
val_dir = "/kaggle/input/oct2017/dataset/val"  # path to validation folder

datagen = ImageDataGenerator(preprocessing_function=None)  # no augmentation, only normalization
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