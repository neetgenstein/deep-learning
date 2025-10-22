import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import to_categorical
import pickle

base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
# Use global average pooling to reduce dims
from tensorflow.keras.layers import GlobalAveragePooling2D
x = GlobalAveragePooling2D()(base.output)
feature_extractor = Model(inputs=base.input, outputs=x)

# Create generator without augmentation for feature extraction
extract_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False, class_mode='categorical'
)
features_train = feature_extractor.predict(
    extract_gen, 
    steps=int(np.ceil(extract_gen.n / BATCH_SIZE))
)
labels_train = extract_gen.classes
np.save("features_train.npy", features_train)
np.save("labels_train.npy", labels_train)

# Similarly for validation
val_extract = val_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False, class_mode='categorical')
features_val = feature_extractor.predict(
    val_gen, 
    steps=int(np.ceil(val_gen.n / BATCH_SIZE))
)
labels_val = val_extract.classes

np.save("features_val.npy", features_val)
np.save("labels_val.npy", val_extract.classes)

# Train classifier on top of saved features
from tensorflow.keras.optimizers import Adam
ytr = to_categorical(labels_train, num_classes=4)
yvl = to_categorical(labels_val, num_classes=4)

clf = Sequential([
    Input(shape=features_train.shape[1:]),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
# Compile the classifier
clf.compile(optimizer=Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Updated ModelCheckpoint for Keras 3.x
callbacks = [
    ModelCheckpoint("/kaggle/working/inceptionv3_oct_feat.weights.h5",   # ✅ updated extension
                    save_best_only=True,
                    save_weights_only=True,              # ✅ important for .h5
                    monitor='val_accuracy')
]

# Train classifier
history = clf.fit(
    features_train, ytr,
    validation_data=(features_val, yvl),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# Optional: save final weights manually (ensures you keep last version)
clf.save_weights("/kaggle/working/inceptionv3_oct_feat_final.weights.h5")

# Save label mapping for inference
with open("/kaggle/working/labels_inceptionv3_oct.pkl", "wb") as f:
    pickle.dump(extract_gen.class_indices, f)

print("Training complete.")
print(" - Best weights: inceptionv3_oct_feat.weights.h5")
print(" - Final weights: inceptionv3_oct_feat_final.weights.h5")
print(" - Label map: labels_inceptionv3_oct.pkl")