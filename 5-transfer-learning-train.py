# pip install tensorflow==2.12.0


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

IMG_SIZE = (299, 299)
BATCH_SIZE = 32

train_dir = "/kaggle/input/oct2017/dataset/train"
val_dir   = "/kaggle/input/oct2017/dataset/val"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
preds = Dense(4, activation='softmax')(x)
model = Model(inputs=base.input, outputs=preds)

# Stage A: freeze base
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint("/kaggle/working/inceptionv3_oct_transfer_initial.hdf5", 
                    save_best_only=True, 
                    monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]


history = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)


callbacks = [
    ModelCheckpoint("/kaggle/working/inceptionv3_oct_transfer_final.hdf5", 
                    save_best_only=True, 
                    monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

# Stage B: unfreeze last N layers and fine-tune
for layer in base.layers[-100:]:  # example: unfreeze last 100 layers
    layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)
