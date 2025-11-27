import tensorflow as tf
import os
import numpy as np

# Access Keras components via tf.keras
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
MobileNetV2 = tf.keras.applications.MobileNetV2
Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Input = tf.keras.layers.Input
Model = tf.keras.models.Model
Adam = tf.keras.optimizers.Adam

# Configuration
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 5 # Reduced for quicker turnaround, increase for better accuracy
# Robust path finding
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'fer2013')

def train_model():
    # 1. Data Generators (Augmentation for training)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Load Data (Assumes folder structure: dataset/train/angry, dataset/train/happy, etc.)
    # Since FER2013 usually comes as a CSV or specific folder structure, 
    # this part might need adjustment based on how the user downloads the data.
    # For this script, we assume a standard 'train' and 'validation' folder structure.
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Please download FER2013 and extract it.")
        return

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH + '/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    print("Loading Validation Data...")
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH + '/test', 
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
        # subset='validation' # Not needed as we use a separate folder
    )

    # 2. Model Architecture (Same as model_saving.py)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze base model initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # 3. Train
    print("Starting Training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # 4. Save
    model.save('emotion_model_trained.h5')
    print("Trained model saved as 'emotion_model_trained.h5'")

if __name__ == "__main__":
    train_model()
