import tensorflow as tf

def create_and_save_model():
    # Access Keras layers via tf.keras
    MobileNetV2 = tf.keras.applications.MobileNetV2
    Model = tf.keras.models.Model
    Dense = tf.keras.layers.Dense
    GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
    Adam = tf.keras.optimizers.Adam

    # 1. Load MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

    # 2. Freeze the base model
    base_model.trainable = False

    # 3. Add our own "Head"
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(7, activation='softmax')(x)

    # 4. Combine them
    model = Model(inputs=base_model.input, outputs=predictions)

    # 5. Compile
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # 6. Save
    print("Saving model to emotion_model.h5...")
    model.save('emotion_model.h5')
    print("Model saved successfully!")

if __name__ == "__main__":
    create_and_save_model()
