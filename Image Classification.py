import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import models
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from PIL import Image

# Set image dimensions
image_height, image_width = 256, 256

# Directory paths
train_directory = 'Datasets/train'
test_directory = 'Datasets/test'
validation_directory = 'Datasets/validation'


def create_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Assuming there are 4 classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    model = create_model()
    print(model.summary())

    # Configure image data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Set up the data generators
    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(image_height, image_width),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_directory,
        target_size=(image_height, image_width),
        batch_size=32,
        class_mode='categorical'
    )

    # Fit the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=1,  # Set a realistic number of epochs for sufficient training
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    # Save the trained model
    model_path = 'Models/image_classification_model.h5'
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Evaluate the model
    model.evaluate(validation_generator)

    # Ensure the model is loaded before predicting
    model = load_model(model_path)

    # Dictionary to convert class indices to labels
    class_labels = {0: 'apple', 1: 'banana', 2: 'mixed', 3: 'orange'}

    from PIL import Image, ImageOps

    def load_and_preprocess_image(img_path, target_size=(256, 256)):
        try:
            img = Image.open(img_path)  # Try to open the image file
            if img.mode != 'RGB':
                img = img.convert('RGB')  # Convert non-RGB images to RGB
            # Resize the image using LANCZOS resampling
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0  # Normalize the image array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except IOError as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def predict_image_class(model, img_path):
        """Predict the class of an image using the loaded model."""
        img_array = load_and_preprocess_image(img_path)
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        return class_labels[predicted_class_idx]

    # List of images in the test directory
    test_images = [f for f in os.listdir(test_directory) if f.endswith('.jpg')]

    # Dataframe to store results
    results = []

    # Process each image in the test directory
    for image_name in test_images:
        img_path = os.path.join(test_directory, image_name)
        predicted_class = predict_image_class(model, img_path)
        actual_class = image_name.split('_')[0]

        # Append results
        results.append({
            'Image': image_name,
            'Actual Class': actual_class,
            'Predicted Class': predicted_class
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_file_path = 'Results/result.csv'
    results_df.to_csv(csv_file_path, index=False)

    print(f'Results saved to {csv_file_path}')


if __name__ == '__main__':
    main()
