import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    
    """
    Colors included within the signs:
        - Red
        - White
        - Black
        - Yellow
        - Blue
    """
    
    # 0. Resize
    # 1. Blur
    # 2. Erosion
    # 3. Dilation
    # 4. Edge detection

    images = list()
    labels = list()

    count = 0
    for obj in os.scandir(data_dir):
        
        for file in os.scandir(obj.path):
            
            image = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
            
            resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            
            blurred = cv2.GaussianBlur(resized, (1, 1), 0)
            
            kernel = np.ones((1, 1), np.uint8)
            eroded = cv2.erode(blurred, kernel, iterations = 1)
            dilated = cv2.dilate(eroded, kernel, iterations = 1)
        
            images.append(cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR))
            labels.append(count)

        count += 1

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    """ As a human looking at each of these stop signs, here are the things in order of which I consider:
            1. The shape of the sign 
            2. The symbol within it
            3. The words within it
            4. The color of the sign
            
        We'll first need to process the sign in such a way that these characteristics can be analyzed and identified properly.
        
        First, we can prune and clean up any unnecessary background or objects that hinder vision of the sign in any way:
            1. Background removal
                a. Apply edge detection
                b. Removal of noise with morphology
                c. Invert into mask
                d. Apply mask to original input
        
        Second, we can contrast the image to ensure the AI does not mistake any colors at all.
            2. Normalization
        
        1. A characteristic that could be pretty easily mistaken for another shape (especially to a neural network),
            something like sigmoid could work well. First, we'll get the edge or main outline of the sign.
    """
    
    model = keras.models.Sequential([
        
        # Simple convolution layer
        keras.layers.Conv2D(
            32, (3, 3), activation = keras.activations.relu, input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Simple pooling layer
        keras.layers.AveragePooling2D(pool_size = (2, 2)),
        
        # Complex and thorough layer
        keras.layers.Conv2D(
            128, (3, 3), activation = keras.activations.mish
        ),
        
        keras.layers.MaxPooling2D(pool_size = (2, 2)),

        keras.layers.Flatten(),
        
        keras.layers.Dense(128, activation = keras.activations.leaky_relu),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(100, activation = keras.activations.relu),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(80, activation = keras.activations.relu),
        keras.layers.Dropout(0.1),

        keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
