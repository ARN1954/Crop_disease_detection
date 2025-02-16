import numpy as np
from tensorflow.keras import models, layers
import tensorflow as tf

image_folder_path = 'C:/Users/ARYAN PUND/Downloads/Telegram Desktop/data'

# Load images with labels (directory names as labels)
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_folder_path,
    image_size=(256, 256),  # Resize images to 256x256
    batch_size=32,  # Batch size
    label_mode='int'  # or 'categorical' for one-hot encoding
)

# Split the dataset
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

training_dataset = dataset.take(train_size)
remaining_dataset = dataset.skip(train_size)
validating_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# Prefetching
training_dataset = training_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validating_dataset = validating_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Data preprocessing layers
resize_rescale = tf.keras.Sequential([
    layers.Resizing(256, 256),
    layers.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# Model definition
model = models.Sequential([
    layers.Input(shape=(256, 256, 3)),
    resize_rescale,
    # data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train the model
model.fit(
    training_dataset,
    epochs=30,
    batch_size=32,
    verbose=1,
    validation_data=validating_dataset
)

# Evaluate the model on the test dataset
scores = model.evaluate(test_dataset)

# Save the model
model_version = 3
model.save(f"potato/models/{model_version}.keras")
