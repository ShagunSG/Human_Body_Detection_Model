#Write a code for ResNet50 model.
# The model should take images from the folder 'Screenshots' and a csv file named 'Image_classification' having two columns, namely, 'New Name' and 'Image Classification' as input. 
# The model should output a single prediction for each image.
# The model should be trained on the given dataset.
# Output the accuracy of the model as well.
# Split the dataset into training and testing datasets, while also splitting the images classified as 'Human Body' and 'Not Human Body' for better accuracy.
# Use the testing dataset to evaluate the model.
# Use the training dataset to train the model.
# Use the given csv file to classify the images.
# Use the ResNet50 model to classify the images.

import os
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import optimizers
from keras.callbacks import ModelCheckpoint


base_dir = 'D:\IIT Indore Files\Assignments\Sem-6\Human_Body_detection\Screenshots'
# names = ["train", 'validation', 'test']
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
# for i in range (3):
#     if not os.path.exists(os.path.join(base_dir, names[i])):
#         os.makedirs(os.path.join(base_dir, names[i]))
#     if not os.path.exists(os.path.join(base_dir, names[i], 'Human Body')):
#         os.makedirs(os.path.join(base_dir, names[i], 'Human Body'))
#     if not os.path.exists(os.path.join(base_dir, names[i], 'Not Human Body')):
#         os.makedirs(os.path.join(base_dir, names[i], 'Not Human Body'))

# Directory with our training Human Body pictures
train_human_dir = os.path.join(train_dir, 'Human Body')

# Directory with our Not Human Body pictures
train_not_human_dir = os.path.join(train_dir, 'Not Human Body')

# Directory with our validation Human Body pictures
validation_human_dir = os.path.join(validation_dir, 'Human Body')

# Directory with our validation Not Human Body pictures
validation_not_human_dir = os.path.join(validation_dir, 'Not Human Body')

# Directory with our testing Human Body pictures
test_human_dir = os.path.join(test_dir, 'Human Body')

# Directory with our testing Not Human Body pictures
test_not_human_dir = os.path.join(test_dir, 'Not Human Body')

# df = pd.read_csv('Image_classification.csv')
# df['New Name'] = df['New Name'].apply(lambda x: x)
# df['Image Classification'] = df['Image Classification'].apply(lambda x: 'Human Body' if x == 'Human Body' else 'Not Human Body')

# # Iterate over the 'Image Classification' column and count the number of entries for each category
# human_body_count = df[df['Image Classification'] == 'Human Body'].shape[0]
# not_human_body_count = df[df['Image Classification'] == 'Not Human Body'].shape[0]

# # Store the names of the separate categories in separate lists
# human_body_images = df[df['Image Classification'] == 'Human Body']['New Name'].tolist()
# not_human_body_images = df[df['Image Classification'] == 'Not Human Body']['New Name'].tolist()

# # Split the dataset into training, validation, and testing datasets based on the category
# train_human_body_images = human_body_images[int(0.00 * human_body_count):int(0.7 * human_body_count)]
# validation_human_body_images = human_body_images[int(0.7 * human_body_count):int(0.85 * human_body_count)]
# test_human_body_images = human_body_images[int(0.85 * human_body_count):int(1.00 * human_body_count)]

# train_not_human_body_images = not_human_body_images[int(0.00 * human_body_count):int(0.7 * not_human_body_count)]
# validation_not_human_body_images = not_human_body_images[int(0.7 * not_human_body_count):int(0.85 * not_human_body_count)]
# test_not_human_body_images = not_human_body_images[int(0.85 * not_human_body_count):int(1.00 * human_body_count)]

# # Copy the images to the respective directories
# for image_name in train_human_body_images:
#     shutil.copy(os.path.join(base_dir, image_name), os.path.join(train_human_dir, image_name))

# for image_name in validation_human_body_images:
#     shutil.copy(os.path.join(base_dir, image_name), os.path.join(validation_human_dir, image_name))

# for image_name in test_human_body_images:
#     shutil.copy(os.path.join(base_dir, image_name), os.path.join(test_human_dir, image_name))

# for image_name in train_not_human_body_images:
#     shutil.copy(os.path.join(base_dir, image_name), os.path.join(train_not_human_dir, image_name))

# for image_name in validation_not_human_body_images:
#     shutil.copy(os.path.join(base_dir, image_name), os.path.join(validation_not_human_dir, image_name))

# for image_name in test_not_human_body_images:
#     shutil.copy(os.path.join(base_dir, image_name), os.path.join(test_not_human_dir, image_name))

# Load the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Train the model
# checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)
history = model.fit(train_generator, epochs=10, validation_data=validation_generator, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print('Test accuracy:', accuracy)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predict on test data
predictions = model.predict(test_generator)
print(predictions)

# Save the model
model.save('model.h5')

# Load the model
model = tf.keras.models.load_model('model.h5')

# Predict on a single image
img_path = 'path/to/image.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array)
print(prediction)