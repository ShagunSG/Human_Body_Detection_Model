# Write a code to take a few images from the training, validation and test sets and augment it to create more images.
# The images are in the directory 'Screenshots', in the subdirectories 'train', 'test', and 'validation' and further in those are the subdirectories named 'Human Body' and 'Not Human Body' each.
# The images should be saved in a new directory.
# The images should be augmented in the following ways:
# 1. Rotation
# 2. Translation
# 3. Shearing
# 4. Zoom
# 5. Horizontal flip
# 6. Vertical flip
# 7. Brightness
# 8. Contrast
# 9. Saturation
# 10. Hue
# 11. Random noise
# 12. Random blur
# 13. Random sharpening
# 14. Random pixelation
# 15. Random erasing
# 16. Random cropping
# 17. Random padding
# 18. Random scaling
# 19. Random aspect ratio
# 20. Random skewing
# 21. Random perspective
# 22. Random rotation
# 23. Random translation
# 24. Random shearing
# 25. Random zoom
# 26. Random horizontal flip
# 27. Random vertical flip
# 28. Random brightness
# 29. Random contrast
# 30. Random saturation
# 31. Random hue

import os
import csv
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
from keras import layers
from keras import models
from keras import Model
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
from PIL import Image
from keras.preprocessing import image
import random
import math
import scipy.ndimage
from scipy.ndimage import zoom, rotate, shift, gaussian_filter, sobel, laplace, prewitt, uniform_filter, maximum_filter, minimum_filter, median_filter, variance, gaussian_laplace, gaussian_gradient_magnitude, generic_gradient_magnitude, generic_filter, binary_erosion, binary_dilation, binary_opening, binary_closing, grey_erosion, grey_dilation, grey_opening, grey_closing, distance_transform_edt, distance_transform_cdt, label, find_objects, center_of_mass, zoom
from scipy.ndimage import zoom

base_dir = 'D:\IIT Indore Files\Assignments\Sem-6\Human_Body_Detection_Model\Screenshots'
# names = ["train", 'validation', 'test']
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

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

# Augment the images
def augment_images(image_dir, new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for image in os.listdir(image_dir):
        img = load_img(os.path.join(image_dir, image))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=new_dir, save_prefix='augmented', save_format='png'):
            i += 1
            if i > 20:
                break

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.2, 1.0],
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# augment_images(train_human_dir, os.path.join(train_dir, 'augmented_human'))
# augment_images(validation_human_dir, os.path.join(validation_dir, 'augmented_human'))
# augment_images(test_human_dir, os.path.join(test_dir, 'augmented_human'))
# augment_images(train_not_human_dir, os.path.join(train_dir, 'augmented_not_human'))
augment_images(validation_not_human_dir, os.path.join(validation_dir, 'augmented_not_human'))
# augment_images(test_not_human_dir, os.path.join(test_dir, 'augmented_not_human'))