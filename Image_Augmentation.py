import os
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img

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

augment_images(train_human_dir, os.path.join(train_dir, 'augmented_human'))
augment_images(validation_human_dir, os.path.join(validation_dir, 'augmented_human'))
augment_images(test_human_dir, os.path.join(test_dir, 'augmented_human'))
augment_images(train_not_human_dir, os.path.join(train_dir, 'augmented_not_human'))
augment_images(validation_not_human_dir, os.path.join(validation_dir, 'augmented_not_human'))
augment_images(test_not_human_dir, os.path.join(test_dir, 'augmented_not_human'))