import os
import subprocess
import time
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import optimizers
from keras.callbacks import ModelCheckpoint

photo_dir = '/home/pi/tf-env/'

pred_val = False

os.makedirs(photo_dir,exist_ok=True)
model = tf.keras.models.load_model(filepath = '/home/pi/tf-env/model.h5')
i=1
while pred_val == False :
    subprocess.run(['libcamera-still','-o',f'{photo_dir}/pic_{i}.jpg'])
    time.sleep(2)
        
    # Load the model
        
        
    # Predict on a single image
    img_path = f'{photo_dir}/pic_{i}.jpg'
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)
        
    prediction = model.predict(img_array)
    print(prediction)
        
        
        
    if prediction[0][0] > 0.5:
        print('Not Human Body')
        pred_val = False
    else:
        print('Human Body')
        pred_val = True
        i=i+1
            
print("Loop exited")