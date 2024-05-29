import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

# Define paths to your dataset
train_dir = r'D:/siddhant project/is_Leaf/trainimage/'
test_dir = r'D:/siddhant project/is_Leaf/testimage/'

# Image size for the model
img_size = (256, 256)
batch_size = 32

# Create data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape= img_size + (3,),
    include_top=False,
    weights='imagenet'
)

# Freeze the pre-trained layers
base_model.trainable = False

# Add custom classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

model.save("is_leaf_model.hdf5")

# Example usage for inference
'''def predict_leaf(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    prediction = model.predict(img_array)
    if prediction[0][0] >= 0.5:
        return True  # Tomato leaf
    else:
        return False  # Not a tomato leaf'''

# Example usage
#import pickle
#pickle.dump(model , open('model.pkl', 'wb'))

#import os
#model_version=max([int(i) for i in os.listdir("./saved_models") + [0]])+1
#model.save(f"./saved_models/{model_version}")

#filename = 'my_model.sav'
#joblib.dump(model, open(filename, 'wb')) 



'''image_path = r'D:\siddhant project\test images\1.jpg'

result = predict_leaf(image_path)
if result:
    print("The image contains a tomato leaf.")
else:
    print("The image does not contain a tomato leaf.")'''
