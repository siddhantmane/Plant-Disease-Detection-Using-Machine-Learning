import tensorflow as tf 

leaf_model = tf.keras.models.load_model(r"models/is_leaf_model.hdf5")
img_size = (224, 224)

def predict_leaf(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    prediction = leaf_model.predict(img_array)
    if prediction[0][0] >= 0.5:
        return True  # Tomato leaf
    else:
        return False  # Not a tomato leaf

def predict_leaf(file):
    image = read_file_as_image(file.read())
    image_batch = np.expand(image,0)
    prediction = leaf_model.predict(image_batch)
    if prediction[0][0] >= 0.5:
        return True  # Tomato leaf
    else:
        return False


image_path = r'D:\siddhant project\test images\test3.jpg'

result = predict_leaf(image_path)
if result:
    print("The image contains a tomato leaf.")
else:
    print("The image does not contain a tomato leaf.")