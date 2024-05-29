from flask import Flask ,render_template ,request , redirect,url_for
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import base64
import io


MODEL = tf.keras.models.load_model(r"./models/model.hdf5")
leaf_model = tf.keras.models.load_model(r"./models/is_leaf_model.hdf5")

app = Flask(__name__,template_folder="templates",static_folder="static")


CLASS_NAMES = ["Tomato_Bacterial_spot",
 "Tomato_Early_blight",
 "Tomato_Late_blight",
 "Tomato_Leaf_Mold",
 "Tomato_Septoria_leaf_spot",
 "Tomato_Spider_mites_Two_spotted_spider_mite",
 "Tomato__Target_Spot",
 "Tomato__Tomato_YellowLeaf__Curl_Virus",
 "Tomato__Tomato_mosaic_virus",
 "Tomato_healthy"]



def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(data))
    image = tf.image.resize(image, [256, 256])
    return image



def predict(file):

    image = read_file_as_image(file)
    leaf_image_batch = np.expand_dims(image,0) / 255.0
    leaf_prediction = leaf_model.predict(leaf_image_batch)
    print(leaf_prediction)
    if leaf_prediction[0][0] >= 0.5:
        img_batch = np.expand_dims(image, 0)
    
        predictions = MODEL.predict(img_batch)
        print("prediction",predictions)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        print("pridction arg max",predicted_class)
        confidence = np.max(predictions[0])
        all_classes = []
        print(all_classes)
        return {
                "is_leaf":"yes",
                "class": predicted_class ,
                "confidence": float(confidence),
                "predictions": predictions,
                "classes": CLASS_NAMES
                }
    else:
        return{
            "is_leaf":"no",
            "class":"None",
            "confidence":"None"
        }

       
    
@app.route("/",methods=["GET","POST"])
def predict_page():
    if request.method == "POST":
        file = request.files["image"]
        file.save("./static/img_uploades/"+file.filename)
        file_location = f"./static/img_uploades/{file.filename}"
        render_location = f"img_uploades/{file.filename}"
        result = predict(file_location)
        is_leaf = result["is_leaf"]
        Result_Class = result["class"]
        if Result_Class == "Tomato_Bacterial_spot" and is_leaf =="yes":
            return render_template("Tomato-Bacteria Spot.html",prediction=result,img_data=render_location)
        elif Result_Class == "Tomato_Early_blight" and is_leaf =="yes":
            return render_template("Tomato-Early_Blight.html",prediction=result, img_data=render_location)
        elif Result_Class == "Tomato_Late_blight" and is_leaf =="yes":
            return render_template("Tomato - Late_blight.html",prediction=result, img_data=render_location)
        elif Result_Class == "Tomato_Leaf_Mold" and is_leaf =="yes":
            return render_template("Tomato - Leaf_Mold.html",prediction=result, img_data=render_location)
        elif Result_Class == "Tomato_Septoria_leaf_spot" and is_leaf =="yes":
            return render_template("Tomato - Septoria_leaf_spot.html",prediction=result, img_data=render_location)
        elif Result_Class == "Tomato_Spider_mites_Two_spotted_spider_mite" and is_leaf =="yes":
            return render_template("Tomato - Two-spotted_spider_mite.html",prediction=result, img_data=render_location)
        elif Result_Class == "Tomato__Target_Spot" and is_leaf =="yes":
            return render_template("Tomato - Target_Spot.html",prediction=result, img_data=render_location)
        elif Result_Class == "Tomato__Tomato_YellowLeaf__Curl_Virus" and is_leaf =="yes":
            return render_template("Tomato - Tomato_Yellow_Leaf_Curl_Virus.html",prediction=result, img_data=render_location)
        elif Result_Class == "Tomato__Tomato_mosaic_virus" and is_leaf =="yes":
            return render_template("Tomato - Tomato_mosaic_virus.html",prediction=result, img_data=render_location)
        elif Result_Class == "Tomato_healthy" and is_leaf =="yes":
            return render_template("Tomato-Healthy.html",prediction=result, img_data=render_location)            
        else:
            msg = "Image is  Not leaf try upload right leaf image"
            return render_template("index.html",msg=msg)
        
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=8080)



















