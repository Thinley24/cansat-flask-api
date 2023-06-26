# https://python.plainenglish.io/develop-your-machine-learning-api-for-image-object-detection-yolov8-with-python-flask-api-f393cb7e1e43

from PIL import Image, ImageDraw
from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
import nest_asyncio # allows nested event loops
import io
import os
import base64
from flask_cors import CORS

app = Flask(__name__) # creating api instance
model = YOLO('model.pt')

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}) # django localdomain:port


@app.route("/disasterdetection/", methods=["POST"]) # change the user request path accordingly
def predict():
    if not request.method == "POST":
        return
    
    if request.files.get("image"): # if request contains file named "image"
        image_file = request.files["image"]

        print("******Received Image for Processing******")
        # return {"return": "ok! API works...."} # for testing
        image_bytes = image_file.read() #read image file received as bytes
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB") # open the image and covert into RGB
        
        # perform object detection
        results = model(img) # feed it to model
        # results_json = {"boxes": results[0].boxes.xyxy.tolist(), "classes":results[0].boxes.cls.tolist()} # sending result to json obj
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        # boxes = results.pred[0].xyxy[0].tolist()
        # classes = results.names[results.pred[0].tolist()]

        if len(boxes) > 0:
            print("Received HTTP request")

            img_with_boxes = draw_boxes_on_image(img, boxes, classes)
            img_path = save_image(img_with_boxes)

            # output_image = io.BytesIO()
            # img.save(output_image, format="JPEG")
            # output_image.seek(0)

            # return {"Success": "Success! Dolomite...."}
            # results_json = {"boxes": boxes, "classes": classes} # sending result to json obj
            # return jsonify(results_json)

            with open(img_path, "rb") as f:
                image_data = f.read()

            encoded_image = base64.b64encode(image_data).decode("utf-8")
            class_labels = get_class_labels(classes)
            # return send_file(output_image, mimetype="image/jpeg")
            results_json = {"class_labels": class_labels, "image_filename": encoded_image}
            return jsonify(results_json)

        else:
            print("Failed to receive request")
            results_json = {"Failure": "The request wasn't received successfully"}
            return jsonify(results_json)
    
    return "Invalid request"

def draw_boxes_on_image(image, boxes, classes):
    draw = ImageDraw.Draw(image)
    for box, cls in zip(boxes, classes):
        x_min, y_min, x_max, y_max = [int(coord) for coord in box] # convert box coordinates to integers

        # class mapping
        cls_label = "lake" if int(cls) == 0 else "landslide"
        # cls = int(cls) # convert class prediction to integer
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
        draw.text((x_min, y_min - 10), cls_label, fill="red") # convert cls to string before drawing
    return image

def save_image(image):
    directory = "image"
    if not os.path.exists(directory):
        os.makedirs(directory)

    image_filename = "detected_image.jpg"
    image_path = os.path.join(directory, image_filename)
    image.save(image_path)
    return image_path

# function to handle label name of detected disaster
def get_class_labels(classes):
    class_labels = []
    for cls in classes:
        cls_label = "lake" if int(cls) == 0 else "landslide"
        class_labels.append(cls_label)
    return class_labels
    
# running the flask app
if __name__ == "__main__":   
    nest_asyncio.apply() # establishes nested event loops
    app.run(host="localhost", port=8000) # Starts the Flask development server on specified ip and port
