# https://python.plainenglish.io/develop-your-machine-learning-api-for-image-object-detection-yolov8-with-python-flask-api-f393cb7e1e43

from PIL import Image, ImageDraw
from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
import nest_asyncio # allows nested event loops
import io

app = Flask(__name__) # creating api instance
model = YOLO('model.pt')

@app.route("/disasterdetection/", methods=["POST"]) # change the user request path accordingly
def predict():
    if not request.method == "POST":
        return
    
    if request.files.get("image"): # if request contains file named "image"
        image_file = request.files["image"]
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
            # detected obj found, draw bounding box and label
            # draw = ImageDraw.Draw(img)
            # for box, cls in zip(boxes, classes):
            #     x_min, y_min, x_max, y_max = box
            #     draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
            #     draw.text((x_min, y_min - 10), cls, fill="red")

            # # save the modified image to a BytesIO obj
            # output_image = io.BytesIO()
            # img.save(output_image, format="JPEG")
            # output_image.seek(0)

            # return the modified image as the response
            # return send_file(output_image, mimetype="image/jpeg")
            print("received HTTP request")
            # return {"Success": "Success! Dolomite...."}
            results_json = {"boxes": boxes, "classes": classes} # sending result to json obj
            return jsonify(results_json)

        else:
            print("failed to receive request")
            results_json = {"Failure": "Sonam Hubby! No image detected"}
            return jsonify(results_json)
    
    return "Invalid request"

    
# running the flask app
if __name__ == "__main__":   
    nest_asyncio.apply() # establishes nested event loops
    app.run(host="localhost", port=8000) # Starts the Flask development server on specified ip and port


# # testing model with image from the directory
# results = model.predict(source='static/landslide_test4.jpg')
# print("Bounding Boxes :", results[0].boxes.xyxy)
# print("Classes :", results[0].boxes.cls)