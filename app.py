from flask import Flask, request, jsonify
from object_detection2 import ObjectDetection
from image_captioning import ImageCaptioning

app = Flask(__name__)

object_detector = ObjectDetection()
image_captioner = ImageCaptioning()

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_path = data['url']

    objects = object_detector.detect_objects(image_path)
    description = image_captioner.generate_caption(image_path)

    response = {
        "objects": objects,
        "description": description
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
