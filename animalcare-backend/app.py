from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load your YOLOv8 model
model = YOLO('path/to/your/yolov8/model.pt')  # Replace with the path to your YOLOv8 model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))

        # Run YOLOv8 inference
        results = model(image)

        # Prepare the results
        output = []
        for result in results:
            output.append({
                'label': result.names[int(result.cls)],
                'confidence': float(result.confidence),
                'bbox': result.xyxy.tolist()
            })

        return jsonify({'results': output})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
