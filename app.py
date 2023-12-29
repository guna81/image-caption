from flask import Flask, request, jsonify
import json
from image_caption import image_caption

app = Flask(__name__)

@app.route("/", methods=["POST"])
def caption_api():
    try:
        # Get user message from the request body
        image = request.files["image"]
        print('image', image)
        multi_response = True # request.json["multi_response"]
        # Check for predefined responses
        responses = image_caption(image)
        
        result = responses if multi_response else responses[0]
        # Generate JSON response for the platform (replace with your platform's specific format)
        # data = json.dumps({"message": responses})
        return jsonify(result), 200
    except Exception as e:
        print("error", e)
        return jsonify({
            "error": True
        })


if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True,
    )