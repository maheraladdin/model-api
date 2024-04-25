from flask import Flask, request, jsonify
from use_model import ImageCaptioningModel

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def example():
    # Assuming the data in the request body is JSON
    data = request.json

    # Accessing specific keys/values from the JSON data
    key1 = data.get('key1')
    key2 = data.get('key2')

    # Create a dictionary with the response data
    response_data = {
        "response": f"{key1} {key2}!",
    }

    # Return a JSON response
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
