import os

# Flask utils
from flask import Flask, request, jsonify

IMG_SIZE = (640, 640)

# Define a flask app
app = Flask(__name__)


@app.route('/predict', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the data from post request
        data = request.form.get("data")
        if data != None:
            print("Received data from client: ", data)
            response = {
                        'status': 200,
                        'message': "Hello, we received your message!",
                        }
            return jsonify(response)
    
if __name__ == '__main__':
    port = os.environ.get('PORT', 8008)
    app.run(debug=False, host='0.0.0.0', port=port)