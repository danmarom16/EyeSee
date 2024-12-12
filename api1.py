from flask import Flask, request, jsonify
from computer_vision_service import ComputerVisionService
from util1 import ResponseStatus

app = Flask(__name__)

@app.route('/video/upload', methods=['POST'])
def process_video():
    try:
        data = request.json()
        cv_service = ComputerVisionService(data)
        cv_service.start()
        return jsonify({'message': 'Video processing complete'}), ResponseStatus.OK
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), ResponseStatus.ERROR