from flask import Flask, request, jsonify
import os


app = Flask(__name__)
UPLOAD_FOLDER = "./uploaded_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    Endpoint to upload a video file for analysis.

    Expects:
        - 'file': The video file in the request payload.

    Returns:
        A JSON response indicating the status of the upload.
    """
    try:
        # Check if the request has a file
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400

        video_file = request.files['file']

        # Save the uploaded video
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)

        # Trigger analysis (pseudo-function, replace with your analysis logic)
        #perform_analysis(video_path)

        return jsonify({"status": "success", "message": "Video uploaded and analysis started"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
