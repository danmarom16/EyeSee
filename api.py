from flask import Flask, request, jsonify
from data_provider import DataProvider
from data_aggregator import DataAggregator
from video_analyzer import VideoAnalyzer
from video_collector import VideoCollector
import cv2


# Initialize Flask app and database components
app = Flask(__name__)
data_provider = DataProvider()  # Initialize DataProvider for database operations
data_aggregator = DataAggregator(data_provider=data_provider, save_interval=100)  # Save every 100 frames


@app.route('/upload', methods=['POST'])
def upload_data():

    video_path = request.form.get('video_path')
    #video_path = "resources/video.mp4"
    # Initialize collectors
    video_collector = VideoCollector(video_path)

    # Analyze video frame by frame
    video_capture = video_collector.collect_video()
    video_analyzer = VideoAnalyzer()

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        # Analyze frame and aggregate data
        frame_data = video_analyzer.analyze_video_frame(frame)
        data_aggregator.add_frame_data(frame_data)
    video_capture.release()  # Release video after processing

    # Generate final summary and return it
    summary = data_aggregator.get_summary()
    return jsonify({"status": "success", "summary": summary})


@app.route('/data', methods=['GET'])
def get_data():
    # Return the current aggregated summary data
    summary = data_aggregator.get_summary()
    return jsonify(summary)


if __name__ == '__main__':
    app.run(debug=True)
