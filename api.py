from flask import Flask, request, jsonify
from cloudinary_service import CloudinaryService
from data_provider import DataProvider
from video_analyzer import VideoAnalyzer
from video_collector import VideoCollector
import cv2

# Constants
WEIGHTS_PATH = ['weights/yolo11n.pt', 'weights/cctv-age-classifier.pt', 'weights/cctv-gender-classifier.pt']

# Initialize Flask app
app = Flask(__name__)

@app.route('/video/upload', methods=['POST'])
def process_video():
    """
    Endpoint to process a video.
    Expects a JSON payload with the video path.
    """
    try:
        data = request.json
        data_provider = DataProvider(CloudinaryService(), VideoCollector())
        video_cap = data_provider.download_video(data["url"])
        data_provider.populate_video_data(video_cap, data)

        height = int(data_provider.video_collector.get_height()) - int(data_provider.video_collector.get_height() / 3)
        line_points = [(int(data_provider.video_collector.get_width()), height),
                       (int(data_provider.video_collector.get_start_x()), height)]

        video_analyzer = VideoAnalyzer(data_provider.video_collector, WEIGHTS_PATH[1], WEIGHTS_PATH[2], data_provider,
                                       show=False, model=WEIGHTS_PATH[0], colormap=cv2.COLORMAP_PARULA,
                                       region=line_points, show_in=True, show_out=True)

        # Process the video
        cap = data_provider.video_collector.get_cap()
        ret, frame = cap.read()
        heatmap_image = None
        prev_heatmap_image = None

        if ret:
            video_analyzer.initialize(frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                data_provider.provide(prev_heatmap_image)
                break

            # Analyze frame and aggregate data
            prev_heatmap_image = heatmap_image
            heatmap_image = video_analyzer.analyze_video_frame(frame)
            video_analyzer.video_writer.write(heatmap_image)

        # Release resources
        cap.release()
        video_analyzer.video_writer.release()

        return jsonify({'message': 'Video processing complete'}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
