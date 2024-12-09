
from data_provider import DataProvider
from video_analyzer import VideoAnalyzer
from video_collector import VideoCollector
import cv2

WEIGHTS_PATH = ['weights/yolo11n.pt', 'weights/cctv-age-classifier.pt', 'weights/cctv-gender-classifier.pt']
def main():

    video_path = "resources/video.mp4"
    # Initialize collectors
    video_collector = VideoCollector(video_path)
    height = int(video_collector.get_height()) - int(video_collector.get_height()/3)
    line_points = [(int(video_collector.get_width()), height),
                   (int(video_collector.get_start_x()), height)]
    data_provider = DataProvider(video_collector)  # Initialize DataProvider for database operations
    video_analyzer = VideoAnalyzer(video_collector, WEIGHTS_PATH[1], WEIGHTS_PATH[2], data_provider, show=True, model=WEIGHTS_PATH[0], colormap=cv2.COLORMAP_PARULA,
                                   region=line_points, show_in=True, show_out=True)


    cap = video_collector.get_cap()
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
    cap.release()  # Release video after processing
    video_analyzer.video_writer.release()  # Release the video writer
    print("Video processing complete")

if __name__ == '__main__':
    main()
