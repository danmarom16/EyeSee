
from data_provider import DataProvider
from video_analyzer import VideoAnalyzer
from video_collector import VideoCollector
import cv2

#TODO:fix lost_id and debug last 2 function in anayzer
WEIGHTS_PATH = ['weights/yolo11n.pt', 'weights/cctv-age-classifier.pt', 'weights/cctv-gender-classifier.pt']
def main():

    video_path = "resources/video.mp4"
    # Initialize collectors
    video_collector = VideoCollector(video_path)
    line_points = [(int(video_collector.get_width()), int(video_collector.get_height()/2)), (int(video_collector.get_width() - video_collector.get_width() / 3),
                              video_collector.get_height())]
    data_provider = DataProvider(video_collector)  # Initialize DataProvider for database operations
    video_analyzer = VideoAnalyzer(video_collector, WEIGHTS_PATH[1], WEIGHTS_PATH[2], data_provider, show=True, model=WEIGHTS_PATH[0], colormap=cv2.COLORMAP_PARULA,
                                   region=line_points, show_in=True, show_out=True)


    cap = video_collector.get_cap()
    ret, first_frame = cap.read()
    #if ret:
        #video_analyzer.initialize(first_frame)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            video_analyzer.analyze_final_frame(frame)
            #data_provider.save_final_data()
            break
        # Analyze frame and aggregate data
        heatmap_image = video_analyzer.analyze_video_frame(frame)
        video_analyzer.video_writer.write(heatmap_image)
    cap.release()  # Release video after processing
    video_analyzer.video_writer.release()  # Release the video writer
    print("Video processing complete")

if __name__ == '__main__':
    main()
