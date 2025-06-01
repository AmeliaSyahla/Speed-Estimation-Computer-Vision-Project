import cv2
import json
import numpy as np
from collections import defaultdict, deque
import supervision as sv
#Config
"""
#amplaz 1
SOURCE = np.array([[776, 42], [1150, 42], [1628, 1079], [358, 1077]]) #ganti tergantung tempat
TARGET_WIDTH = 6 #ganti tergantung tempat
TARGET_HEIGHT = 55 #ganti tergantung tempat

#amplaz 2
SOURCE = np.array([[774, 61], [1199, 79], [1597, 1077], [238, 1079]]) 
TARGET_WIDTH = 6 
TARGET_HEIGHT = 53

FKH 1
SOURCE = np.array([[688,161], [1256, 172], [1912, 1074], [59, 1079]]) #ganti tergantung tempat
TARGET_WIDTH = 12 #ganti tergantung tempat
TARGET_HEIGHT = 37 #ganti tergantung tempat

FKH 2
SOURCE = np.array([[596, 191], [1317, 199], [1882, 1070], [51, 1070]]) #ganti tergantung tempat
TARGET_WIDTH = 10 #ganti tergantung tempat
TARGET_HEIGHT = 27 #ganti tergantung tempat 
"""

#KONFIGURASI
source_video_path = r"Dataset/FKH02/FKH02a_part_1.mp4" #ganti
target_video_path = r"Output_Videos/FKH02a_output_1.mp4" #ganti
json_path = r"Output_Json_Data/FKH02a_part_1_tracked_data.json" #ganti


confidence_threshold = 0.3 #tetap
iou_threshold = 0.7 #tetap

# Titik homografi
SOURCE = np.array([[596, 191], [1317, 199], [1882, 1070], [51, 1070]]) #ganti tergantung tempat
TARGET_WIDTH = 10 #ganti tergantung tempat
TARGET_HEIGHT = 27 #ganti tergantung tempat
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

#TRANSFORMASI
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


if __name__ == "__main__":
    with open(json_path) as f:
        tracking_data = json.load(f)

    frame_tracks = {frame["frame_id"]: frame["tracked_objects"] for frame in tracking_data}

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=confidence_threshold)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 2, position=sv.Position.BOTTOM_CENTER)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    with sv.VideoSink(target_video_path, video_info) as sink:
        frame_id = 0
        for frame in frame_generator:
            frame_id += 1
            tracked_objects = frame_tracks.get(frame_id, [])

            if not tracked_objects:
                sink.write_frame(frame)
                continue

            xyxy, class_id, tracker_id, confidence = [], [], [], []
            for obj in tracked_objects:
                xyxy.append(obj["box"])
                class_id.append(obj["class_id"])
                tracker_id.append(obj["track_id"])
                confidence.append(obj["score"])

            detections = sv.Detections(
                xyxy=np.array(xyxy),
                class_id=np.array(class_id),
                tracker_id=np.array(tracker_id),
                confidence=np.array(confidence),
            )

            detections = detections[detections.confidence > confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            for tid, [_, y] in zip(detections.tracker_id, points):
                coordinates[tid].append(y)

            labels = []
            for tid in detections.tracker_id:
                y_coords = coordinates[tid]
                if len(y_coords) < video_info.fps / 2:
                    labels.append(f"#{tid}")
                else:
                    dist = abs(y_coords[-1] - y_coords[0])
                    time = len(y_coords) / video_info.fps
                    speed = dist / time * 3.6
                    labels.append(f"#{tid} {int(speed)} km/h")

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
