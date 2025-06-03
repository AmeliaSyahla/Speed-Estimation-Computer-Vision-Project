import cv2
import os
import json
import numpy as np
from collections import defaultdict, deque
import supervision as sv

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
    config_path = "model_config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{config_path}'. Check file format.")
        exit()

    # Ensure the output video folder exists
    output_video_base_folder = "Output_Videos"
    if not os.path.exists(output_video_base_folder):
        os.makedirs(output_video_base_folder, exist_ok=True)
        print(f"[INFO] Created output video folder: {output_video_base_folder}")

    for video_config in config["videos"]:
        source_video_path = video_config["source_video_path"]
        target_video_path = video_config["target_video_path"]
        json_path = video_config["json_path"]
        SOURCE = np.array(video_config["homography"]["SOURCE"])
        TARGET_WIDTH = video_config["homography"]["TARGET_WIDTH"]
        TARGET_HEIGHT = video_config["homography"]["TARGET_HEIGHT"]
        confidence_threshold = video_config["processing_params"]["confidence_threshold"]
        iou_threshold = video_config["processing_params"]["iou_threshold"]

        TARGET = np.array(
            [
                [0, 0],
                [TARGET_WIDTH - 1, 0],
                [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
                [0, TARGET_HEIGHT - 1],
            ]
        )

        print(f"\n[INFO] Processing video: {video_config['name']}")
        print(f"  Source Video: {source_video_path}")
        print(f"  Output Video: {target_video_path}")
        print(f"  JSON Data: {json_path}")
        print(f"  Homography Source: {SOURCE.tolist()}")
        print(f"  Target Dimensions: {TARGET_WIDTH}x{TARGET_HEIGHT}")
        print(f"  Confidence Threshold: {confidence_threshold}")
        print(f"  IOU Threshold: {iou_threshold}")

        try:
            with open(json_path) as f:
                tracking_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON tracking data file '{json_path}' not found. Skipping video.")
            continue
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{json_path}'. Check file format. Skipping video.")
            continue

        try:
            frame_tracks = {frame["frame_id"]: frame["tracked_objects"] for frame in tracking_data}

            video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
            frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

            # Initialize ByteTrack and remapping for each video
            byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=confidence_threshold)
            coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
            
            # For remapping ByteTrack IDs to start from 1 for each video
            current_id_map = {}
            next_remapped_id = 1

            thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

            box_annotator = sv.BoxAnnotator(thickness=thickness)
            label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)
            trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 2, position=sv.Position.BOTTOM_CENTER)

            polygon_zone = sv.PolygonZone(polygon=SOURCE)
            view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

            with sv.VideoSink(target_video_path, video_info) as sink:
                frame_id = 0
                for frame in frame_generator:
                    frame_id += 1
                    tracked_objects = frame_tracks.get(frame_id, [])

                    if not tracked_objects:
                        sink.write_frame(frame)
                        continue

                    xyxy, class_id, confidence = [], [], []
                    for obj in tracked_objects:
                        xyxy.append(obj["box"])
                        class_id.append(obj["class_id"])
                        confidence.append(obj["score"])

                    detections = sv.Detections(
                        xyxy=np.array(xyxy),
                        class_id=np.array(class_id),
                        confidence=np.array(confidence),
                    )

                    detections = detections[detections.confidence > confidence_threshold]
                    detections = detections[polygon_zone.trigger(detections)]
                    detections = detections.with_nms(threshold=iou_threshold)
                    detections = byte_track.update_with_detections(detections=detections)

                    # Remap ByteTrack's assigned IDs
                    remapped_tracker_ids = []
                    for original_tid in detections.tracker_id:
                        if original_tid not in current_id_map:
                            current_id_map[original_tid] = next_remapped_id
                            next_remapped_id += 1
                        remapped_tracker_ids.append(current_id_map[original_tid])
                    detections.tracker_id = np.array(remapped_tracker_ids)

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
            print(f"[INFO] Successfully processed and saved: {target_video_path}")
        except FileNotFoundError:
            print(f"Error: Source video file '{source_video_path}' not found. Skipping video.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing {video_config['name']}: {e}. Skipping video.")
            import traceback
            traceback.print_exc()
            continue
