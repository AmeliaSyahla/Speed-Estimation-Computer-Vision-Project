import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from PIL import Image
import cv2
import numpy as np
import os
import json
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def get_custom_model_architecture(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# SORT Algorithm for object tracking 
def iou_batch(bb_test, bb_gt):
    bb_gt = np.asarray(bb_gt)
    bb_test = np.asarray(bb_test)

    if bb_gt.ndim == 1:
        bb_gt = bb_gt[np.newaxis, :]
    if bb_test.ndim == 1:
        bb_test = bb_test[np.newaxis, :]

    bb_test_expanded = bb_test[:, np.newaxis, :]
    bb_gt_expanded = bb_gt[np.newaxis, :, :]

    xx1 = np.maximum(bb_test_expanded[..., 0], bb_gt_expanded[..., 0])
    yy1 = np.maximum(bb_test_expanded[..., 1], bb_gt_expanded[..., 1])
    xx2 = np.minimum(bb_test_expanded[..., 2], bb_gt_expanded[..., 2])
    yy2 = np.minimum(bb_test_expanded[..., 3], bb_gt_expanded[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersect_area = w * h

    area_test = (bb_test_expanded[..., 2] - bb_test_expanded[..., 0]) * \
                (bb_test_expanded[..., 3] - bb_test_expanded[..., 1])
    area_gt = (bb_gt_expanded[..., 2] - bb_gt_expanded[..., 0]) * \
              (bb_gt_expanded[..., 3] - bb_gt_expanded[..., 1])

    union_area = area_test + area_gt - intersect_area
    iou = intersect_area / np.maximum(union_area, 1e-7)
    return iou


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h) if h != 0 else 0
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    area = x[2]
    aspect_ratio = x[3]

    if area < 0: area = 0
    if aspect_ratio < 0: aspect_ratio = 0

    w = np.sqrt(area * aspect_ratio) if area * aspect_ratio >= 0 else 0
    h = area / w if w != 0 and area >= 0 else 0

    return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))


class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox_xyxy, class_id, label_name, score):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox_xyxy)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = class_id
        self.label_name = label_name
        self.score = score

    def update(self, bbox_xyxy, class_id, label_name, score):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox_xyxy))
        self.class_id = class_id
        self.label_name = label_name
        self.score = score

    def predict(self):
        if (self.kf.x[2] + self.kf.x[6]) <= 0:
            self.kf.x[6] *= 0.0
        if self.kf.x[2] < 0: self.kf.x[2] = 0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        predicted_bbox = convert_x_to_bbox(self.kf.x)
        self.history.append(predicted_bbox)
        return predicted_bbox

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

class Sort(object):
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.25):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 1

    def update(self, detections_in_frame):
        self.frame_count += 1
        dets_boxes = []
        dets_info = []
        for det in detections_in_frame:
            dets_boxes.append(det['box'])
            dets_info.append({'class_id': det['class_id'], 'label': det['label'], 'score': det['score']})

        dets_boxes = np.array(dets_boxes) if len(dets_boxes) > 0 else np.empty((0, 4))

        active_trackers_indices = []
        predicted_boxes_for_iou = []

        for i in range(len(self.trackers)):
            predicted_box = self.trackers[i].predict()[0]
            if not np.any(np.isnan(predicted_box)):
                active_trackers_indices.append(i)
                predicted_boxes_for_iou.append(predicted_box)

        current_active_trackers = [self.trackers[i] for i in active_trackers_indices]
        if len(predicted_boxes_for_iou) > 0:
            trks_predicted_boxes_for_iou = np.asarray(predicted_boxes_for_iou)
        else:
            trks_predicted_boxes_for_iou = np.empty((0, 4))

        matched_indices = []
        if dets_boxes.shape[0] > 0 and trks_predicted_boxes_for_iou.shape[0] > 0:
            iou_matrix = iou_batch(dets_boxes, trks_predicted_boxes_for_iou)
            iou_matrix[iou_matrix < self.iou_threshold] = 0.0

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))

        unmatched_detections = [d for d in range(dets_boxes.shape[0])]
        unmatched_trackers_active_list_indices = [t for t in range(len(current_active_trackers))]

        for r, c in matched_indices:
            if r in unmatched_detections:
                unmatched_detections.remove(r)
            if c in unmatched_trackers_active_list_indices:
                unmatched_trackers_active_list_indices.remove(c)

        for r_det_idx, c_active_trk_idx in matched_indices:
            trk_to_update = current_active_trackers[c_active_trk_idx]
            trk_to_update.update(dets_boxes[r_det_idx, :],
                                 dets_info[r_det_idx]['class_id'],
                                 dets_info[r_det_idx]['label'],
                                 dets_info[r_det_idx]['score'])

        newly_created_trackers = []
        for i_det_idx in unmatched_detections:
            det_box = dets_boxes[i_det_idx, :]
            info = dets_info[i_det_idx]
            trk = KalmanBoxTracker(det_box, info['class_id'], info['label'], info['score'])
            newly_created_trackers.append(trk)

        all_potential_trackers = current_active_trackers + newly_created_trackers

        ret = []
        final_trackers_for_next_frame = []

        for trk in all_potential_trackers:
            current_bbox = trk.get_state()[0]

            if (trk.time_since_update == 0 and trk.hits >= self.min_hits) or \
               (self.frame_count <= self.min_hits and trk.hits > 0):
                ret.append(np.concatenate((current_bbox,
                                           [trk.id, trk.class_id, trk.score, trk.label_name])).reshape(1,-1))

            if trk.time_since_update <= self.max_age:
                final_trackers_for_next_frame.append(trk)

        self.trackers = final_trackers_for_next_frame

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 8))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Menggunakan device: {device}")

NUM_CLASSES = 4 

try:
    model = get_custom_model_architecture(num_classes=NUM_CLASSES)
    model_path = "best_model30.pth" #model yang sudah dilatih sebelumnya
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model '{model_path}' berhasil dimuat.")
    
    preprocess_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Pastikan file 'best_modal_colab.pth' sudah diunggah ke lingkungan Colab Anda.")
    exit()
except RuntimeError as e:
    print(f"Error saat memuat state_dict model: {e}")
    print("Ini mungkin terjadi karena jumlah kelas (NUM_CLASSES) atau arsitektur model tidak cocok.")
    print(f"Pastikan NUM_CLASSES ({NUM_CLASSES}) sesuai dengan jumlah kelas saat '{model_path}' dilatih.")
    print("Juga pastikan fungsi 'get_custom_model_architecture' mendefinisikan arsitektur yang sama persis.")
    exit()
except Exception as e:
    print(f"Error tak terduga saat memuat model: {e}")
    exit()

model.eval()
model.to(device)

CUSTOM_MODEL_CLASS_NAMES = [
    '__background__', 'car', 'bus', 'truck' 
]

CLASS_NAMES_USED_BY_MODEL = CUSTOM_MODEL_CLASS_NAMES

TARGET_CLASS_NAMES = ['car', 'bus', 'truck']
TARGET_CLASS_IDS = []
missing_classes = []
for class_name in TARGET_CLASS_NAMES:
    try:
        TARGET_CLASS_IDS.append(CLASS_NAMES_USED_BY_MODEL.index(class_name))
    except ValueError:
        missing_classes.append(class_name)

if missing_classes:
    print(f"Error: Kelas target berikut tidak ditemukan dalam daftar kelas model Anda: {missing_classes}")
    exit()
if not TARGET_CLASS_IDS:
    print("Error: Tidak ada kelas target yang valid ditemukan.")
    exit()

print(f"Tracking target classes: {TARGET_CLASS_NAMES} with IDs: {TARGET_CLASS_IDS}")


CONFIDENCE_THRESHOLD = 0.65
OUTPUT_VIDEO_FOLDER = "Output_Videos"
OUTPUT_JSON_FOLDER = "Output_Json_Data"

if not os.path.exists(OUTPUT_VIDEO_FOLDER):
    os.makedirs(OUTPUT_VIDEO_FOLDER)
if not os.path.exists(OUTPUT_JSON_FOLDER):
    os.makedirs(OUTPUT_JSON_FOLDER)

def transform_frame_for_model(frame_cv2):
    image_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    img_tensor = preprocess_transform(pil_image)
    return img_tensor

def get_detections_from_frame(frame_cv2, current_model, current_device):
    img_tensor_original_shape = transform_frame_for_model(frame_cv2)
    img_tensor_device = img_tensor_original_shape.unsqueeze(0).to(current_device)

    raw_detections_data = []
    top_edge_threshold = 10

    with torch.no_grad():
        prediction = current_model(img_tensor_device)

    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()

    h_frame, w_frame = frame_cv2.shape[:2]

    for i in range(len(pred_scores)):
        score = float(pred_scores[i])
        label_id = int(pred_labels[i])
        box = pred_boxes[i]

        # Extract coordinates first
        xmin = max(0, int(box[0]))
        ymin = max(0, int(box[1]))
        xmax = min(w_frame, int(box[2]))
        ymax = min(h_frame, int(box[3]))

        if ymin > top_edge_threshold and label_id in TARGET_CLASS_IDS and score >= CONFIDENCE_THRESHOLD:
            label_name = CLASS_NAMES_USED_BY_MODEL[label_id]

            width = xmax - xmin
            height = ymax - ymin
            aspect_ratio = width / (height + 1e-6)
            area = width * height

            min_area = 1000
            max_aspect = 3
            min_aspect = 0.5

            if area > min_area and min_aspect < aspect_ratio < max_aspect:
                raw_detections_data.append({
                    'box': [xmin, ymin, xmax, ymax],
                    'label': label_name,
                    'class_id': label_id,
                    'score': score
                })
    return raw_detections_data


if __name__ == "__main__":
    video_to_process = 'Amplaz01a_part_1.mp4' #ganti sesuai video yang ingin diproses

    if not video_to_process or not os.path.exists(video_to_process):
        print(f"Error: Video file not found at '{video_to_process}'.")
        print("Please check the path and try again.")
        exit()

    base_video_filename = os.path.basename(video_to_process)
    video_name_without_ext = os.path.splitext(base_video_filename)[0]

    output_video_path = os.path.join(OUTPUT_VIDEO_FOLDER, f"tracked_{base_video_filename}")
    output_json_path = os.path.join(OUTPUT_JSON_FOLDER, f"{video_name_without_ext}_tracked_data.json")

    print(f"\n[INFO] Memproses video: {video_to_process}...")
    print(f"[INFO] Output video akan disimpan ke: {output_video_path}")
    print(f"[INFO] Output JSON akan disimpan ke: {output_json_path}")

    sort_tracker = Sort(max_age=40, min_hits=5, iou_threshold=0.3)
    current_video_tracked_frames_data = []
    video_writer = None
    cap = None

    try:
        cap = cv2.VideoCapture(video_to_process)
        if not cap.isOpened():
            print(f"Peringatan: Tidak bisa membuka video {video_to_process}. Dilewati.")
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                print(f"Peringatan: Tidak bisa mendapatkan total frame untuk {video_to_process}. Progress tidak akan akurat.")
                total_frames = -1

            if fps <= 0:
                print(f"Peringatan: FPS video {video_to_process} adalah {fps}. Menggunakan default FPS = 25.")
                fps = 25

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            if not video_writer.isOpened():
                print(f"Peringatan: Tidak bisa menginisialisasi VideoWriter untuk {output_video_path}. Output video tidak akan disimpan.")
                video_writer = None
    except Exception as e:
        print(f"Error saat membuka video {video_to_process} atau menginisialisasi VideoWriter: {e}. Dilewati.")
        if video_writer and video_writer.isOpened(): video_writer.release()
        if cap and cap.isOpened(): cap.release()
        exit() 

    frame_id_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id_counter += 1
        frame_for_drawing = frame.copy()

        if total_frames > 0:
            print(f"\r[INFO] Memproses video: {base_video_filename} - Frame: {frame_id_counter}/{total_frames}", end="")
        else:
            print(f"\r[INFO] Memproses video: {base_video_filename} - Frame: {frame_id_counter}", end="")


        try:
            detections_current_frame = get_detections_from_frame(frame, model, device)
            tracked_objects_output = sort_tracker.update(detections_current_frame)
            frame_tracked_data_for_json = []

            if tracked_objects_output.shape[0] > 0:
                for trk_data in tracked_objects_output:
                    try:
                        x1 = int(float(trk_data[0]))
                        y1 = int(float(trk_data[1]))
                        x2 = int(float(trk_data[2]))
                        y2 = int(float(trk_data[3]))
                        track_id = int(float(trk_data[4]))
                        class_id_trk = int(float(trk_data[5]))
                        score_trk = float(trk_data[6])
                        label_name_trk = str(trk_data[7]) 

                    except (IndexError, ValueError) as e:
                        print(f"Error memproses data tracker: {e}, data: {trk_data}. Dilewati.")
                        continue

                    frame_tracked_data_for_json.append({
                        'box': [x1, y1, x2, y2],
                        'track_id': track_id,
                        'label': label_name_trk,
                        'class_id': class_id_trk,
                        'score': score_trk
                    })

                    cv2.rectangle(frame_for_drawing, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"ID:{track_id} {label_name_trk}: {score_trk:.2f}"
                    text_y_position = y1 - 10 if y1 - 10 > 10 else y1 + 15
                    cv2.putText(frame_for_drawing, label_text, (x1, text_y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_video_tracked_frames_data.append({
                'frame_id': frame_id_counter,
                'tracked_objects': frame_tracked_data_for_json
            })

            if video_writer:
                video_writer.write(frame_for_drawing)

        except Exception as e:
            print(f"\nError saat deteksi/tracking pada frame {frame_id_counter} dari video {video_to_process}: {e}")
            import traceback
            traceback.print_exc()

    print() 

    if cap and cap.isOpened():
        cap.release()
    if video_writer and video_writer.isOpened():
        video_writer.release()
        print(f"[INFO] Video hasil tracking disimpan ke: {output_video_path}")

    try:
        with open(output_json_path, 'w') as f_json:
            json.dump(current_video_tracked_frames_data, f_json, indent=4)
        print(f"[INFO] Data tracking JSON untuk '{base_video_filename}' disimpan ke: {output_json_path}")
    except IOError:
        print(f"Error: Tidak bisa menulis data JSON ke file {output_json_path}.")
    except Exception as e:
        print(f"Error saat menyimpan file JSON {output_json_path}: {e}")

    print(f"[INFO] Selesai memproses video: {video_to_process}")

    print("\n[INFO] Semua proses selesai.")