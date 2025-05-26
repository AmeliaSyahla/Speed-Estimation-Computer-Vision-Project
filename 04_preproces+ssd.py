import torch
import torchvision
from torchvision.models.detection import SSD300_VGG16_Weights
from PIL import Image
import cv2
import numpy as np
import os # Untuk operasi file dan direktori
import json # Untuk menyimpan hasil dalam format JSON

# --- Konfigurasi Awal ---
# 1. Tentukan Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Menggunakan device: {device}")

# 2. Muat Model SSD Pre-trained dan dapatkan transformasi pra-pemrosesannya
try:
    weights = SSD300_VGG16_Weights.COCO_V1
    model = torchvision.models.detection.ssd300_vgg16(weights=weights)
    preprocess_transform = weights.transforms()
except AttributeError:
    print("Menggunakan metode fallback untuk memuat model (mungkin tanpa weights.transforms())")
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    preprocess_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("PERINGATAN: Menggunakan transformasi manual. Pastikan ini sesuai.")

# 3. Set Model ke Mode Evaluasi dan pindahkan ke device
model.eval()
model.to(device)

# Daftar kelas COCO
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

TARGET_CLASS_NAMES = ['car', 'bus', 'truck']
TARGET_CLASS_IDS = []
missing_classes = []
for class_name in TARGET_CLASS_NAMES:
    try:
        TARGET_CLASS_IDS.append(COCO_INSTANCE_CATEGORY_NAMES.index(class_name))
    except ValueError:
        missing_classes.append(class_name)

if missing_classes:
    print(f"Error: Kelas target berikut tidak ditemukan dalam daftar COCO: {missing_classes}")
    exit()
if not TARGET_CLASS_IDS:
    print("Error: Tidak ada kelas target yang valid ditemukan.")
    exit()

CONFIDENCE_THRESHOLD = 0.4 # Anda bisa menyesuaikan threshold ini
OUTPUT_JSON_FILE = "detection_results.json" # Nama file untuk menyimpan hasil

# --- TAMBAHAN: Konfigurasi Ukuran Tampilan ---
DISPLAY_WIDTH = 960  # Lebar jendela tampilan yang diinginkan
DISPLAY_HEIGHT = 540 # Tinggi jendela tampilan yang diinginkan
# Anda bisa menyesuaikan ukuran ini, misalnya 1280x720

# --- Fungsi Pra-pemrosesan Input ---
def preprocess_frame_for_ssd(frame_cv2):
    image_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    img_tensor = preprocess_transform(pil_image)
    return img_tensor

# --- Fungsi Deteksi Objek ---
def detect_objects_on_frame(frame_cv2, current_model, current_device):
    img_tensor_original_shape = preprocess_frame_for_ssd(frame_cv2) # Tensor sudah di-transform
    img_tensor_device = img_tensor_original_shape.to(current_device)

    detected_objects_data = [] # Untuk menyimpan data deteksi frame ini

    with torch.no_grad():
        prediction = current_model([img_tensor_device]) # Model menerima list of tensors

    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()

    frame_with_detections = frame_cv2.copy() # Salin frame untuk digambari

    for i in range(len(pred_scores)):
        if pred_scores[i] >= CONFIDENCE_THRESHOLD and pred_labels[i] in TARGET_CLASS_IDS:
            box = pred_boxes[i]
            label_id = pred_labels[i]
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
            score = float(pred_scores[i])

            h, w = frame_cv2.shape[:2]
            xmin = max(0, int(box[0]))
            ymin = max(0, int(box[1]))
            xmax = min(w, int(box[2]))
            ymax = min(h, int(box[3]))
            
            detected_objects_data.append({
                'box': [xmin, ymin, xmax, ymax],
                'label': label_name,
                'class_id': int(label_id),
                'score': score
            })

            cv2.rectangle(frame_with_detections, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label_text = f"{label_name}: {score:.2f}"
            cv2.putText(frame_with_detections, label_text, (xmin, ymin - 10 if ymin -10 > 10 else ymin + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
    return frame_with_detections, detected_objects_data

# --- Loop Utama Pemrosesan Video ---
if __name__ == "__main__":
    dataset_folder = "Dataset"
    all_videos_detections_results = {}

    try:
        video_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    except FileNotFoundError:
        print(f"Error: Folder '{dataset_folder}' tidak ditemukan.")
        exit()

    if not video_files:
        print(f"Tidak ada file video yang ditemukan di folder '{dataset_folder}'.")
        exit()

    for video_file in video_files:
        video_path = os.path.join(dataset_folder, video_file)
        print(f"\n[INFO] Memproses video: {video_path}...")
        
        current_video_frames_data = []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Peringatan: Tidak bisa membuka video {video_path}. Dilewati.")
                continue
        except Exception as e:
            print(f"Error saat membuka video {video_path}: {e}. Dilewati.")
            continue

        frame_id_counter = 0
        # --- PERUBAHAN: Buat nama jendela yang unik untuk setiap video ---
        window_name = f'Deteksi - {video_file} ({DISPLAY_WIDTH}x{DISPLAY_HEIGHT})'

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id_counter += 1
            
            try:
                frame_with_visuals, detections_in_frame = detect_objects_on_frame(frame, model, device)
                
                current_video_frames_data.append({
                    'frame_id': frame_id_counter,
                    'detections': detections_in_frame
                })

                # --- PERUBAHAN: Resize frame sebelum ditampilkan ---
                display_frame = cv2.resize(frame_with_visuals, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                cv2.imshow(window_name, display_frame) # Gunakan nama jendela yang sudah didefinisikan
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Proses dihentikan oleh pengguna untuk video ini.")
                    break 
            except Exception as e:
                print(f"Error saat deteksi pada frame {frame_id_counter} dari video {video_file}: {e}")
        
        all_videos_detections_results[video_file] = current_video_frames_data
        cap.release()
        # --- PERUBAHAN: Gunakan nama jendela yang sama saat menutup ---
        cv2.destroyWindow(window_name) 
        print(f"[INFO] Selesai memproses video: {video_file}")

    try:
        with open(OUTPUT_JSON_FILE, 'w') as f:
            json.dump(all_videos_detections_results, f, indent=4)
        print(f"\n[INFO] Semua hasil deteksi telah disimpan ke: {OUTPUT_JSON_FILE}")
    except IOError:
        print(f"Error: Tidak bisa menulis hasil deteksi ke file {OUTPUT_JSON_FILE}.")
    except Exception as e:
        print(f"Error saat menyimpan file JSON: {e}")

    cv2.destroyAllWindows()
    print("\n[INFO] Semua proses selesai.")