import cv2
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import os

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Memuat model yang telah dilatih
def load_trained_model(model_path, num_classes, device):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Transformasi untuk inferensi
inference_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Fungsi untuk melakukan deteksi pada satu frame
def detect_vehicles(model, frame, device, score_threshold=0.7):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = inference_transform(pil_image)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        prediction = model([img_tensor])

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()


    indices = scores > score_threshold
    filtered_boxes = boxes[indices]
    filtered_labels = labels[indices]
    filtered_scores = scores[indices]

    return filtered_boxes, filtered_labels, filtered_scores

# Fungsi untuk inferensi video
def infer_video(video_path, model_path, output_video_path="output_video.mp4", num_classes=2):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(model_path, num_classes, DEVICE)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    print(f"Processing video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi kendaraan
        boxes, labels, scores = detect_vehicles(model, frame, DEVICE)

        for box, label, score in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = map(int, box)
            
            # Visualisasi
            color = (0, 255, 0)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, f"Vehicle: {score:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        frame_count += 1
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")

    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")
    
if __name__ == '__main__':
    video_input_path = "Amplaz01a_part_1.mp4" #ganti input video path sesuai kebutuhan
    
    video_output_path = "output_detected_video.mp4"
    
    trained_model_path = "best_model_colab.pth" 

    if not os.path.exists(trained_model_path):
        print(f"Error: Model file not found at {trained_model_path}")
        print("Please ensure 'best_model_colab.pth' is uploaded to your Colab environment or the path is correct.")
    else:
        NUM_CLASSES = 4 

        print(f"Starting inference with model: {trained_model_path}")
        infer_video(video_input_path, trained_model_path, video_output_path, num_classes=NUM_CLASSES)