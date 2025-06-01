import numpy as np
import cv2
import json
import argparse
import os
import glob

refPt = []
cropping = False 
image = None 
clone = None 

FRAMES_BASE_DIR = r"Dataset"

POINTS_JSON_PATH = "source_material/reference_pts/points.json"
LINES_JSON_PATH = "source_material/speedDetectionSpots/detectionSpots.json"
ANNOTATED_FRAMES_DIR = "source_material/annotated_frames"

# Callback for point selection
def pointSelector(event, x, y, flags, param):
    global refPt, image 

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 255), -1) 
        cv2.imshow("image", image) 
        pass

# Callback for line selection
def lineSelector(event, x, y, flags, param):
    global refPt, cropping, image, clone 

    if event == cv2.EVENT_LBUTTONDOWN:
        # If starting a new line or have 2 points already, reset refPt for the new line
        if not cropping or len(refPt) == 2:
            refPt = [(x, y)]
            image = clone.copy() 
        elif len(refPt) == 1: 
            refPt.append((x,y))

        cropping = True
        # Draw the first point of the line
        if refPt: 
            cv2.circle(image, refPt[0], 5, (0, 255, 0), -1)
        cv2.imshow("image", image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping and refPt: 
            temp_image = image.copy()
            if len(refPt) == 1:
                cv2.line(temp_image, refPt[0], (x, y), (0, 255, 0), 2)
                cv2.imshow("image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        if cropping: 
            if len(refPt) == 1: 
                refPt.append((x, y))
            
            cropping = False
            if len(refPt) == 2: 
                cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 2)
                cv2.circle(image, refPt[0], 5, (0, 255, 0), -1)
                cv2.circle(image, refPt[1], 5, (0, 255, 0), -1) 
                cv2.imshow("image", image)

def main():
    global refPt, image, clone, cropping
    
    OUTPUT_WINDOW_WIDTH = 1280  
    OUTPUT_WINDOW_HEIGHT = 720 

    ap = argparse.ArgumentParser(description="Select points or lines on a video frame for annotation.")
    ap.add_argument("-n", "--number", required=True, 
                    help="Identifier for the video subfolder within the dataset (e.g., 'amplaz01', 'FKH01')")
    ap.add_argument("-m", "--mode", required=True, choices=['point', 'line'], help="Annotation mode: 'point' or 'line'")
    args = vars(ap.parse_args())

    video_id_str = args["number"] 
    mode = args["mode"]
    print(f"Selected Video Subfolder: {video_id_str}, Mode: {mode}")
    
    video_frames_folder = os.path.join(FRAMES_BASE_DIR, video_id_str)
    if not os.path.isdir(FRAMES_BASE_DIR):
        raise Exception(f"Error: The base dataset directory does not exist: {FRAMES_BASE_DIR}\n"
                        "Please update the 'FRAMES_BASE_DIR' variable in the script.")
    if not os.path.isdir(video_frames_folder):
        raise Exception(f"Error: The specified video subfolder does not exist: {video_frames_folder}\n"
                        f"Ensure that a folder named '{video_id_str}' exists inside '{FRAMES_BASE_DIR}'.")

    frame_loaded = False
    image_path_loaded = None 
    for ext_pattern in ["*.png", "*.jpg", "*.jpeg"]:
        current_frame_search_pattern = os.path.join(video_frames_folder, ext_pattern)
        matching_frames = sorted(glob.glob(current_frame_search_pattern)) 
        
        if matching_frames:
            image_path_loaded = matching_frames[0] 
            image = cv2.imread(image_path_loaded)
            if image is not None:
                print(f"Loaded frame: {image_path_loaded}")
                frame_loaded = True
                break 
            else:
                print(f"Warning: Found frame file {image_path_loaded} but could not read it. Trying next extension or type.")
    
    if not frame_loaded:
        searched_patterns_info = [os.path.join(video_frames_folder, ext) for ext in ["*.png", "*.jpg", "*.jpeg"]]
        raise Exception(f"Error: Could not load any frame for video subfolder '{video_id_str}'.\n"
                        f"Looked for image files (e.g., {', '.join(searched_patterns_info)})\n"
                        f"within folder: {video_frames_folder}\n"
                        "Please ensure image frames (not videos) are present in this subfolder.")

    clone = image.copy()
    cv2.namedWindow("image")

    if mode == "point":
        cv2.setMouseCallback("image", pointSelector)
        print("Point selection mode. Click to add points. Press 'r' to reset, 'c' to confirm and save.")
    elif mode == "line":
        cv2.setMouseCallback("image", lineSelector)
        print("Line selection mode. Click and drag to draw a line. Press 'r' to reset, 'c' to confirm and save.")
    
    refPt = [] 
    
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"): 
            image = clone.copy()
            refPt = []
            cropping = False 
            print("Frame reset. Redraw your annotations.")
            if mode == "line": 
                cv2.imshow("image", image)

        elif key == ord("c"): 
            print("Annotations confirmed.")
            break

    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1) 
    
    os.makedirs(os.path.dirname(POINTS_JSON_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LINES_JSON_PATH), exist_ok=True)
    os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)

    if mode == "point":
        try:
            with open(POINTS_JSON_PATH, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"points": {}} 

        if "points" not in data: 
            data["points"] = {}
        data["points"][video_id_str] = {"2dpts": refPt} 
        
        with open(POINTS_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Point data saved to {POINTS_JSON_PATH} under ID '{video_id_str}'")

        annotated_image_path = os.path.join(ANNOTATED_FRAMES_DIR, f"vid_{video_id_str}_points_annotated.jpg")
        cv2.imwrite(annotated_image_path, image)
        print(f"Annotated image saved to {annotated_image_path}")

    elif mode == "line":
        try:
            with open(LINES_JSON_PATH, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        
        data[video_id_str] = refPt 
        
        with open(LINES_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Line data saved to {LINES_JSON_PATH} under ID '{video_id_str}'")
        
        annotated_image_path = os.path.join(ANNOTATED_FRAMES_DIR, f"vid_{video_id_str}_lines_annotated.jpg")
        cv2.imwrite(annotated_image_path, image)
        print(f"Annotated image saved to {annotated_image_path}")

if __name__ == "__main__":
    main()