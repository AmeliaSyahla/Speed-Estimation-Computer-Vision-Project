import numpy as np
import cv2
import json
import argparse
import os
import glob

# Global variables
refPt = []
cropping = False # Initialize cropping state
image = None # Will hold the current frame
clone = None # Will hold a copy of the original frame for reset

# --- Configuration for Frame Input ---
FRAMES_BASE_DIR = r"D:\Amel Cantik\Semester 4\PKAC - Amelia Syahla Aurellia Sambudi\Final Project CV\frames"

# --- Output Directories ---
# These directories will be created if they don't exist.
POINTS_JSON_PATH = "source_material/reference_pts/points.json"
LINES_JSON_PATH = "source_material/speedDetectionSpots/detectionSpots.json"
ANNOTATED_FRAMES_DIR = "source_material/annotated_frames"

# Callback for point selection
def pointSelector(event, x, y, flags, param):
    global refPt, image # image is modified in place

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 255), -1) # Draw a yellow circle at the point
        cv2.imshow("image", image) # Update the image display
    # elif event == cv2.EVENT_LBUTTONUP: # Not strictly needed for point selection
        # cv2.imshow("image", image)
        pass

# Callback for line selection
def lineSelector(event, x, y, flags, param):
    global refPt, cropping, image, clone # image and clone are used

    if event == cv2.EVENT_LBUTTONDOWN:
        # If starting a new line or have 2 points already, reset refPt for the new line
        if not cropping or len(refPt) == 2:
            refPt = [(x, y)]
            image = clone.copy() # Reset image to draw new line from scratch if needed
            # Redraw previous permanent lines if any (not implemented in this simple version)
        elif len(refPt) == 1: # If one point is already set, this is the second point
            refPt.append((x,y))

        cropping = True
        # Draw the first point of the line
        cv2.circle(image, refPt[0], 5, (0, 255, 0), -1)
        cv2.imshow("image", image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping and refPt: # If dragging and a start point exists
            temp_image = image.copy()
            if len(refPt) == 1: # Drawing the line dynamically
                cv2.line(temp_image, refPt[0], (x, y), (0, 255, 0), 2)
                cv2.imshow("image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        if cropping: # Ensure we were actually cropping
            if len(refPt) == 1: # This is the second point of the line
                refPt.append((x, y))
            
            cropping = False # End cropping
            if len(refPt) == 2: # If two points are defined, draw the permanent line
                cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 2)
                cv2.circle(image, refPt[0], 5, (0, 255, 0), -1) # Start point
                cv2.circle(image, refPt[1], 5, (0, 255, 0), -1) # End point
                cv2.imshow("image", image)
            # else: # Only one point was set, or something went wrong
                # image = clone.copy() # Reset if line was not completed
                # refPt = []
                # cv2.imshow("image", image)


def main():
    global refPt, image, clone, cropping
    
    OUTPUT_WINDOW_WIDTH = 1280  
    OUTPUT_WINDOW_HEIGHT = 720 

    # Argument parser
    ap = argparse.ArgumentParser(description="Select points or lines on a video frame for annotation.")
    ap.add_argument("-n", "--number", required=True, help="Number of the video instance (e.g., '1' for 'vid_1_...')")
    ap.add_argument("-m", "--mode", required=True, choices=['point', 'line'], help="Annotation mode: 'point' or 'line'")
    args = vars(ap.parse_args())

    video_id_str = args["number"]
    mode = args["mode"]
    print(f"Selected Video ID: {video_id_str}, Mode: {mode}")

    # --- Load the first available frame for the specified video ID ---
    # Assuming frame names like 'vid_1_frame_0000.png', 'vid_1_frame_0001.jpg', etc.
    # It will try .png, then .jpg, then .jpeg
    frame_loaded = False
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        frame_files_pattern = os.path.join(FRAMES_BASE_DIR, f"vid{video_id_str}_frame_{ext}")
        matching_frames = sorted(glob.glob(frame_files_pattern))
        if matching_frames:
            image_path = matching_frames[0] # Take the first frame found
            image = cv2.imread(image_path)
            if image is not None:
                print(f"Loaded frame: {image_path}")
                frame_loaded = True
                break
            else:
                print(f"Warning: Found frame file {image_path} but could not read it.")
        else:
            print(f"No frames found with pattern: {frame_files_pattern}")


    if not frame_loaded or image is None:
        raise Exception(f"Error: Could not load any frame for video ID {video_id_str} from {FRAMES_BASE_DIR}")

    clone = image.copy()
    cv2.namedWindow("image")

    if mode == "point":
        cv2.setMouseCallback("image", pointSelector)
        print("Point selection mode. Click to add points. Press 'r' to reset, 'c' to confirm and save.")
    elif mode == "line":
        cv2.setMouseCallback("image", lineSelector)
        print("Line selection mode. Click and drag to draw a line. Press 'r' to reset, 'c' to confirm and save.")
    
    refPt = [] # Reset refPt for each run

    # Main loop for display and interaction
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"): # Reset
            image = clone.copy()
            refPt = []
            cropping = False # Reset cropping state for lines
            print("Frame reset. Redraw your annotations.")
            if mode == "line": # For line mode, ensure the original image is shown without temp lines
                cv2.imshow("image", image)


        elif key == ord("c"): # Confirm and save
            print("Annotations confirmed.")
            break

    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1) # Helps ensure windows close properly

    # --- Create output directories if they don't exist ---
    os.makedirs(os.path.dirname(POINTS_JSON_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LINES_JSON_PATH), exist_ok=True)
    os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)

    # --- Save annotations to JSON file and image ---
    if mode == "point":
        try:
            with open(POINTS_JSON_PATH, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"points": {}} # Initialize if file not found or invalid JSON

        # Ensure the nested structure exists
        if "points" not in data:
            data["points"] = {}
        if video_id_str not in data["points"]:
            data["points"][video_id_str] = {}
        
        data["points"][video_id_str]["2dpts"] = refPt
        
        with open(POINTS_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Point data saved to {POINTS_JSON_PATH}")

        annotated_image_path = os.path.join(ANNOTATED_FRAMES_DIR, f"vid_{video_id_str}_points_annotated.jpg")
        cv2.imwrite(annotated_image_path, image)
        print(f"Annotated image saved to {annotated_image_path}")

    elif mode == "line":
        try:
            with open(LINES_JSON_PATH, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {} # Initialize if file not found or invalid JSON
        
        data[video_id_str] = refPt # Store the line (pair of points)
        
        with open(LINES_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Line data saved to {LINES_JSON_PATH}")
        
        annotated_image_path = os.path.join(ANNOTATED_FRAMES_DIR, f"vid_{video_id_str}_lines_annotated.jpg")
        cv2.imwrite(annotated_image_path, image)
        print(f"Annotated image saved to {annotated_image_path}")

if __name__ == "__main__":
    main()
