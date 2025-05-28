import sys
import numpy as np
import cv2
import json
import argparse
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POINTS_JSON_PATH = os.path.join(BASE_DIR, "source_material", "reference_pts", "points.json")
FRAMES_BASE_DIR      = os.path.join(BASE_DIR, "frames")
OUTPUT_HOMOGRAPHY_PATH = os.path.join(BASE_DIR, "source_material", "homography_matrix.json")
ANNOTATED_FRAMES_DIR   = os.path.join(BASE_DIR, "source_material", "annotated_frames")

def load_points(video_id_str):
    """Loads both 2D image points and corresponding world points from points.json."""
    try:
        with open(POINTS_JSON_PATH, 'r') as f:
            data = json.load(f)
        rec = data.get("points", {}).get(video_id_str, {})
        img = rec.get("2dpts", [])
        wld = rec.get("world_pts", [])
        if not img:
            print(f"Error: No 2D points for video ID '{video_id_str}' in {POINTS_JSON_PATH}")
            return None, None
        if not wld:
            print(f"Warning: No world_pts for video ID '{video_id_str}' in {POINTS_JSON_PATH}. Will rely on script definition if available.")
            wld = []
        img = np.array(img, dtype="float32")
        wld = np.array(wld, dtype="float32")
        if wld.size > 0 and img.shape[0] != wld.shape[0]:
            print(f"Error: Mismatch count from JSON: {img.shape[0]} image points vs {wld.shape[0]} world points")
            return None, None
        if img.shape[0] < 4:
            print("Error: Need at least 4 point correspondences for homography")
            return None, None
        return img, wld
    except Exception as e:
        print(f"Failed loading points: {e}")
        return None, None

def load_first_frame(video_id_str):
    for ext in ("*.png","*.jpg","*.jpeg"):
        pattern = os.path.join(FRAMES_BASE_DIR, f"vid{video_id_str}_frame_{ext}")
        files = sorted(glob.glob(pattern))
        if files:
            img_path = files[0]
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_path
    print(f"Warning: No frame found for video ID {video_id_str} in {FRAMES_BASE_DIR} with pattern vid{video_id_str}_frame_*")
    return None, None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n","--number", required=True, help="video ID number")
    args = p.parse_args()
    vid = args.number

    print(f"--- Homography Calculation for Video ID: {vid} ---")
    img_pts, world_pts_from_json = load_points(vid)

    if img_pts is None:
        sys.exit(1)

    print(f"Loaded Image Points (from {POINTS_JSON_PATH}):")
    print(img_pts)
    num_expected_points = len(img_pts)
    print(f"Number of image points loaded: {num_expected_points}")

    world_pts = None 

    if vid == "1" and num_expected_points == 12:
        print("Defining world_points manually for Video ID 1 (12 points).")
        world_pts = np.array([
            [0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0],
            [0.0, 10.0], [10.0, 10.0], [20.0, 10.0], [30.0, 10.0],
            [0.0, 20.0], [10.0, 20.0], [20.0, 20.0], [30.0, 20.0]
        ], dtype="float32")
    elif vid == "2" and num_expected_points == 14: 
        print("Defining world_points manually for Video ID 2 (14 points).") 
        # TIITIK JARAK METER GROUND TRUTH (DUMMY WORLD PTS)
        world_pts = np.array([
            # Left
            [0.0, 50.0], [10.0, 45.0], [20.0, 40.0], [30.0, 35.0],
            [40.0, 30.0], [50.0, 25.0], [60.0, 20.0],
            # Right
            [0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0],
            [40.0, 0.0], [50.0, 0.0], [60.0, 0.0]
        ], dtype="float32")
    elif vid == "3":
        # FOR VID 3 IF 3 point Dummy Data
        if num_expected_points == 10:
            print(f"Defining world_points manually for Video ID 3 ({num_expected_points} points).")
            world_pts = np.array([
                [0.0, 30.0], [10.0, 25.0], [20.0, 20.0], [30.0, 15.0], [40.0, 10.0], # Left
                [0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]    # Right
            ], dtype="float32")
        elif num_expected_points == 13:
            print(f"Defining world_points manually for Video ID 3 ({num_expected_points} points).")
            world_pts = np.array([
                [0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0], 
                [0.0, 8.0], [5.0, 8.0], [10.0, 8.0], [15.0, 8.0], [20.0, 8.0], 
                [0.0, 16.0], [5.0, 16.0], [10.0, 16.0]                         
            ], dtype="float32")
        else:
            print(f"Error: For Video ID '{vid}', loaded {num_expected_points} image points.")
            print(f"Manual definition for vid == '3' in this script expects 10 or 13 points.")
            print(f"Please adjust the condition or check your 2dpts in points.json for video 3.")
            sys.exit(1)
    else: 
        if world_pts_from_json.size > 0 and len(world_pts_from_json) == num_expected_points:
            print("Using world_pts loaded from JSON.")
            world_pts = world_pts_from_json
        else:
            print(f"Error: world_pts not found or mismatch for video ID '{vid}' ({num_expected_points} points).")
            print(f" - Not found in JSON (JSON has {len(world_pts_from_json)} points for world_pts), or count mismatch.")
            print(f" - Not manually defined in script for this ID and point count combination.")
            print("Please update points.json or add/correct the manual definition in 03_calib_homography.py main().")
            sys.exit(1)

    if world_pts is None: 
        print(f"CRITICAL ERROR: world_pts was not assigned for video ID '{vid}' with {num_expected_points} points. Review logic for defining world_pts.")
        sys.exit(1)
    if len(img_pts) != len(world_pts):
        print(f"CRITICAL ERROR: Mismatch after world_pts definition. Image points: {len(img_pts)}, World points: {len(world_pts)}")
        sys.exit(1)

    print("Using World Points:")
    print(world_pts)
    H, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Error: Homography computation failed. Check point correspondences and values.")
        sys.exit(1)

    print("Homography matrix:")
    print(H)
    if mask is not None:
        print(f"Inliers: {int(mask.sum())}/{len(mask)}")
    else:
        print("Warning: RANSAC mask is None.")

    os.makedirs(os.path.dirname(OUTPUT_HOMOGRAPHY_PATH), exist_ok=True)
    allH = {}
    try:
        if os.path.exists(OUTPUT_HOMOGRAPHY_PATH):
            with open(OUTPUT_HOMOGRAPHY_PATH, 'r') as f: allH = json.load(f)
    except Exception as e: 
        print(f"Warning: Could not decode/load {OUTPUT_HOMOGRAPHY_PATH}: {e}. Starting with an empty homography dictionary.")
        allH = {} 
    allH[vid] = H.tolist()
    with open(OUTPUT_HOMOGRAPHY_PATH,"w") as f: json.dump(allH, f, indent=4)
    print("Saved homography to", OUTPUT_HOMOGRAPHY_PATH)

    # --- IMAGE ANNOTATE ---
    img_to_annotate, frame_path = load_first_frame(vid)
    if img_to_annotate is not None:
        out_image = img_to_annotate.copy()

        for i, (x, y) in enumerate(img_pts):
            cv2.circle(out_image, (int(x), int(y)), 7, (0, 255, 255), -1) 
            cv2.putText(out_image, f"I{i}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(out_image, f"I{i}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 2. Projecting this to your world homography
        if len(world_pts) > 0: 
            world_pts_reshaped = world_pts.reshape(-1, 1, 2)
            projected_image_pts_cv = cv2.perspectiveTransform(world_pts_reshaped, H)

            if projected_image_pts_cv is not None:
                for i, pt_proj_cv in enumerate(projected_image_pts_cv):
                    x_proj, y_proj = int(pt_proj_cv[0][0]), int(pt_proj_cv[0][1])
                    cv2.circle(out_image, (x_proj, y_proj), 8, (255, 100, 0), 2) 
                    cv2.circle(out_image, (x_proj, y_proj), 5, (0, 0, 255), -1)  
                    cv2.putText(out_image, f"P{i}", (x_proj + 8, y_proj - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                    cv2.putText(out_image, f"P{i}", (x_proj + 8, y_proj - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # --- MODIFICATE DRAWING ROAD---
        if len(img_pts) >= 4 and len(img_pts) % 2 == 0: 
            num_total_pts = len(img_pts)
            num_pts_per_edge = num_total_pts // 2
            print(f"Menggambar grid perspektif jalan menggunakan {num_total_pts} titik gambar ({num_pts_per_edge} per sisi).")
            pts_int = img_pts.astype(np.int32)

            color_longitudinal = (255, 255, 0)  
            color_transversal = (0, 255, 255)   
            thickness = 2

            # 1. Draw line in left road
            for i in range(num_pts_per_edge - 1):
                pt1 = tuple(pts_int[i])
                pt2 = tuple(pts_int[i+1])
                cv2.line(out_image, pt1, pt2, color_longitudinal, thickness)

            # 2. Draw line in right road
            for i in range(num_pts_per_edge - 1):
                idx1 = num_pts_per_edge + i
                idx2 = num_pts_per_edge + i + 1
                pt1 = tuple(pts_int[idx1])
                pt2 = tuple(pts_int[idx2])
                cv2.line(out_image, pt1, pt2, color_longitudinal, thickness)
                
        else:
            print(f"Peringatan: Jumlah titik gambar adalah {len(img_pts)}. Grid perspektif tidak digambar (membutuhkan jumlah titik genap >= 4).")

        os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)
        out_path = os.path.join(ANNOTATED_FRAMES_DIR, f"vid_{vid}_annot_grid.jpg") 
        try:
            cv2.imwrite(out_path, out_image)
            print("Annotated image saved to", out_path)
        except Exception as e: print(f"Error saving annotated image: {e}")
    else:
        print(f"Warning: no frame loaded to annotate for video ID {vid}. Checked in {FRAMES_BASE_DIR}")
    print("--- Done ---")

if __name__=="__main__":
    main()