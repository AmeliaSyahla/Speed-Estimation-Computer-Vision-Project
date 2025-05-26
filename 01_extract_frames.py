import cv2
import os
import glob

def extract_frames_from_videos(video_folder_path, output_folder_path, frame_interval_seconds=0.5):
    # Buat folder output jika belum ada
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output folder: {output_folder_path}")

    # format video dataset
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm") 
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder_path, ext)))

    if not video_files:
        print(f"No video files found in {video_folder_path}")
        return

    print(f"Found {len(video_files)} video(s) to process.")

    for video_path in video_files:
        video_filename = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(video_filename)[0]
        print(f"\nProcessing video: {video_filename}...")

        try:
            # Buka file video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_filename}")
                continue

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                print(f"Warning: FPS for {video_filename} is 0. Skipping or using default might be needed.")
                fps = 30  # Default FPS if unable to read
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_skip = int(fps * frame_interval_seconds)
            if frame_skip <= 0 : # Jika frame_skip <= 0, set ke 1 untuk menghindari pembacaan frame yang terlalu cepat
                frame_skip = 1
                print(f"Warning: Calculated frame_skip for {video_filename} was <= 0. Setting to 1 to extract at least some frames.")


            frame_count = 0
            saved_frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video or error

                if frame_count % frame_skip == 0:
                    # Nama unik setiap frame
                    # Format: <video_name>_frame_<frame_number>.jpg
                    output_frame_name = f"{video_name_without_ext}_frame_{saved_frame_count:04d}.jpg"
                    output_frame_path = os.path.join(output_folder_path, output_frame_name)

                    # Save frame
                    cv2.imwrite(output_frame_path, frame)
                    saved_frame_count += 1

                frame_count += 1

            cap.release()
            print(f"Finished processing {video_filename}. Saved {saved_frame_count} frames.")

        except Exception as e:
            print(f"An error occurred while processing {video_filename}: {e}")

    print(f"\n--- All videos processed. Frames saved in {output_folder_path} ---")

if __name__ == "__main__":
    # 1. Path input
    input_videos_directory = "D:\Amel Cantik\Semester 4\PKAC - Amelia Syahla Aurellia Sambudi\Final Project CV\Dataset"  

    # 2. Output folder untuk frames 
    output_frames_directory = "frames"

    # 3. Set time per frame (sekon)
    interval_seconds = 0.5

    # Cek apakah folder input video ada
    if not os.path.isdir(input_videos_directory):
        print(f"Error: The input video directory '{input_videos_directory}' does not exist.")
        print("Please create the directory and place your video files inside it, or correct the path.")
    else:
        extract_frames_from_videos(input_videos_directory, output_frames_directory, interval_seconds)
        print(f"\nExtraction complete! The '{output_frames_directory}/' folder is ready for upload to Roboflow.")