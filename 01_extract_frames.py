import cv2
import os
import glob

def extract_frames_from_videos(root_video_folder_path, root_output_folder_path, frame_interval_seconds=0.5):
    if not os.path.exists(root_output_folder_path):
        os.makedirs(root_output_folder_path)
        print(f"Created root output folder: {root_output_folder_path}")

    video_extensions = (".mp4")
    videos_processed_count = 0

    # Dataset to folder root
    for dirpath, dirnames, filenames in os.walk(root_video_folder_path):
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                videos_processed_count += 1
                video_path = os.path.join(dirpath, filename)
                video_name_without_ext = os.path.splitext(filename)[0]

                relative_dir = os.path.relpath(dirpath, root_video_folder_path)
                
                # Make folder for output
                if relative_dir == ".":
                    current_video_output_folder = root_output_folder_path
                else:
                    current_video_output_folder = os.path.join(root_output_folder_path, relative_dir)
                
                if not os.path.exists(current_video_output_folder):
                    os.makedirs(current_video_output_folder)
                    print(f"Created subfolder for video frames: {current_video_output_folder}")

                print(f"\nProcessing video: {video_path}...")

                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Error: Could not open video {filename}")
                        continue

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps == 0:
                        print(f"Warning: FPS for {filename} is 0. Using default FPS of 30.")
                        fps = 30
                    
                    frame_skip = max(1, int(fps * frame_interval_seconds))
                    if int(fps * frame_interval_seconds) <= 0 :
                        print(f"Warning: Calculated frame_skip for {filename} was <= 0 (FPS: {fps}, Interval: {frame_interval_seconds}s). Adjusted to 1.")


                    current_frame_index = 0 
                    saved_frame_count = 0   

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if current_frame_index % frame_skip == 0:
                            output_frame_name = f"{video_name_without_ext}_frame{saved_frame_count:04d}.jpg"
                            output_frame_path = os.path.join(current_video_output_folder, output_frame_name)

                            cv2.imwrite(output_frame_path, frame)
                            saved_frame_count += 1
                        
                        current_frame_index += 1

                    cap.release()
                    print(f"Finished processing {filename}. Saved {saved_frame_count} frames in {current_video_output_folder}.")

                except Exception as e:
                    print(f"An error occurred while processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()


    if videos_processed_count == 0:
        print(f"No video files found in {root_video_folder_path} and its subdirectories.")
    else:
        print(f"\n--- All {videos_processed_count} videos processed. Frames saved within {root_output_folder_path} ---")


if __name__ == "__main__":
    input_videos_directory = r"D:\Amel Cantik\Semester 4\PKAC - Amelia Syahla Aurellia Sambudi\Final Project CV\Dataset"
    output_frames_directory = "frames_dataset"
    interval_seconds = 0.5

    if not os.path.isdir(input_videos_directory):
        print(f"Error: The input video directory '{input_videos_directory}' does not exist.")
        print("Please create the directory and place your video files inside it, or correct the path.")
    else:
        extract_frames_from_videos(input_videos_directory, output_frames_directory, interval_seconds)
        print(f"\nExtraction complete! Check the '{output_frames_directory}/' directory.")