import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
import time


def extract_frame_from_video(video_path, frame_idx, output_folder, gpu_enabled=False):
    """
    Extracts a single frame from a video at the given index.

    Args:
    - video_path (str): Path to the input video file.
    - frame_idx (int): Index of the frame to be extracted.
    - output_folder (str): Folder to save the extracted frame.
    - gpu_enabled (bool): Flag to indicate if GPU processing is enabled.

    Returns:
    - frame (numpy array): The extracted frame.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Set the frame position
    ret, frame = cap.read()

    if not ret:
        cap.release()
        return None

    # Convert the frame to a GPU array if GPU is enabled
    if gpu_enabled:
        frame_gpu = cp.asarray(frame)  # Convert to CuPy array (GPU)
        frame = cp.asnumpy(frame_gpu)  # Convert back to NumPy array (CPU)

    # Save the frame to disk
    output_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
    cv2.imwrite(output_path, frame)
    cap.release()

    return frame


def process_video(video_path, output_folder, gpu_enabled=False, max_workers=8):
    """
    Processes a video by extracting all frames in parallel.

    Args:
    - video_path (str): Path to the video file.
    - output_folder (str): Directory to save the extracted frames.
    - gpu_enabled (bool): Flag to indicate if GPU processing is enabled.
    - max_workers (int): Maximum number of threads to use for parallelism.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video to get total number of frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Total frames in video: {total_frames}")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Extract frames in parallel
        futures = []
        for i in range(total_frames):
            futures.append(executor.submit(extract_frame_from_video, video_path, i, output_folder, gpu_enabled))

        # Wait for all futures to complete
        for future in futures:
            result = future.result()  # You can handle the result if needed

    print(f"All frames have been extracted to {output_folder}")


if __name__ == "__main__":
    video_path = 'input_video.mp4'  # Path to your input video
    output_folder = 'extracted_frames'  # Output folder where frames will be saved
    gpu_enabled = True  # Set to True if you want to use GPU for processing
    max_workers = 8  # Number of threads for parallel processing

    start_time = time.time()
    process_video(video_path, output_folder, gpu_enabled, max_workers)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
