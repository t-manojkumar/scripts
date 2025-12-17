import ffmpeg
import os
import re
from tqdm import tqdm

# GPU and CPU codecs
GPU_CODECS = ['h264_nvenc', 'hevc_nvenc']  # NVIDIA GPU
CPU_CODECS = ['libx264', 'libx265', 'mpeg4', 'vp8', 'vp9', 'av1']

def get_video_codec(video_file):
    """Get the video codec of the file."""
    probe = ffmpeg.probe(video_file)
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    return video_streams[0]['codec_name'] if video_streams else None

def convert_video(input_file, output_file, codec, preset='fast', crf=23):
    """Convert video with GPU/CPU codec and live progress bar."""
    try:
        # Get duration
        duration = float(ffmpeg.probe(input_file)['format']['duration'])
    except:
        print("Error reading video duration")
        return

    # Build FFmpeg command
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(
        stream,
        output_file,
        vcodec=codec,
        preset=preset,
        crf=crf,
        acodec='copy'  # copy audio stream for speed
    )

    # Run FFmpeg asynchronously, capture stderr for progress
    process = ffmpeg.run_async(stream, pipe_stdout=True, pipe_stderr=True, overwrite_output=True)

    # Live progress bar
    pbar = tqdm(total=int(duration), unit='s', ncols=80, desc='Converting')

    # Parse time from FFmpeg stderr
    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+\.\d+)')
    while True:
        line = process.stderr.readline()
        if not line:
            break
        line = line.decode('utf-8')
        match = time_pattern.search(line)
        if match:
            h, m, s = map(float, match.groups())
            elapsed_seconds = h * 3600 + m * 60 + s
            pbar.n = int(elapsed_seconds)
            pbar.refresh()

    process.wait()
    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()
    print("\nConversion complete!")

def main():
    # Input video
    video_file = input("Enter video path: ").strip().strip('"').replace('\\', '/')
    if not os.path.isfile(video_file):
        print("File does not exist")
        return

    # Detect current codec
    current_codec = get_video_codec(video_file)
    print(f"Current codec: {current_codec}")

    # List all codecs
    all_codecs = GPU_CODECS + CPU_CODECS
    print("\nAvailable conversion options:")
    for idx, c in enumerate(all_codecs, 1):
        print(f"{idx}. {c}")

    # User selects codec
    choice = input("Select codec number: ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(all_codecs):
        print("Invalid choice")
        return
    target_codec = all_codecs[int(choice)-1]

    # Automatic fallback for GPU encoders
    if target_codec in GPU_CODECS and current_codec not in ['h264', 'hevc']:
        print(f"GPU encoder '{target_codec}' may not support '{current_codec}' input. Falling back to 'libx264'.")
        target_codec = 'libx264'

    # Output file
    output_file = input("Enter output filename (with extension, e.g., output.mp4): ").strip().strip('"').replace('\\', '/')

    # Start conversion
    convert_video(video_file, output_file, target_codec)

if __name__ == "__main__":
    main()