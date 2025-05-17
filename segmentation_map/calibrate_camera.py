import cv2

video_path = r'D:\grad project\YOLOP 21\data\input\test_video.mp4'
output_path = r'D:\grad project\YOLOP 21\frame_0010.png'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
target_frame = 10  # Extract the 10th frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    frame_count += 1
    if frame_count == target_frame:
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_count} saved as {output_path}")
        break

cap.release()