import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

def has_static_segment(
    video_path,
    min_static_duration=5.0,  # 秒，静止时间阈值
    diff_threshold=3.0,  # 均值帧差小于该值就视为静止
) -> bool:
    """
    判断视频中是否存在长时间静止画面。

    参数：
        video_path (str): 视频路径
        min_static_duration (float): 静止段最小时长（秒）
        diff_threshold (float): 帧差小于该值判定为静止

    返回：
        has_static (bool): 是否存在静止段
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_static_frames = int(min_static_duration * fps)

    prev_gray = None
    consecutive_static_frames = 0

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = np.mean(diff)

            if mean_diff < diff_threshold:
                consecutive_static_frames += 1
                if consecutive_static_frames >= min_static_frames:
                    cap.release()
                    with open("logs/static_videos.log", "a") as f:
                        f.write(video_path + "\n")
                    return True
            else:
                consecutive_static_frames = 0

        prev_gray = gray

    cap.release()
    return False

video_folders = os.listdir(dir)
videos_to_be_verified = []
for video_folder in video_folders:
    video_path = os.path.join(dir, video_folder)
    if os.path.isdir(video_path):
        video_files = os.listdir(video_path)
        for video_file in video_files:
            video_file_path = os.path.join(video_path, video_file)
            videos_to_be_verified.append(video_file_path)

max_workers = 64
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    tqdm(executor.map(has_static_segment, videos_to_be_verified), total=len(videos_to_be_verified), desc="Verifying videos")