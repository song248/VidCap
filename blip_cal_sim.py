# BLIP 모델을 이용하여 이미지와 텍스트간 유사도(Similarity Score) 계산

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import json
import pandas as pd
from transformers import BlipProcessor, BlipForImageTextRetrieval
from tqdm import tqdm

processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

with open("prompt.json", "r") as f:
    prompt_data = json.load(f)

prompts = []
for event, event_data in prompt_data["PROMPT_CFG"]["event"].items():
    prompts.extend(event_data["prompt"])

# 영상에서 15프레임 단위로 이미지 추출
def extract_frames(video_path, frame_interval=15):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    # 전체 프레임 수 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
                frames.append(Image.fromarray(frame_rgb))  # PIL 이미지로 변환
            frame_count += 1
            pbar.update(1)  # 진행 표시줄 업데이트

    cap.release()
    return frames

# 이미지와 텍스트 간의 유사도 계산
def calculate_similarity(image, text):
    inputs = processor(image, text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.itm_score, dim=1)
    return probs[0][1].item()

video_path = "./video/Explosion002_x264.mp4"
print("Extracting frames from video...")
frames = extract_frames(video_path)

results = []
print("Calculating similarity between frames and prompts...")
for idx, frame in enumerate(tqdm(frames, desc="Processing Frames")):
    frame_results = {"frame": (idx + 1) * 15}  # 프레임 번호 (15, 30, 45, ...)
    for prompt in prompts:
        similarity = calculate_similarity(frame, prompt)
        frame_results[prompt] = similarity
    results.append(frame_results)

df = pd.DataFrame(results)
df.to_csv("output_similarity.csv", index=False)

print("Complete calculate similarity between frames and prompts.\n Result saved to 'output_similarity.csv'.\n")