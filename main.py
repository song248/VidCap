import cv2
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm

model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

video_path = "./video/video_name.mp4"
output_json_path = "./caption_json/output.json"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"동영상을 열 수 없습니다: {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = 15  # 15프레임마다 이미지 추출
frame_count = 0
captions = {}

with tqdm(total=total_frames, desc="Captioning") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(pil_image, return_tensors="pt")

            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            captions[f"{frame_count // frame_interval * frame_interval}fps"] = caption
        frame_count += 1
        pbar.update(1)
cap.release()

output_data = {
    "filename": video_path,
    "captions": captions
}

with open(output_json_path, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"\nResult save to '{output_json_path}'")