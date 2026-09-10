import os
import sys
import subprocess
import numpy as np
import cv2
import librosa

import torch
import torch.nn as nn
from torch.amp import autocast

from transformers import (
    ViTModel,
    ViTImageProcessor,
    Wav2Vec2Model,
    AutoFeatureExtractor
)

# ---- EDIT THESE ----
CHECKPOINT_PATH = r"./checkpoints/best_model.pth"   # <-- set to the .pth weights file you want to use
VIDEO_PATH = r"./TestFiles/Vid_01.mp4"              # <-- set to the unseen video you want to predict on
NUM_FRAMES = 4          # must match what the model was trained with
USE_AMP = True
# ---------------------

emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
label2id = {
    "neutral": 0, "calm": 1, "happy": 2, "sad": 3,
    "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7
}
id2label = {v: k for k, v in label2id.items()}


class MultiModalEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8)
        )

    def forward(self, video, audio):
        # video: (B, num_frames, 3, 224, 224)
        B, T, C, H, W = video.shape
        video = video.view(B * T, C, H, W)

        frame_features = self.vit(pixel_values=video).pooler_output
        frame_features = frame_features.view(B, T, -1)
        video_features = frame_features.mean(dim=1)

        audio_features = self.wav2vec(audio).last_hidden_state.mean(dim=1)

        fused = torch.cat([video_features, audio_features], dim=1)
        return self.classifier(fused)


def extract_frames(video_path, image_processor, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

    processed = image_processor(images=frames, return_tensors="pt")
    return processed["pixel_values"]   # (num_frames, 3, 224, 224)


def extract_audio(video_path, audio_processor, temp_audio="./_temp_infer_audio.wav"):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-y",
        temp_audio
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if result.returncode != 0 or not os.path.exists(temp_audio):
        raise RuntimeError(
            "ffmpeg failed to extract audio from the video. "
            "Make sure ffmpeg is installed and available on your system PATH.\n"
            f"ffmpeg stderr: {result.stderr.decode(errors='ignore')}"
        )

    audio, sr = librosa.load(temp_audio, sr=16000)

    max_length = 16000 * 4
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))
    else:
        audio = audio[:max_length]

    processed = audio_processor(audio, sampling_rate=16000, return_tensors="pt")

    # clean up temp file
    try:
        os.remove(temp_audio)
    except OSError:
        pass

    return processed["input_values"][0]


def predict(video_path, checkpoint_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading processors...")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    audio_processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    print("Building model...")
    model = MultiModalEmotionModel().to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    # handle both a raw state_dict and a dict wrapping "model_state_dict"
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint loaded successfully.")

    print(f"Processing video: {video_path}")
    video = extract_frames(video_path, image_processor).unsqueeze(0).to(device)   # (1, num_frames, 3, 224, 224)
    audio = extract_audio(video_path, audio_processor).unsqueeze(0).to(device)    # (1, seq_len)

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=(USE_AMP and device.type == "cuda")):
            outputs = model(video, audio)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_id = int(np.argmax(probs))

    print("\n--- Prediction ---")
    print(f"Predicted emotion: {id2label[pred_id]}")
    print("\nConfidence per emotion:")
    for i in range(len(id2label)):
        print(f"  {id2label[i]:<10s}: {probs[i]*100:.2f}%")

    return id2label[pred_id]


if __name__ == "__main__":
    # optional: allow overriding via command line
    #   python predict_emotion.py <checkpoint_path> <video_path>
    if len(sys.argv) == 3:
        ckpt = sys.argv[1]
        vid = sys.argv[2]
    else:
        ckpt = CHECKPOINT_PATH
        vid = VIDEO_PATH

    predict(vid, ckpt)
