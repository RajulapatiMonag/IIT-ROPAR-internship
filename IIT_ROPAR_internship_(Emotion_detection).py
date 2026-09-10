import os
import cv2
import random
import librosa
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    ViTModel,
    ViTImageProcessor,
    Wav2Vec2Model,
    AutoFeatureExtractor
)

# ---- EDIT THESE ----
DATASET_ROOT = r"/home/cvprlab/Monag/RAVDESS"   # <-- change to your local RAVDESS folder
CHECKPOINT_DIR = "./checkpoints"
NUM_FRAMES = 4                     # frames sampled per video clip
EPOCHS = 10
BATCH_SIZE = 4                     # raise this if your GPU has enough VRAM (8/16 etc.)
LR = 1e-5
NUM_WORKERS = 0                    # 0 = no multiprocessing workers, avoids Windows spawn issues
USE_AMP = True                     # mixed precision — big speedup on most NVIDIA GPUs, safe to leave True
# ---------------------

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

label2id = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7
}

id2label = {v: k for k, v in label2id.items()}

# ---- Feature extractors / processors ----
# These are loaded unconditionally at module level (not inside the
# __main__ guard) because Windows DataLoader worker processes re-import
# this module and need these objects to exist in their own namespace too.
image_processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

audio_processor = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base"
)


def augment_frame(frame):
    """Light augmentation for a single RGB video frame (HxWx3 uint8)."""
    # random horizontal flip
    if random.random() < 0.5:
        frame = cv2.flip(frame, 1)

    # random brightness/contrast jitter
    alpha = random.uniform(0.85, 1.15)   # contrast
    beta = random.uniform(-15, 15)       # brightness
    frame = np.clip(alpha * frame.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # small random rotation to simulate head tilt
    angle = random.uniform(-10, 10)
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return frame


def augment_audio(audio):
    """Light augmentation for a raw audio waveform (1D float32 array)."""
    # light additive gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.005, audio.shape)
        audio = audio + noise

    # random gain
    gain = random.uniform(0.8, 1.2)
    audio = audio * gain

    # random time shift (wrap-around), up to 0.1s at 16kHz
    shift = random.randint(-1600, 1600)
    audio = np.roll(audio, shift)

    return audio.astype(np.float32)


class RAVDESSDataset(Dataset):

    def __init__(self, dataframe, num_frames=NUM_FRAMES, train=False):
        self.df = dataframe.reset_index(drop=True)
        self.num_frames = num_frames
        self.train = train   # only apply augmentation to the training set

    def __len__(self):
        return len(self.df)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            frames = [blank] * self.num_frames
        else:
            # evenly spaced frame indices across the clip
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
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

        if self.train:
            frames = [augment_frame(f) for f in frames]

        processed = image_processor(images=frames, return_tensors="pt")
        # shape: (num_frames, 3, 224, 224)
        return processed["pixel_values"]

    def extract_audio(self, video_path):
        audio, sr = librosa.load(video_path, sr=16000)

        max_length = 16000 * 4  # 4 seconds

        if len(audio) < max_length:
            pad = max_length - len(audio)
            audio = np.pad(audio, (0, pad))
        else:
            audio = audio[:max_length]

        if self.train:
            audio = augment_audio(audio)

        processed = audio_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        return processed["input_values"][0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video = self.extract_frames(row["path"])   # (num_frames, 3, 224, 224)
        audio = self.extract_audio(row["path"])

        label = torch.tensor(row["label"], dtype=torch.long)

        return video, audio, label


def collate_fn(batch):
    videos = torch.stack([item[0] for item in batch])   # (B, num_frames, 3, 224, 224)
    audios = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    return videos, audios, labels


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

        frame_features = self.vit(pixel_values=video).pooler_output   # (B*T, 768)
        frame_features = frame_features.view(B, T, -1)
        video_features = frame_features.mean(dim=1)                   # temporal average pooling -> (B, 768)

        audio_features = self.wav2vec(audio).last_hidden_state.mean(dim=1)   # (B, 768)

        fused = torch.cat([video_features, audio_features], dim=1)
        output = self.classifier(fused)

        return output


# ==========================================================================
# Everything below actually DOES work (builds the dataset, trains, evaluates,
# saves). It's guarded by `if __name__ == "__main__":` because this script
# uses num_workers > 0 in its DataLoaders. On Windows, worker processes are
# created via `spawn`, which re-imports this file — without this guard, each
# worker would try to re-run the whole script and spawn its own workers,
# recursively, which is exactly the RuntimeError you hit.
# ==========================================================================
if __name__ == "__main__":

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- disk space check ----
    # Each epoch now saves one weights-only checkpoint (~350-400MB for this
    # model), plus occasionally overwriting best_model.pth. Warn early if
    # the drive looks too full to comfortably finish training.
    import shutil
    disk_usage = shutil.disk_usage(os.path.abspath(CHECKPOINT_DIR))
    free_gb = disk_usage.free / (1024 ** 3)
    print(f"Free disk space at checkpoint location: {free_gb:.2f} GB")
    estimated_needed_gb = 0.4 * EPOCHS + 0.4  # rough estimate: per-epoch + best_model
    if free_gb < estimated_needed_gb:
        print(f"WARNING: free disk space ({free_gb:.2f} GB) may not be enough for "
              f"{EPOCHS} epoch checkpoints (~{estimated_needed_gb:.2f} GB estimated). "
              "Consider freeing up space, or checkpoints may fail partway through training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    else:
        print("WARNING: No CUDA GPU detected — training will run on CPU and be very slow.")
        print("Check that: (1) you have an NVIDIA GPU, (2) drivers are installed (nvidia-smi works),")
        print("(3) you installed the CUDA build of torch, not the CPU-only build.")

    video_files = glob(
        os.path.join(DATASET_ROOT, "**", "*.mp4"),
        recursive=True
    )

    print("Total videos found:", len(video_files))
    if len(video_files) == 0:
        raise FileNotFoundError(
            f"No .mp4 files found under {DATASET_ROOT}. "
            "Double check DATASET_ROOT in the config cell above."
        )

    data = []
    for path in video_files:
        filename = os.path.basename(path)
        parts = filename.replace(".mp4", "").split("-")
        emotion_code = parts[2]
        actor_id = int(parts[6])   # RAVDESS filename: ...-actor.mp4, actor 01-24
        gender = "male" if actor_id % 2 == 1 else "female"   # odd = male, even = female
        emotion = emotion_map[emotion_code]
        data.append([path, label2id[emotion], actor_id, gender])

    df = pd.DataFrame(data, columns=["path", "label", "actor_id", "gender"])
    print(df.head())

    male_actors = sorted(df.loc[df["gender"] == "male", "actor_id"].unique())
    female_actors = sorted(df.loc[df["gender"] == "female", "actor_id"].unique())
    print("\nMale actors:", male_actors)
    print("Female actors:", female_actors)

    # ---- Actor-level split ----
    # Whole actors are assigned to train or val (never split across both), so the
    # model is evaluated on speakers it has never seen. 3 of 12 actors per gender
    # (25%) go to validation.
    N_VAL_ACTORS_PER_GENDER = 3

    rng = np.random.RandomState(42)

    male_actors_shuffled = rng.permutation(male_actors)
    female_actors_shuffled = rng.permutation(female_actors)

    male_val_actors = set(male_actors_shuffled[:N_VAL_ACTORS_PER_GENDER])
    male_train_actors = set(male_actors_shuffled[N_VAL_ACTORS_PER_GENDER:])

    female_val_actors = set(female_actors_shuffled[:N_VAL_ACTORS_PER_GENDER])
    female_train_actors = set(female_actors_shuffled[N_VAL_ACTORS_PER_GENDER:])

    train_actors = male_train_actors | female_train_actors
    val_actors = male_val_actors | female_val_actors

    print("\nMale actors -> train:", sorted(male_train_actors), "| val:", sorted(male_val_actors))
    print("Female actors -> train:", sorted(female_train_actors), "| val:", sorted(female_val_actors))

    train_df = df[df["actor_id"].isin(train_actors)].sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = df[df["actor_id"].isin(val_actors)].sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nTrain:", len(train_df), "| Male:", (train_df["gender"] == "male").sum(), "| Female:", (train_df["gender"] == "female").sum())
    print("Validation:", len(val_df), "| Male:", (val_df["gender"] == "male").sum(), "| Female:", (val_df["gender"] == "female").sum())

    train_dataset = RAVDESSDataset(train_df, train=True)
    val_dataset = RAVDESSDataset(val_df, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    model = MultiModalEmotionModel().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-2   # L2-style regularization to reduce overfitting
    )

    scaler = GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        # ---- train ----
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader)

        for videos, audios, labels in loop:
            videos = videos.to(device, non_blocking=True)
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=(USE_AMP and device.type == "cuda")):
                outputs = model(videos, audios)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---- validation ----
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for videos, audios, labels in val_loader:
                videos = videos.to(device, non_blocking=True)
                audios = audios.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type=device.type, enabled=(USE_AMP and device.type == "cuda")):
                    outputs = model(videos, audios)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ---- save a checkpoint every epoch (weights only, kept small) ----
        # Wrapped in try/except so a disk-write failure (e.g. drive full,
        # antivirus lock) prints a warning and lets training continue,
        # instead of crashing the whole run.
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        except (RuntimeError, OSError) as e:
            print(f"WARNING: failed to save checkpoint for epoch {epoch+1}: {e}")
            print("This is usually caused by low disk space on the drive holding CHECKPOINT_DIR.")
            print("Training will continue; free up disk space if you want checkpoints saved.")

        # ---- separately track the single best checkpoint by val loss ----
        # Keeping only one "best_model.pth" (overwritten, not accumulated)
        # avoids filling the disk with many large near-duplicate files.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"New best val loss ({best_val_loss:.4f}) — saved: {best_model_path}")
            except (RuntimeError, OSError) as e:
                print(f"WARNING: failed to save best model: {e}")

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curve.png"), dpi=150)
    plt.show()

    print("Final Train Loss:", train_losses[-1])
    print("Final Val Loss:", val_losses[-1])

    # ---- load the best checkpoint (lowest val loss) for final evaluation ----
    # The last epoch is often more overfit than the best epoch, so evaluation
    # and the final saved model use best_model.pth instead of the final weights.
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best checkpoint (val loss {best_val_loss:.4f}) for final evaluation.")
    else:
        print("WARNING: best_model.pth not found, evaluating with final-epoch weights instead.")

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, audios, labels in tqdm(val_loader, desc="Evaluating"):
            videos = videos.to(device, non_blocking=True)
            audios = audios.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=(USE_AMP and device.type == "cuda")):
                outputs = model(videos, audios)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    target_names = [id2label[i] for i in range(len(id2label))]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(id2label))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accuracy: {accuracy*100:.2f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "confusion_matrix.png"), dpi=150)
    plt.show()

    model_path = os.path.join(CHECKPOINT_DIR, "emotion_model.pth")
    torch.save(model.state_dict(), model_path)
    print("Final model (best val loss checkpoint) saved to:", model_path)
