

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from ultralytics import YOLO
from scipy.spatial.distance import cosine

# Load detection model
model = YOLO("best.pt")

# Appearance feature extractor (ResNet18)
resnet = resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor()
])

player_db = {}
next_player_id = 1
REID_THRESHOLD = 0.3
MAX_INACTIVE_FRAMES = 30

def get_similarity(emb1, emb2):
    return 1 - cosine(emb1, emb2)

def extract_embedding(crop):
    with torch.no_grad():
        tensor = transform(crop).unsqueeze(0)
        emb = resnet(tensor).squeeze().numpy()
        return emb / np.linalg.norm(emb)

def identify_player(new_emb, current_frame):
    global next_player_id
    for pid, (emb, last_seen) in player_db.items():
        if current_frame - last_seen > MAX_INACTIVE_FRAMES:
            continue
        if get_similarity(new_emb, emb) > 1 - REID_THRESHOLD:
            return pid
    pid = next_player_id
    next_player_id += 1
    return pid

cap = cv2.VideoCapture("15sec_input_720p.mp4")
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("final_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    results = model(frame)[0]
    players = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] != 'player':
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        emb = extract_embedding(crop)
        pid = identify_player(emb, frame_num)
        player_db[pid] = (emb, frame_num)
        players.append((pid, x1, y1, x2, y2))

    for pid, x1, y1, x2, y2 in players:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
