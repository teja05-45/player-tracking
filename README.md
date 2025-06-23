
# Player Tracking with Re-Identification

This project tracks soccer players in a video using a YOLOv11 object detection model and assigns consistent player IDs, even when players leave and re-enter the frame.

## 🛠 Requirements

- Python 3.8+
- torch
- torchvision
- ultralytics
- opencv-python
- scipy

Install dependencies:

```bash
pip install -r requirements.txt
python track_players.py

---

### 📄 `report.md` – Brief Report

```markdown
# Player Re-Identification Tracker – Report

## ✅ Objective
Track players in a short video and assign consistent IDs — even when players go out of frame and return.

## 🔧 Methodology

1. **YOLOv11**: Used to detect players in each frame.
2. **Appearance Embeddings**: A ResNet18 model was used to extract visual features from detected players.
3. **Re-ID Logic**:
   - Stored previous player embeddings.
   - Compared new detections using cosine similarity.
   - Reused the same ID if a close match was found.

## 🧪 Techniques Tried

- ✅ Basic DeepSORT — Failed to retain IDs after reappearance.
- ✅ Custom Re-ID via embeddings — Successful and robust.

## ❗ Challenges

- Players wearing similar jerseys can sometimes confuse embedding matching.
- Appearance matching can break under motion blur or occlusion.

## 🚧 What's Incomplete / Next Steps

- Replace ResNet18 with OSNet for stronger Re-ID.
- Use feature buffers instead of 1 embedding per player.
- Save match logs in JSON for analysis.

