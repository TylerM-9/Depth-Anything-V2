import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# 1. Select device and encoder model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitb' # Use 'vits' or 'vitb' for real-time webcam
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# 2. Open live webcam (0) or video file
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run depth inference
    depth = model.infer_image(frame) # Returns HxW depth map

    # 4. Normalize and apply colormap for visualization
    depth_normalized = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    # 5. Display original frame and depth side-by-side or stacked
    combined = np.hstack((frame, depth_color))
    cv2.imshow('Live Feed | Depth Anything V2', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
