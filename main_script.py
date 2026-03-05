import torch
from ultralytics import YOLO
import cv2
import CFG

cap = cv2.VideoCapture(index=0)
model = YOLO(CFG.MODEL_PATH)
current_img = ''
label = ''
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, show=False, conf=0.5, device='cpu')
    plot = results[0].plot()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
    cv2.imshow('Emoji Detector', plot)

    if label != current_img:
        current_img = label
        emoji_path = CFG.emoji_to_path[label]
        if emoji_path:
            emoji_img = cv2.imread(emoji_path)
            emoji_img = cv2.resize(emoji_img, (CFG.emoji_resolution['x'], CFG.emoji_resolution['y']))
            cv2.imshow("Emoji", emoji_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
