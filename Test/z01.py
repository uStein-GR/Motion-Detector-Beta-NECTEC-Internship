#ตรวจจับดวงตาและใบหน้า
import cv2 
import torch
from ultralytics import YOLO
import pandas
cap = cv2.VideoCapture(0)

model = YOLO(r'Test\Model\yolov8n.pt')

while cap.isOpened():
    ret, frame = cap.read()
    # img = frame
    img = cv2.resize(frame, (800, 600))
    results = model(source=img, conf=0.4)

    img = results[0].plot(conf=True, line_width=None, font_size=None, font='Arial.ttf', 
        pil=False, img=None, im_gpu=None, kpt_radius=5, kpt_line=True, labels=True,
        boxes=True, masks=True, probs=True, show=False, save=False, filename=None) #ซ้อน labels และ กรอบ    result = results[0]
    result = results[0]
    print("Result", result)
    if len(result.boxes) > 0:
        box = result.boxes[0]
        cords = box.xyxy[0].tolist()
        class_id = box.cls[0].item()
        conf = box.conf[0].item()
        cords = box.xyxy[0].tolist()

        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            # print("Object type:", class_id)
            # print("Coordinates:", cords)
            # print("Probability:", conf)
            # print("---")
    else:
        print("No detected")

    cv2.imshow('Boxes1', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
