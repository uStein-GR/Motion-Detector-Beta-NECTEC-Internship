import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
now = datetime.now()
# Load the YOLOv8 model
model = YOLO("runs/detect/train/weights/best_sack.pt")
# Read the image
img0 = cv2.imread("2024.6.17-20240618T074553Z-001/2024.6.17/1718595010299.jpg")
img = cv2.resize(img0, (600, 800))
# Perform prediction
results = model.predict(img, conf = 0.5, imgsz=640, visualize=False)
img = results[0].plot(labels = False, conf=True, boxes=False)
result = results[0] 
a = results[0].boxes.data.cpu()
# print('resul', a)
# px = pd.DataFrame(a).astype('float')
px = pd.DataFrame(a.numpy()).astype('int')
print(px) 
'''
0	X1 coordinate (top-left corner of the bounding box)
1	Y1 coordinate (top-left corner of the bounding box)
2	X2 coordinate (bottom-right corner of the bounding box)
3	Y2 coordinate (bottom-right corner of the bounding box)
4	Confidence score of the detection
5	Class label of the detected object
'''
count =0
if len(result.boxes) > 0:
    box = result.boxes[0]
    class_id = box.cls[0].item()
    cords = box.xyxy[0].tolist()
    boxes = result.boxes.cpu().numpy()
    count = len(boxes)
    print("count: ", count)
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        print('conf', conf)
for index, row in px.iterrows():
    x1 = row[0]
    y1 = row[1]
    x2 = row[2]
    y2 = row[3]
    confidence = row[4]
    class_label = row[5]
    label = f'{index+1}: {class_id}{conf}'
    count = len(px)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, f'{label}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

if len(result.boxes) > 0:
    box = result.boxes[0]
    class_id = box.cls[0].item()
    cords = box.xyxy[0].tolist()
    boxes = result.boxes.cpu().numpy()
    count = len(boxes)
    print("count: ", count)
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        print('conf', conf)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, f"sack: {count}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("img", img)
id_text =  datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
surname = '.png'
filename = '/cap'
path = "capture_out" #บันทึกภาพใน floder
# cv2.imwrite(path+filename+id_text+surname, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
