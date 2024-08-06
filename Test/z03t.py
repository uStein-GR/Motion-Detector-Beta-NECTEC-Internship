import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import torch
from tracker import*

now = datetime.now()

# Load the YOLOv8 model
device = torch.device("cuda")
model = YOLO(r"Test\Model\yolov8n.pt").to(device)
# Read the image
# img = cv2.imread(r"Test\Picture\coin.jpg")

cap = cv2.VideoCapture(0)

list = []
counting = {}
num_count = set()
count =0


my_file = open(r"Test\Model\coco.txt")
data = my_file.read()
class_list = data.split("\n") 

tracker = Tracker()
while cap.isOpened():
    ret, frame = cap.read()
    # img = cv2.resize(frame, (1280, 720))
    img = frame
    # Perform prediction
    results = model(img, conf = 0.7, imgsz=640, visualize=False)
    img0 = results[0].plot(labels = True, conf=True, boxes=True)

    result = results[0]
    a = results[0].boxes.data.cpu()
    # px = pd.DataFrame(a).astype('float')
    px = pd.DataFrame(a.numpy()).astype('int')
    # print("PX: ",px)

    print("Length Resutl Box: ", len(result.boxes))
    if len(result.boxes) > 0:
        box = result.boxes[0]
        class_id = box.cls[0].item()
        cords = box.xyxy[0].tolist()
        boxes = result.boxes.cpu().numpy()
        count = len(boxes)
        # print("count: ", count)
        for box in result.boxes:
            # class_id = result.names[box.cls[0].item()]
            class_name = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            print('conf', conf)
            print("CORDS: ", cords)
            print("Class Name:  ",class_name)
    
    for index, row in px.iterrows():
        i = 0
        x1 = row[0]
        y1 = row[1]
        x2 = row[2]
        y2 = row[3]
        confidence = row[4]
        class_label = row[5]
        label = f'{index+1}: {class_name}{conf}'
        clist = class_list[class_label]
        # print("Index: ", index+1)
        count = len(px)
        if 'person' in clist:
            list.append([x1, y1, x2, y2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'{label}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            bbox_id = tracker.update(list)
            for bbox in bbox_id:
                x3, y3, x4, y4, id=bbox
                counting[id] = (x4, y4)
                if id in counting:
                    # cv2.putText(img, str(id), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    num_count.add(id)
                    i += 1
                    print("i",i)
    else:
        pass
    # if len(result.boxes) > 0:
    #     box = result.boxes[0]
    #     class_id = box.cls[0].item()
    #     cords = box.xyxy[0].tolist()
    #     boxes = result.boxes.cpu().numpy()
    #     count = len(boxes)
    #     # print("count: ", count)
    #     for box in result.boxes:
    #         class_name = result.names[box.cls[0].item()]
    #         cords = box.xyxy[0].tolist()
    #         cords = [round(x) for x in cords]
    #         conf = round(box.conf[0].item(), 2)
    #         print('conf', conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Count object: {count}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Output", img)
    cv2.imshow("Output_model", img0)
    print("Number Counting: ", len(num_count))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()

