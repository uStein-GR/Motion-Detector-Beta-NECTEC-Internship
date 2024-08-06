import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import torch
from tracker import*

now = datetime.now()
text_color = (255, 255, 255)  # white color for text
red_color = (0, 0, 255)  # (B, G, R)   
blue_color = (255, 0, 0)  # (B, G, R)
green_color = (0, 255, 0)  # (B, G, R) 

area1 = [(470,0), (465,0), (465,720), (470,720)]
area2 = [(495,0), (490,0), (490,720), (495,720)]

count =0
count_left = {}
count_right = {}
left = set()
right = set()

# Load the YOLOv8 model
device = torch.device("cuda")
model = YOLO(r"Test\Model\yolov8n.pt").to(device)
# Read the image
# img = cv2.imread(r"Test\Picture\coin.jpg")

cap = cv2.VideoCapture(r'Test\Movie\trwalkway.mp4')

my_file = open(r"Test\Model\coco.txt")
data = my_file.read()
class_list = data.split("\n") 

tracker = Tracker()
while cap.isOpened():
    ret, frame = cap.read()
    # img = cv2.resize(frame, (1280, 720))
    img = cv2.resize(frame, (960, 640))
    # img = frame
    # Perform prediction
    results = model(img, conf = 0.5, imgsz=640, visualize=False)
    img0 = results[0].plot(labels = True, conf=True, boxes=True)

    result = results[0]
    a = results[0].boxes.data.cpu()
    print("---------------------------------------------------------------------------")
    px = pd.DataFrame(a.numpy()).astype('float')
    # px = pd.DataFrame(a.numpy()).astype('int')
    print(px)
    print("---------------------------------------------------------------------------")

    print("Detected amount: ", len(result.boxes))
    if len(result.boxes) > 0:
        # box = result.boxes[0]
        # class_id = box.cls[0].item()
        # cords = box.xyxy[0].tolist()
        # boxes = result.boxes.cpu().numpy()
        for box in result.boxes:
            class_name = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            u1,v1,u2,v2 = cords
            cv2.putText(img, f'{class_name}{conf}', (u1, v1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # print('conf', conf)
            # print("CORDS: ", cords)
            # print("Class Name:  ",class_name)

            list = []
            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                confidence = int(row[4])
                class_label = int(row[5])
                clist = class_list[class_label]
                # label = f'{index+1}: {class_name}{conf}'
                # print("class_label: ", class_label)

                # if 'person' in clist:
                if True:
                    list.append([x1, y1, x2, y2])
                    label = f'{index+1}: {class_name}{conf}'
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.putText(img, f'{class_name}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            bbox_id = tracker.update(list)
            # print("Bbox_id: ", bbox_id)
            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
                if results >= 0:
                    count_left[id] = (x4, y4)
                if id in count_left:
                    results1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
                    if results1 >= 0:
                        left.add(id)

                results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
                if results2 >= 0:
                    count_right[id] = (x4, y4)
                if id in count_right:
                    results3 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
                    if results3 >= 0:
                        right.add(id)  


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
    x = len(left)
    y = len(right)

    # cv2.polylines(img, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    # cv2.polylines(img, [np.array(area2, np.int32)], True, (0, 255, 0), 2)
    # # cv2.line(img, (630, 0), (630, 720), red_color, 2)
    # cv2.line(img, (650, 0), (650, 720), blue_color, 2)
    # cv2.putText(img, f"Count object: {count}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(img,"Count: :" + str(x),(10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2,)
    cv2.putText(img,"Count: :" + str(y),(10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2,)
    cv2.imshow("Output", img)
    cv2.imshow("Output0", img0)
    # cv2.imshow("Output_model", img0)
    # print("Count Number: ", count)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()

