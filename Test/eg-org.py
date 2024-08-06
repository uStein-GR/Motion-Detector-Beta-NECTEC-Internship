import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import torch

device = torch.device("cuda")
model = YOLO(r"Test\Model\yolov8n.pt").to(device)


cap = cv2.VideoCapture(r'Test\Movie\trwalkway.mp4')


my_file = open(r"Test\Model\coco.txt")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()

cy1=322
cy2=368
offset=6

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data.cpu()
    px = pd.DataFrame(a.numpy()).astype('float')
    print("---------------------------------------------------------------------------")
    print(px)
    print("---------------------------------------------------------------------------")
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
        cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)
        cv2.rectangle(frame,(x3, y3), (x4, y4), (0, 0, 255), 1)
           


#    cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1)
#    cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()