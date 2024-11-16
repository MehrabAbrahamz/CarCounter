import cv2 , cvzone , math; from ultralytics import YOLO ; from sort import * ; import numpy as np
# Video
cap = cv2.VideoCapture("CarCounter1/Car1.mp4")
cv2.namedWindow("frame" , cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame" , cv2.WND_PROP_FULLSCREEN , cv2.WINDOW_FULLSCREEN)
# Mask
LeftMask = cv2.imread("CarCounter1/Left.png")
RightMask =  cv2.imread("CarCounter1/Right.png")
# Model
model = YOLO("Yolo/yolov8n.pt" , verbose = False)
ClassNames = model.names.items()
Names = []
for i in ClassNames:
    Names.append(list(i).pop())
# Tracker
LeftTracker = Sort(max_age = 60 , min_hits = 3 , iou_threshold = 0.3)
RightTracker = Sort(max_age = 60 , min_hits = 3 , iou_threshold = 0.3)
LeftTracked = []
RightTracked = []
Leftcounted = []
Rightcounted = []
# Detection
while True:
    ret , frame = cap.read()
    LeftCrop = cv2.bitwise_and(frame , LeftMask)
    RightCrop = cv2.bitwise_and(frame , RightMask)
    Leftresults = model(LeftCrop , stream = True)
    Rightresults = model(RightCrop , stream = True)
    LeftDetections = np.empty((0 , 5))
    RightDetections = np.empty((0 , 5))
    # Left Box
    for Leftrect in Leftresults:
        Leftboxes = Leftrect.boxes
    for Leftbox in Leftboxes:
        Leftx1 , Lefty1 , Leftx2 , Lefty2 = Leftbox.xyxy[0]
        Leftx1 , Lefty1 , Leftx2 , Lefty2 = int(Leftx1) , int(Lefty1) , int(Leftx2) , int(Lefty2)
        # conf
        Leftconf = math.ceil((Leftbox.conf)*100)/100
        # cls
        Leftcls = int(Leftbox.cls)
        # Tracker Values
        LeftTrackerBox = np.array([Leftx1 , Lefty1 , Leftx2 , Lefty2 , Leftconf])
        LeftDetections = np.vstack((LeftDetections , LeftTrackerBox))
    # Right Box
    for Rightrect in Rightresults:
        Rightboxes = Rightrect.boxes
    for Rightbox in Rightboxes:
        Rightx1 , Righty1 , Rightx2 , Righty2 = Rightbox.xyxy[0]
        Rightx1 , Righty1 , Rightx2 , Righty2 = int(Rightx1) , int(Righty1) , int(Rightx2) , int(Righty2)
        # conf
        Rightconf = math.ceil((Rightbox.conf)*100)/100
        # cls
        Rightcls = int(Rightbox.cls)
        # Tracker Values
        RightTrackerBox = np.array([Rightx1 , Righty1 , Rightx2 , Righty2 , Rightconf])
        RightDetections = np.vstack((RightDetections , RightTrackerBox))
    # Line
    LeftLimits = [100 , 1000 , 1000 , 1000]
    RightLimits = [1060 , 660 , 1310 , 660]
    cv2.line(frame , (LeftLimits[0] , LeftLimits[1]) , (LeftLimits[2] , LeftLimits[3]) , (0 , 0 , 255) , 2)
    cv2.line(frame , (RightLimits[0] , RightLimits[1]) , (RightLimits[2] , RightLimits[3]) , (0 , 0 , 255) , 2)
    # Detections update
    Leftupdates = LeftTracker.update(LeftDetections)
    Rightupdates = RightTracker.update(RightDetections)
    # Left Detection rect
    for Leftupdate in Leftupdates:
        Leftx1 , Lefty1 , Leftx2 , Lefty2 , LeftID = Leftupdate
        Leftx1 , Lefty1 , Leftx2 , Lefty2 , LeftID = int(Leftx1) , int(Lefty1) , int(Leftx2) , int(Lefty2) , int(LeftID)
        Leftcx , Leftcy = Leftx1+(Leftx2-Leftx1)//2 , Lefty1+(Lefty2-Lefty1)//2
        if Names[Leftcls] == "car":
            cvzone.cornerRect(frame , (Leftx1 , Lefty1 , Leftx2-Leftx1 , Lefty2-Lefty1) , l = 1 , rt = 1 , colorR = (0 , 0 , 0) , colorC=(0 , 0 , 0))
            cv2.circle(frame , (Leftcx , Leftcy) , radius=1 , color = (0 , 0 , 0) , thickness = cv2.FILLED)
            if LeftLimits[0]<Leftcx<LeftLimits[2] and LeftLimits[1]-10<Leftcy<LeftLimits[3]+10:
                cv2.line(frame , (LeftLimits[0] , LeftLimits[1]) , (LeftLimits[2] , LeftLimits[3]) , (0 , 255 , 0) , 2)
                if Leftcounted.count(LeftID) == 0:
                    Leftcounted.append(LeftID)
        cvzone.putTextRect(frame , f"Left: {len(Leftcounted)}" , (20 , 30) , scale = 2 , thickness = 2)
    # Right Detection rect
    for Rightupdate in Rightupdates:
        Rightx1 , Righty1 , Rightx2 , Righty2 , RightID = Rightupdate
        Rightx1 , Righty1 , Rightx2 , Righty2 , RightID = int(Rightx1) , int(Righty1) , int(Rightx2) , int(Righty2) , int(RightID)
        Rightcx , Rightcy = Rightx1+(Rightx2-Rightx1)//2 , Righty1+(Righty2-Righty1)//2
        if Names[Rightcls] == "car":
            cvzone.cornerRect(frame , (Rightx1 , Righty1 , Rightx2-Rightx1 , Righty2-Righty1) , l = 1 , rt = 1 , colorR = (255 , 255 , 255) , colorC = (255 , 255 , 255))
            cv2.circle(frame , (Rightcx , Rightcy) , radius=1 , color = (255 , 255 , 255) , thickness = cv2.FILLED)
            if RightLimits[0]<Rightcx<RightLimits[2] and RightLimits[1]-10<Rightcy<RightLimits[3]+10:
                cv2.line(frame , (RightLimits[0] , RightLimits[1]) , (RightLimits[2] , RightLimits[3]) , (0 , 255 , 0) , 2)
                if Rightcounted.count(RightID) == 0:
                    Rightcounted.append(RightID)
    cvzone.putTextRect(frame , f"{len(Rightcounted)} :Right" , (1770 , 30) , scale = 2 , thickness = 2)   
    # show
    cv2.imshow("frame" , frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()