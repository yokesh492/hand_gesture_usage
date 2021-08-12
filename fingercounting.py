import cv2
import HandTrackingModule as hmt
import time
import os

folderpath = 'FingerImage'
mylist = os.listdir(folderpath)
print(mylist)
overlaylist = []
for imgpath in mylist:
    image = cv2.imread(f'{folderpath}\{imgpath}')
    overlaylist.append(image)



wcam, hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wcam) # alram system can needs to intiate

cap.set(4, hcam)
ptime =0
detector = hmt.HandDetector(detect_confidence=0.75)
tipids = [4, 8, 12, 16, 20] #tips of the fingers landmarks
while True:
    success, img = cap.read()
    img = detector.hand_detector(img)
    landmarkslist = detector.position_detector(img,draw=False)
    #print(landmarkslist)
    #finding the fingercounting
    fingers = []
    if len(landmarkslist)!=0:
        for i in range(0,5):
            #thumb for it is diffrent from other fingerns folding methods
            if i == 0:
                if landmarkslist[tipids[i]][1] > landmarkslist[tipids[i]-1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # for other four fingers it is diffrent folding methods
                if landmarkslist[tipids[i]][2] < landmarkslist[tipids[i]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        #print(fingers) end of coding
        totalfingers = fingers.count(1)
        #print(totalfingers)
        h, w, c = overlaylist[totalfingers-1].shape
        img[0:h, 0:w] = overlaylist[totalfingers-1]

        #cv2.rectangle(img,(50,225),(170,145),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalfingers),(30,400),cv2.FONT_HERSHEY_COMPLEX_SMALL,10,(255,0,0),25)

    ctime = time.time()
    fps = 1/ (ctime - ptime)
    ptime=ctime
    cv2.putText(img,f'FPS:{int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(0,244,243),1)

    cv2.imshow('image',img)
    cv2.waitKey(1)

