import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self,static_img=False,max_hands=2,detect_confidence=0.5,track_confidence=0.5):
        self.static_img = static_img
        self.max_hands = max_hands
        self.detect_confidence = detect_confidence
        self.track_confidence = track_confidence
        self.mpHand = mp.solutions.hands
        self.hand = self.mpHand.Hands(self.static_img,self.max_hands,self.detect_confidence,self.track_confidence)
        self.mpdraw = mp.solutions.drawing_utils

    def hand_detector(self,img,draw = True):#this is used to draw and the hand tracking
        rgbimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.process = self.hand.process(rgbimg)
        #print(process.multi_hand_landmarks)

        if self.process.multi_hand_landmarks:
            for handlmks in self.process.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlmks, self.mpHand.HAND_CONNECTIONS)
        return img

    def position_detector(self, img, handno=0, draw=False):
        lmslist=[]
        if self.process.multi_hand_landmarks:
            myhand = self.process.multi_hand_landmarks[handno]
            for id,lms in enumerate(myhand.landmark):
                #print(id,lms)
                h,w,c = img.shape
                cx,cy = int(lms.x*w),int(lms.y*h)# getting the posistion from points of landmark(x,y,z)
                #print(id,cx,cy)
                lmslist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmslist


def main():
    cap = cv2.VideoCapture(0)
    ctime = 0
    ptime = 0
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.hand_detector(img)
        landmrk_list = detector.position_detector(img)
        if len(landmrk_list) !=0:
            print(landmrk_list[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 210), 3)

        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()