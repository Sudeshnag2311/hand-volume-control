import cv2
import mediapipe as mp
import time
import pyautogui
import math
import numpy as np
pyautogui.FAILSAFE=False
cap=cv2.VideoCapture(0)


mphand=mp.solutions.hands
hand=mphand.Hands()
mpdraw=mp.solutions.drawing_utils
style=mpdraw.DrawingSpec(color=(0,255,0),thickness=2)
land=mpdraw.DrawingSpec(color=(0,0,255),thickness=2)
center=0.5
COOLDOWN = 0.6
last_action_time = 0
pre=None
def can_act():
    global last_action_time
    now = time.time()
    if now - last_action_time > COOLDOWN:
        last_action_time = now
        return True
    return False


while True:
    ret,img=cap.read()
    img=cv2.flip(img,1)
    h,w,_=img.shape
    imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hand.process(imgrgb)

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,handlms,mphand.HAND_CONNECTIONS,style,land)

            thumb_tip = handlms.landmark[4]
            index_tip = handlms.landmark[8]
           
            
            x1,y1=int(thumb_tip.x*w),int(thumb_tip.y*h)
            x2,y2=int(index_tip.x*w),int(index_tip.y*h)
            cx,cy=(x1+x2)//2,(y1+y2)//2
            dis=math.hypot(x2-x1,y2-y1)


            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.circle(img,(cx,cy),15,(255.0,255),cv2.FILLED)
            
            vol=np.interp(dis,[30,200],[0,100])
            vol=int(vol)
            if pre is None:
                pre=dis
                continue
            diff=dis-pre
            if can_act():
                if diff>8:
                    pyautogui.press('volumeup')
                elif diff <-8:
                    pyautogui.press('volumedown')
            pre=dis

    cv2.imshow("Hand Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

   
cap.release()
cv2.destroyAllWindows()