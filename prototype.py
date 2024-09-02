import cv2
import pyttsx3
import threading
import mediapipe as mp
import numpy as np
import time
import pygame

pygame.mixer.init()
pygame.mixer.music.load('clap1.mp3')

prev=None
curr=None 


rate=150
mp_drawing=mp.solutions.drawing_utils 
mp_pose=mp.solutions.pose
voice=pyttsx3.init()
voice.setProperty('rate', rate)


def speak(alert):
    voice.say(alert)
    voice.runAndWait()
#function to calculate angle
def calculate_angle(a,b,c=0):
    a=np.array(a)#first point
    b=np.array(b)#mid point
    c=np.array(c)#last point

    radians=np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)

  
    return angle

# reading video feed
cap=cv2.VideoCapture(0)
#setup mediapipe instance
ret,frame=cap.read() 
cv2.imshow('frame',frame)
elapsed_time=0
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
     ret,frame=cap.read()
     if elapsed_time >9:
        pygame.mixer.music.play()
        break
    #opencv feed by default is in bgr ,to convert to rgb-
     image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
     image.flags.writeable=False

    #makes detection
     results=pose.process(image)

    #restoring back to bgr format
     image.flags.writeable=True
     image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #extract landmarks 
     try:
         landmarks=results.pose_landmarks.landmark
        #get coordinates of shoulder,elbow and wrist
         Lshoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
         Lelbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
         Lwrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
         Lhip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
         Lknee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
         Lankle=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        
        
       
        #get coordinates of shoulder,elbow and wrist
         Rshoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
         Relbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
         Rwrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
         Rhip=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
         Rknee=[landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
         Rankle=[landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        #calculate angle between shodler,elbow and wrist
         angle1=(180-calculate_angle(Lshoulder,Lelbow).astype(int))
         print('angle1-',angle1)
        
        #print the angle on the screen
         cv2.putText(image,str(angle1),
                     tuple(np.multiply(Lshoulder,[850,480]).astype(int)),
                     cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
                     )
        #calculate angle between sholer,elbow and wrist
         angle2=calculate_angle(Rshoulder,Relbow).astype(int)
         print('angle2-',angle2)
        #print the angle on the screen
         cv2.putText(image,str(angle2),
                     tuple(np.multiply(Rshoulder,[850,480]).astype(int)),
                     cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
                     )
        

         angle3=calculate_angle(Lhip,Lknee).astype(int)
         print('angle3-',angle3)
        #print the angle on the screen
         cv2.putText(image,str(angle3),
                     tuple(np.multiply(Lankle,[850,480]).astype(int)),
                     cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
                     )
      
         
         angle4=90+calculate_angle(Rhip,Rknee).astype(int)
         print('angle4-',angle4)
        #print the angle on the screen
         cv2.putText(image,str(angle4),
                     tuple(np.multiply(Rknee,[850,480]).astype(int)),
                     cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
                     )
         cv2.putText(image, 'Well done! Hold for 10 seconds.', (24, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
         cv2.putText(image, f'Time: {elapsed_time:.2f} sec', (24, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
         
         if angle1>20 or angle2>20 :
             
             alert='please, raise your hands upwards'
             timer_started = False 
             prev=alert
             cv2.putText(image,alert,(24,25),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255),1,cv2.LINE_AA)
            #---
             if prev!=curr:
                threading.Thread(target=speak, args=(alert,)).start()
                curr=prev
            
            

         elif angle4>135 :
             curr=alert
             alert='please,bend your right knee.'
             timer_started = False
             prev=alert
             cv2.putText(image,alert,(24,25),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255),1,cv2.LINE_AA)
             if prev!=curr:
                threading.Thread(target=speak, args=(alert,)).start()

                curr=prev
            


         else :
            alert='perfect,you are doing great.'
            prev=alert
            if not timer_started:
                start_time = time.time()
                timer_started = True
            else:
                elapsed_time = time.time() - start_time
                if elapsed_time >= 10:
                    alert = 'Exercise complete!'
                    exercise_complete = True
                    timer_started = False
            if prev!=curr:
                threading.Thread(target=speak, args=(alert,)).start()

                curr=prev
            
    
            cv2.putText(image,alert,(24,25),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255),1,cv2.LINE_AA)
      
            if timer_started and not exercise_complete:
                elapsed_time = start_time
                
            elif exercise_complete:
                elapsed_time = time.time() - start_time

         cv2.putText(image, 'Well done! Hold for 10 seconds.', (24, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
         cv2.putText(image, f'Time: {elapsed_time:.2f} sec', (24, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)


  
     except Exception as e:
                print(f"Error: {e}")  
    #render detections
     mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                               mp_drawing.DrawingSpec(color=(255,255,255),thickness=4,circle_radius=3))

    
    #visualize the feed
     cv2.imshow('frame',image)
     
     #exit key q
     if elapsed_time >= 10 or cv2.waitKey(10) & 0xFF == ord('q'):
         break


#to release all the resource held(like camera)
cap.release()
cv2.destroyAllWindows()


