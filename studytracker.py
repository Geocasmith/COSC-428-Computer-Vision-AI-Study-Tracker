import os
import numpy as np
import cv2
import time
import sys

#run with python studytracker.py
class StudyTracker(object):
    def study_toggle(self, *args):
        self.study_toggle_bool = not self.study_toggle_bool
        
    def exit(self, *args):
        total_time = time.time()-self.start_time
        print(f"You were distracted for {self.distracted_total_time:.2f} seconds out of {total_time:.2f} seconds")
        sys.exit()
        
    
    def __init__(self):
        self.study_toggle_bool = False
        # Open the capture
        directory = os.path.dirname(__file__)
        capture = cv2.VideoCapture(0) # camera
        if not capture.isOpened():
            exit()
    
        # Load the DNN model weights
        weights = os.path.join(directory, "yunet.onnx")
        face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
        
        distracted = False;
        previously_distracted=False
        start_study_session = False
        
        self.distracted_total_time=0
        self.start_time = time.time()
            
        
    
        #ctrl+p to show buttons
        cv2.namedWindow("Frame")
        cv2.createButton("Start Study",self.study_toggle,None,cv2.QT_PUSH_BUTTON,1)
        cv2.createButton("Exit",self.exit,None,cv2.QT_PUSH_BUTTON,1)
        
        #cv2.createButton("Start Study Session", start_session,None,cv2.QT_PUSH_BUTTON,1)
        #cv2.createButton("End Study Session", end_session,None,cv2.QT_PUSH_BUTTON,1)        
        #def back(*args):
            #print("Pressed")
        
        #cv2.createButton("Back",back,None,cv2.QT_PUSH_BUTTON,1)
    
        
        while cv2.waitKey(1) < 0:
            #timer
            current_time = time.time()
            
            # Capture frame and read image
            result, image = capture.read()
            if result is False:
                cv2.waitKey(0)
                break
    
            # If the image has less than 3 channels, it is converted to have 3 channels.
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            if channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
            # specify the input size
            height, width, _ = image.shape
            face_detector.setInputSize((width, height))
    
            # detect faces
            _, faces = face_detector.detect(image)
            faces = faces if faces is not None else []
    
            if faces == []:
                distracted = True
    
     
                    
                               
            #calculations for face 
            for face in faces:
                # bounding box
                box = list(map(int, face[:4]))
                width=box[2]
                height=box[3]
                
                color = (0, 0, 255)
                thickness = 2
                cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
            
                # check if head is turned
                landmarks = list(map(int, face[4:len(face)-1]))
                landmarks = np.array_split(landmarks, len(landmarks) / 2)
                left_eye_x, left_eye_y = landmarks[0]
                right_eye_x, right_eye_y = landmarks[1]
                nose_x, nose_y = landmarks[2]
                mouth_left_x, mouth_left_y = landmarks[3]
                mouth_right_x, mouth_right_y = landmarks[4]
                
                
            
                # calculate angle between eyes and horizontal axis
                dy = right_eye_y - left_eye_y
                dx = right_eye_x - left_eye_x
                angle = np.degrees(np.arctan2(dy, dx))
            
                # draw line between eyes
                color = (255, 0, 0)
                thickness = 2
                cv2.line(image, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y), color, thickness)
            
                # check if head is turned away from camera
                threshold = 15
                if abs(angle) > threshold:
                    distracted=True                
                    print("Head turned away from camera!")
                    #print("distracted")                
                else:
                    distracted=False
                    
                    
                # calculate eye-nose distance
                eye_nose_dist = np.sqrt((left_eye_x - right_eye_x)**2 + (left_eye_y - right_eye_y)**2 + (nose_x - (left_eye_x + right_eye_x) / 2)**2 + (nose_y - (left_eye_y + right_eye_y) / 2)**2)
                    
                # set thresholds for looking up or down
                up_threshold = 70
                down_threshold =81
                    
                # check if looking up or down
                #if eye_nose_dist < up_threshold:                
                    #distracted=True
                    #print("Looking up")
                if(width>height*0.95):
                    distracted=True
                    #print("Looking down")
                    
                    
                #if eye_nose_dist > down_threshold:
                    #distracted=True
                    #print("Looking down")
                else:
                    #print("Looking straight")                
                    distracted=False                
            
                # Degree of reliability
                confidence = face[-1]
                confidence = "{:.2f}".format(confidence)
                position = (box[0], box[1] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 2
                cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)
                
            if self.study_toggle_bool:
                if distracted:
                    if not previously_distracted:
                        distracted_start_time=time.time()
                        previously_distracted=True
                else:
                    if previously_distracted:
                        previously_distracted=False
                        self.distracted_total_time=self.distracted_total_time+(current_time-distracted_start_time)
                        print(self.distracted_total_time)
    
            # Display the image
            cv2.imshow("face detection", image)
            
            
        
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    StudyTracker()
