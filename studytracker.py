import os
import numpy as np
import cv2
import time


def main():
    # Open the capture
    directory = os.path.dirname(__file__)
    #capture = cv2.VideoCapture(os.path.join(directory, "image.jpg")) # image file
    capture = cv2.VideoCapture(0) # camera
    if not capture.isOpened():
        exit()

    # Load the model
    weights = os.path.join(directory, "yunet.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
    distracted = False;
    
    start_time = time.time()
    
    # Initialize variables for timer
    start_time_visible = None
    start_time_not_visible = None
    face_visible = False
    elapsed_time_visible = 0
    elapsed_time_not_visible = 0

    while True:
        #timer
        current_time = time.time()
        elapsed_time = current_time - start_time        
        
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
            if face_visible:
                face_visible = False
                end_time_visible = time.time()
                elapsed_time_visible += end_time_visible - start_time_visible
                print(f"Face visible for {elapsed_time_visible:.2f} seconds")
            if start_time_not_visible is None:
                start_time_not_visible = start_face_not_visible_timer()
        else:
            if not face_visible:
                face_visible = True
                start_time_visible = time.time()
                if start_time_not_visible:
                    elapsed_time_not_visible += time.time() - start_time_not_visible
                    print(f"Face not visible for {elapsed_time_not_visible:.2f} seconds")
                    start_time_not_visible = None
            if start_time_visible:
                elapsed_time_visible = time.time() - start_time_visible
                print(f"Face visible for {elapsed_time_visible:.2f} seconds")
        for face in faces:
            # bounding box
            box = list(map(int, face[:4]))
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
            threshold = 20
            if abs(angle) > threshold:
                distracted=True                
                print("Head turned away from camera!")
            else:
                distracted=False
                
                
            # calculate eye-nose distance
            eye_nose_dist = np.sqrt((left_eye_x - right_eye_x)**2 + (left_eye_y - right_eye_y)**2 + (nose_x - (left_eye_x + right_eye_x) / 2)**2 + (nose_y - (left_eye_y + right_eye_y) / 2)**2)
                
            # set thresholds for looking up or down
            up_threshold = 70
            down_threshold = 100
                
            # check if looking up or down
            if eye_nose_dist < up_threshold:
                distracted=True
                print("Looking up")
            elif eye_nose_dist > down_threshold:
                distracted=True
                print("Looking down")
            else:
                print("Looking straight")                
                distracted=False                
        
            # Degree of reliability
            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)
            
        if distracted:
            start_time += elapsed_time  # add elapsed time to start time to keep the timer going
        else:
            start_time = current_time  # reset start time to current time if flag is False
        
        # Display the image
        cv2.imshow("face detection", image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
def start_face_not_visible_timer():
    start_time_not_visible = time.time()
    return start_time_not_visible

if __name__ == '__main__':
    main()

