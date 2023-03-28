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

    # Initialize variables for timer
    start_time_visible = None
    start_time_not_visible = None
    face_visible = False
    elapsed_time_visible = 0
    elapsed_time_not_visible = 0

    while True:
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
                start_time_not_visible = time.time()
            elif time.time() - start_time_not_visible > 1:
                elapsed_time_not_visible += time.time() - start_time_not_visible
                print(f"Face not visible for {elapsed_time_not_visible:.2f} seconds")
                start_time_not_visible = time.time()
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
            if(face==[]):
                print("NO FACE")
            print(box)

            # Landmarks (Right Eye, Left Eye, Nose, Right Mouth Corner, Left Mouth Corner)
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)
                
            # Degree of reliability
            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)

        # Display the image
        cv2.imshow("face detection", image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

