from sre_constants import SUCCESS
import cv2
import mediapipe as mp
import time

from FaceDetection import FaceDetector


def main():
    cap = cv2.VideoCapture("videos/love.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
