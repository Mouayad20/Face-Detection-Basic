import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, miDetectionCon=0.5):
        self.miDetectionCon = miDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(miDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # self.mpDraw.draw_detection(img,detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih),\
                    int(bboxC.width*iw), int(bboxC.height*ih)
                bboxs.append([id, bbox, detection.score])
                # cv2.rectangle(img, bbox, (200, 0, 0), 2)
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]),
                                cv2.FONT_HERSHEY_PLAIN, 3, (125, 54, 200), 3)
        return img, bboxs

    def fancyDraw(self, img, bbox, length=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (125, 54, 200), rt)
        # top left
        cv2.line(img, (x, y), (x+length, y), (125, 54, 200), t)
        cv2.line(img, (x, y), (x, y+length), (125, 54, 200), t)
        # top right
        cv2.line(img, (x1, y), (x1-length, y), (125, 54, 200), t)
        cv2.line(img, (x1, y), (x1, y+length), (125, 54, 200), t)
        # bottom left
        cv2.line(img, (x, y1), (x+length, y1), (125, 54, 200), t)
        cv2.line(img, (x, y1), (x, y1-length), (125, 54, 200), t)
        # bootom right
        cv2.line(img, (x1, y1), (x1-length, y1), (125, 54, 200), t)
        cv2.line(img, (x1, y1), (x1, y1-length), (125, 54, 200), t)
        return img
