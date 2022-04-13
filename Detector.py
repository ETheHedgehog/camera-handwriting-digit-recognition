import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.fingers = {}
        self.state = "No Action"
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.landmarkList = []

    def find_hands(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def find_position(self, frame, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for lm in myHand.landmark:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append((cx, cy))
                if draw:
                    cv2.rectangle(frame, (cx - 3, cy - 3), (cx + 3, cy + 3), (255, 0, 255), 2)
        self.landmarkList = lmlist
        return lmlist

    def update_fingers(self):
        if len(self.landmarkList) == 0:
            self.state = "No Action"
            return
        self.fingers = {"thumb": False,
                        "index": False,
                        "middle": False,
                        "ring": False,
                        "pinky": False}
        if self.landmarkList[4][0] < self.landmarkList[2][0] and self.landmarkList[3][0] < self.landmarkList[2][0]:
            self.fingers["thumb"] = True
        if self.landmarkList[7][1] < self.landmarkList[6][1] and self.landmarkList[8][1] < self.landmarkList[6][1]:
            self.fingers["index"] = True
        if self.landmarkList[11][1] < self.landmarkList[10][1] and self.landmarkList[12][1] < self.landmarkList[10][1]:
            self.fingers["middle"] = True
        if self.landmarkList[15][1] < self.landmarkList[14][1] and self.landmarkList[16][1] < self.landmarkList[14][1]:
            self.fingers["ring"] = True
        if self.landmarkList[19][1] < self.landmarkList[18][1] and self.landmarkList[20][1] < self.landmarkList[18][1]:
            self.fingers["pinky"] = True
        if self.draw():
            self.state = "Draw"
        elif self.recognize():
            self.state = "Recognize"
        elif self.clear():
            self.state = "Clear"
        else:
            self.state = "No Action"

    def draw(self):
        ret = True
        for key in self.fingers:
            if key == "index":
                ret = ret and self.fingers[key]
            else:
                ret = ret and (not self.fingers[key])
        return ret

    def recognize(self):
        ret = True
        for key in self.fingers:
            ret = ret and (not self.fingers[key])
        return ret

    def clear(self):
        ret = True
        for key in self.fingers:
            if key == "index" or key == "thumb":
                ret = ret and self.fingers[key]
            else:
                ret = ret and (not self.fingers[key])
        return ret
