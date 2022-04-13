import cv2
import numpy as np
import time
from Output import recognize_digit
from Detector import HandDetector

cap = cv2.VideoCapture(0)
prevTime = 0
hand = HandDetector()


def check_inside_canvas(coordinates, offset, canvas_size):
    absPos = (coordinates[0] - offset[0], coordinates[1] - offset[1])
    if absPos[0] >= canvas_size or absPos[0] < 0:
        return False
    if absPos[1] >= canvas_size or absPos[1] < 0:
        return False
    return True


def check_inside_clear(coordinates, offset):
    if coordinates[0] >= offset - 40 or coordinates[0] < offset - 180:
        return False
    if coordinates[1] >= 100 or coordinates[1] < 40:
        return False
    return True


def paint(canvas, points):
    if points[1][0] == -1:
        return
    cv2.line(canvas, points[0], points[1], (0, 0, 0), 20)


def update_screen(bg, canv, length):
    bgh, bgw, bgc = bg.shape
    cv2.putText(bg, f'FPS:{int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(bg, f'{states}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.rectangle(bg, (bgw // 2, bgh // 6), (bgw // 2 + length, bgh // 6 + length), (0, 0, 0), 1)
    # cv2.rectangle(bg, (bgw - 180, 40), (bgw - 40, 100), (0, 0, 0), -1)
    # cv2.putText(bg, 'CLEAR', (bgw - 160, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Test", bg)
    cv2.imshow("Draw", canv)


test, testFrame = cap.read()
canvasLength = 2 * (testFrame.shape[1] // 3)
canvas = np.ones((canvasLength, canvasLength), np.uint8) * 255
delay = 1000000000
lastClick = 0
curTime = 0
lastPos = (-1, -1)

while True:
    suc, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_AREA)
    frameInv = cv2.flip(frame, 1)
    h, w, c = frameInv.shape
    frameInv = hand.find_hands(frameInv)
    hand.find_position(frameInv)
    hand.update_fingers()

    if hand.state == "Draw":
        if check_inside_canvas(hand.landmarkList[8], (w // 2, h // 6), canvasLength):
            pos = (hand.landmarkList[8][0] - w // 2, hand.landmarkList[8][1] - h // 6)
            paint(canvas, (pos, lastPos))
            lastPos = pos
    elif hand.state == "Clear":
        lastPos = (-1, -1)
        canvas = np.ones((canvasLength, canvasLength), np.uint8) * 255
    elif hand.state == "Recognize":
        lastPos = (-1, -1)
        if curTime - lastClick > delay:
            lastClick = curTime
            recognized_digit, input_img = recognize_digit(canvas)
            print(recognized_digit)
            canvas = np.ones((canvasLength, canvasLength), np.uint8) * 255
    else:
        lastPos = (-1, -1)

    curTime = time.time_ns()
    fps = 1000000000 / (curTime - prevTime)
    prevTime = curTime
    update_screen(frameInv, canvas, canvasLength)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
