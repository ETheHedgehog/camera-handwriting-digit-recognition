import numpy as np
import cv2
from keras.models import load_model

model = load_model('trained_model.h5')


def recognize_digit(img):
    resized = cv2.resize(img, (28, 28))
    resized_inverted = 255 - resized
    final_input = resized_inverted.reshape(1, 28, 28, 1)
    ans_mat = model.predict(final_input)
    ans = np.argmax(ans_mat, axis=1)[0]
    return ans, resized_inverted
