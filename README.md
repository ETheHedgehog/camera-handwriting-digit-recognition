# camera-handwriting-digit-recognition
Recognize digits written by hand gestures using a neural network

Using the Mediapipe and OpenCV library, this program recognizes three main gestures:
1. Draw: open the index finger and close all other fingers, the program will allow you to draw using the tip of the index finger within the specified region

2. Clear: open both the thumb and the index finger to clear the canvas

3. Recognize: close all fingers and the program will recognize the digit and clear the canvas

Currently the program only works with right hands.

The current trained_model.h5 was trained at 500 iterations, you can change the number of iterations in TrainNN.py and train it again for more accuracy

Start the program by running the Recognize.py file
