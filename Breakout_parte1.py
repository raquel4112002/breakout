import cv2
import numpy as np


def cv_setup(game):
    cv_init(game)
    cv_update(game)


def cv_init(game):
    game.cap = cv2.VideoCapture(0)
    if not game.cap.isOpened():
        game.cap.open(-1)

    # rest of init


def cv_update(game):
    cap = game.cap
    if not cap.isOpened():
        cap.open(-1)
    ret, image = cap.read()
    image = image[:, ::-1, :]
    cv_process(game, image)
    cv_output(image)
    # game.paddle.move(-1)
    game.after(1, cv_update, game)


def cv_process(game, image):
    color = [[91, 82, 46, 152, 255, 243]]


    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for color in color:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)


    kernel_dil = np.ones((3, 3), np.uint8)
    kernel_ero = np.ones((2, 2), np.uint8)
    erode = cv2.erode(mask, kernel_ero)
    dilate = cv2.dilate(erode, kernel_dil)



    left = np.sum(dilate[:, :int(dilate.shape[1] / 2)])
    right = np.sum(dilate[:, int(dilate.shape[1] / 2):])
    if left > right:
        game.paddle.move(-10)
    elif right > left:
     game.paddle.move(10)

    cv2.imshow("dilate", dilate)
def cv_output(image):
    cv2.imshow("window", image)

    # rest of output rendering
    cv2.waitKey(1)