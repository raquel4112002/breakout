import cv2
import numpy as np

def cv_setup(game):
    cv_init(game)
    cv_update(game)


def cv_init(game):
    game.cap = cv2.VideoCapture(0)
    if not game.cap.isOpened():
        game.cap.open(-1)


def cv_update(game):
    cv_process(game)
    game.after(1, cv_update, game)

def cv_process(game):
    processImg = ProcessImage()
    processImg.DetectObject(game)


def cv_output(frame):
    cv2.imshow("window", frame)

    # rest of output rendering
    cv2.waitKey(1)

# Instantiate OCV kalman filter
class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    #This function estimates the position of the object
    def Estimate(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

#Performs required image processing to get obeject coordinated
class ProcessImage:

    def DetectObject(self, game):

        # Create Kalman Filter Object
        kfObj = KalmanFilter()
        predictedCoords = np.zeros((2, 1), np.float32)

        cap = game.cap
        if not cap.isOpened():
            cap.open(-1)
        ret, frame = cap.read()

        [objectX, objectY] = self.DetectColor(frame)
        predictedCoords = kfObj.Estimate(objectX, objectY)

        # Draw Actual coords from segmentation
        cv2.circle(frame, (int(objectX), int(objectY)), 20, [0,0,255], 2, 8)
        cv2.line(frame,(int(objectX), int(objectY + 20)), (int(objectX + 50), int(objectY + 20)), [100,100,255], 2,8)
        cv2.putText(frame, "Actual", (int(objectX + 50), int(objectY + 20)), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])

        # Draw Kalman Filter Predicted output
        cv2.circle(frame, (int(predictedCoords[0]), int(predictedCoords[1])), 20, [0,255,255], 2, 8)
        cv2.line(frame, (int(predictedCoords[0]) + 16, int(predictedCoords[1]) - 15), (int(predictedCoords[0]) + 50, int(predictedCoords[1]) - 30), [100, 10, 255], 2, 80)
        cv2.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
        """left = np.sum(frame[:, :int(frame.shape[1] / 2)])
           right = np.sum(frame[:, int(frame.shape[1] / 2):])"""

        """ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        biggestArea = 0
        cx = 0
        biggestContour = None

        if len(contours) != 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 15000:
                    M = cv2.moments(contour)
                    cx = int(M['m10'] / M['m00'])
                    if area > biggestArea:
                        biggestArea = area
                        biggestContour = contour
                    x, y, w, h = cv2.boundingRect(biggestContour)
                    cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), 3)

        cv2.imshow("windowbg", mask)"""

        if int(predictedCoords[0])> 300:
            game.paddle.move(10)
        elif int(predictedCoords[0]) < 300:
            game.paddle.move(-10)
        else:
            game.paddle.move(0)

        cv_output(frame)

    # Segment the Blue obeject in a  frame
    def DetectColor(self, frame):

        # Set threshold to filter only green color & Filter it
        lower = np.array([91, 82, 46], dtype = "uint8")
        upper = np.array([152, 255, 243], dtype = "uint8")
        mask = cv2.inRange(frame, lower, upper)

        # Dilate
        kernel = np.ones((5, 5), np.uint8)
        MaskDilated = cv2.dilate(mask, kernel)


        # Find obeject as it is the biggest blue object in the frame
        [nLabels, labels, stats, centroids] = cv2.connectedComponentsWithStats(MaskDilated, 8, cv2.CV_32S)

        # First biggest contour is image border always, Remove it
        stats = np.delete(stats, (0), axis = 0)
        try:
            maxBlobIdx_i, maxBlobIdx_j = np.unravel_index(stats.argmax(), stats.shape)

        # This is our obeject coords that needs to be tracked
            objectX = stats[maxBlobIdx_i, 0] + (stats[maxBlobIdx_i, 2]/2)
            objectY = stats[maxBlobIdx_i, 1] + (stats[maxBlobIdx_i, 3]/2)
            return [objectX, objectY]

        except:
               pass

        return [0,0]