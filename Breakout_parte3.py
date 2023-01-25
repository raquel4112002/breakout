import cv2
import numpy as np
import yolov5

model = yolov5.load('../yolov5n.pt')
model.conf = 0.33

def cv_setup(game):
    cv_init(game)
    cv_update(game)


def cv_init(game):
    game.cap = cv2.VideoCapture(0)



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

        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(imageRGB)
        output = frame.copy()
        w = 0.0
        h = 0.0
        x = 0.0
        y = 0.0

        for pred in enumerate(results.pred):
            im = pred[0]
            im_boxes = pred[1]
            for *box, conf, cls in im_boxes:
                box_class = int(cls)
                conf = float(conf)
                x = float(box[0])
                y = float(box[1])
                w = float(box[2]) - x
                h = float(box[3]) - y
                pt1 = np.array(np.round((float(box[0]), float(box[1]))), dtype=int)
                pt2 = np.array(np.round((float(box[2]), float(box[3]))), dtype=int)
                box_color = (255, 0, 0)
                if results.names[box_class] == 'botele':
                    text = "{}:{:.2f}".format(results.names[box_class], conf)
                    cv2.rectangle(img=output,
                                  pt1=pt1,
                                  pt2=pt2,
                                  color=box_color,
                                  thickness=1)
                    cv2.putText(img=output,
                                text=text,
                                org=np.array(np.round((float(box[0]), float(box[1] - 1))), dtype=int),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=box_color,
                                thickness=1)



        objectX =  x
        objectY =  y

        predictedCoords = kfObj.Estimate(objectX, objectY)




        if x > 250:
            game.paddle.move(10)
        elif x < 250:
            game.paddle.move(-10)
        else:
            game.paddle.move(0)

        cv2.imshow("YOLOv5", output)
        cv_output(frame)

