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
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(imageRGB)
    output = image.copy()

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
            if results.names[box_class] == 'person':
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

                if x > 250:
                    game.paddle.move(10)
                elif x < 250:
                    game.paddle.move(-10)
                else:
                    game.paddle.move(0)

    cv2.imshow("YOLOv5", output)
def cv_output(image):
        cv2.imshow("window", image)

        # rest of output rendering
        cv2.waitKey(1)