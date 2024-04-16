from tensorflow.keras.models import load_model
import cv2
import numpy as np
from utils import sigmoid

class YoloDetector:
    """
    Represents an object detector for robot soccer based on the YOLO algorithm.
    """
    def __init__(self, model_name, anchor_box_ball=(5, 5), anchor_box_post=(2, 5)):
        """
        Constructs an object detector for robot soccer based on the YOLO algorithm.

        :param model_name: name of the neural network model which will be loaded.
        :type model_name: str.
        :param anchor_box_ball: dimensions of the anchor box used for the ball.
        :type anchor_box_ball: bidimensional tuple.
        :param anchor_box_post: dimensions of the anchor box used for the goal post.
        :type anchor_box_post: bidimensional tuple.
        """
        self.network = load_model(model_name + '.hdf5')
        self.network.summary()  # prints the neural network summary
        self.anchor_box_ball = anchor_box_ball
        self.anchor_box_post = anchor_box_post

    def detect(self, image):
        """
        Detects robot soccer's objects given the robot's camera image.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        # Todo: implement object detection logic
        output = self.network.predict(self.preprocess_image(image))
        ball_detection = (0.0, 0.0, 0.0, 0.0, 0.0)  # Todo: remove this line
        post1_detection = (0.0, 0.0, 0.0, 0.0, 0.0)  # Todo: remove this line
        post2_detection = (0.0, 0.0, 0.0, 0.0, 0.0)  # Todo: remove this line
        ball_detection, post1_detection, post2_detection = self.process_yolo_output(output)

        return ball_detection, post1_detection, post2_detection

    def preprocess_image(self, image):
        """
        Preprocesses the camera image to adapt it to the neural network.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: image suitable for use in the neural network.
        :rtype: NumPy 4-dimensional array with dimensions (1, 120, 160, 3).
        """
        # Todo: implement image preprocessing logic
        width = 160
        height = 120
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = image/255.0
        image = np.reshape(image, (1, 120, 160, 3))
        return image

    def process_yolo_output(self, output):
        """
        Processes the neural network's output to yield the detections.

        :param output: neural network's output.
        :type output: NumPy 4-dimensional array with dimensions (1, 15, 20, 10).
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        coord_scale = 4 * 8  # coordinate scale used for computing the x and y coordinates of the BB's center
        bb_scale = 640  # bounding box scale used for computing width and height
        output = np.reshape(output, (15, 20, 10))  # reshaping to remove the first dimension
        # Todo: implement YOLO logic

        ball_prob = -9999
        ball_index = [0, 0]
        post1_prob = -9999
        post1_index = [0, 0]
        post2_prob = -9999
        post2_index = [0, 0]
        ball_detection = (0.0, 0.0, 0.0, 0.0, 0.0)
        post1_detection = (0.0, 0.0, 0.0, 0.0, 0.0)
        post2_detection = (0.0, 0.0, 0.0, 0.0, 0.0)

        for i in range(15):
            for j in range(20):
                if output[i][j][0] > ball_prob:
                    ball_prob = output[i][j][0]
                    ball_index = [i, j]
                if output[i][j][5] > post1_prob:
                    post1_prob = output[i][j][5]
                    post1_index = [i, j]
                if output[i][j][5] > post2_prob and output[i][j][5] < post1_prob:
                    post2_prob = output[i][j][5]
                    post2_index = [i, j]

        ball_detection = output[ball_index[0]][ball_index[1]][:5]
        post1_detection = output[post1_index[0]][post1_index[1]][5:]
        post2_detection = output[post2_index[0]][post2_index[1]][5:]

        self.object_normalization(ball_detection, ball_index, [5, 5])
        self.object_normalization(post1_detection, post1_index, [2, 5])
        self.object_normalization(post2_detection, post2_index, [2, 5])

        return ball_detection, post1_detection, post2_detection

    def object_normalization(obj, idx, anchor):
        #Legend for detection parameters:
        # obj[0] = probability of the object
        # obj[1] = x position of the center of the object
        # obj[2] = y position of the center of the object
        # obj[3] = width of the bounding box
        # obj[3] = height of the bounding box

        obj[0] = sigmoid(obj[0])
        obj[1] = (idx[1] + sigmoid(obj[1]))*32
        obj[2] = (idx[0] + sigmoid(obj[2]))*32
        obj[3] = anchor[0]*np.exp(obj[3])*640
        obj[4] = anchor[1]*np.exp(obj[4])*640
