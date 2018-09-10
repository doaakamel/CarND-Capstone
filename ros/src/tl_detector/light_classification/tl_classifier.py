from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

        # Path to frozen detection graph.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        self.model = None
        self.width = 0
        self.height = 0
        self.channels = 3

        # Load a frozen model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
            # Input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        #return TrafficLight.UNKNOWN
        image_np = np.asarray(image, dtype="uint8")
        image_np_expanded = np.expand_dims(image_np, axis=0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        #tl_cropped = None
        for idx, classID in enumerate(classes):
            if classID == 10: #10 is traffic light
                if scores[idx] > 0.50: #confidence level

                    nbox = boxes[idx]

                    height = image.shape[0]
                    width = image.shape[1]

                    box = np.array([nbox[0]*height, nbox[1]*width, nbox[2]*height, nbox[3]*width]).astype(int)
                    tl_cropped = image[box[0]:box[2], box[1]:box[3]]
                    tl_color = self.get_color(tl_cropped)
                    #augment image with detected TLs
                    cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_color = (0, 0, 0)
                    cv2.putText(image, tl_color, (box[1], box[0]), font, 1.0, font_color, lineType=cv2.LINE_AA)
        return image

    def get_color(self, image_rgb):
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l = image_lab.copy()
        # set a and b channels to 0
        l[:, :, 1] = 0
        l[:, :, 2] = 0

        std_l = self.standardize_input(l)

        red_slice, yellow_slice, green_slice = self.slice_image(std_l)

        y, x, c = red_slice.shape
        px_sums = []
        color = ['RED', 'YELLOW', 'GREEN']
        px_sums.append(np.sum(red_slice[0:y, 0:x, 0]))
        px_sums.append(np.sum(yellow_slice[0:y, 0:x, 0]))
        px_sums.append(np.sum(green_slice[0:y, 0:x, 0]))

        max_value = max(px_sums)
        max_index = px_sums.index(max_value)

        return color[max_index]

    def crop_n_blur(self, image):
        row = 2
        col = 6
        blured_img = image.copy()
        blured_img = blured_img[row:-row, col:-col, :]
        blured_img = cv2.GaussianBlur(blured_img, (3, 3), 0)
        return blured_img

    def standardize_input(self, image):
        standard_img = np.copy(image)
        standard_img = cv2.resize(standard_img, (32, 32))
        standard_img = self.crop_n_blur(standard_img)
        return standard_img

    def slice_image(self, image):
        img = image.copy()
        shape = img.shape
        slice_height = shape[0] / 3
        upper = img[0:slice_height, :, :]
        middle = img[slice_height:2 * slice_height, :, :]
        lower = img[2 * slice_height:3 * slice_height, :, :]
        return upper, middle, lower