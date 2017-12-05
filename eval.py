import numpy as np
import cv2 
import pandas as pd
import matplotlib.image as mpimg
from slide_window.vehicle_tracking import *

def draw_bb(img, bb_list, color=(0, 0, 255)):
    # draw bounding box on a img
    imcopy = np.copy(img)
    for bb in bb_list:
        cv2.rectangle(imcopy, (bb[0], bb[1]), (bb[2], bb[3]), color, 10)
    plt.imshow(imcopy)
    return


def bb_IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


class img_frame(object):
    def __init__(self, frame_name):
        self.frame_name = frame_name
        self.bb_true = None
        self.bb_pred = None

    def get_bb_true(self, db):
        df = db[db['Frame'] == self.frame_name]
        if len(df) == 0:
            return
        else:
            self.bb_true = []
            for row in df.iterrows():
                self.bb_true.append(
                    [row[1]['x1'], row[1]['y1'], row[1]['x2'], row[1]['y2'], row[1]['Label']])
            self.bb_true = np.array(self.bb_true)
            return

    def get_bb_pred(self, predictor):
        img = mpimg.imread(self.frame_name)
        self.bb_pred = predictor(img)
        return

    def annotate(self):
        img = mpimg.imread(self.frame_name)
        imcopy = np.copy(img)
        for bb in self.bb_true:
            cv2.rectangle(imcopy, (bb[0], bb[1]),
                          (bb[2], bb[3]), (0, 255, 0), 10)
        for bb in self.bb_pred:
            cv2.rectangle(imcopy, (bb[0], bb[1]),
                          (bb[2], bb[3]), (0, 0, 255), 10)
        plt.imshow(imcopy)
        return

    def score(self, class_id=0, threshold=0.5):
        """
        return TP, FP, FN
        """
        bb_true = self.bb_true
        bb_pred = self.bb_pred
        if (len(bb_true) == 0) and (len(bb_pred) == 0):
            return 0, 0, 0
        elif (len(bb_true) == 0) and (len(bb_pred) > 0):
            bb_pred_class = bb_pred[bb_pred[:, 4] == class_id]
            return 0, len(bb_pred_class), 0
        elif (len(bb_true) > 0) and (len(bb_pred) == 0):
            bb_true_class = bb_true[bb_true[:, 4] == class_id]
            return 0, 0, len(bb_true_class)
        else:
            bb_true_class = bb_true[bb_true[:, 4] == class_id]
            bb_pred_class = bb_pred[bb_pred[:, 4] == class_id]
            if (len(bb_true_class) == 0) and (len(bb_pred_class) == 0):
                return 0, 0, 0
            elif (len(bb_true_class) == 0) and (len(bb_pred_class) > 0):
                return 0, len(bb_pred_class), 0
            elif (len(bb_true_class) > 0) and (len(bb_pred_class) == 0):
                return 0, 0, len(bb_true_class)
            else:
                T_match = np.zeros(len(bb_true_class))
                P_match = np.zeros(len(bb_pred_class))
                for idx_t, tb in enumerate(bb_true_class):
                    for idx_p, pb in enumerate(bb_pred_class):
                        if P_match[idx_p]:
                            continue
                        else:
                            iou = bb_IOU(tb, pb)
                            if iou > threshold:
                                T_match[idx_t] = 1
                                P_match[idx_p] = 1
                                continue
                TP = len(T_match[T_match == 1])
                FP = len(P_match[P_match == 0])
                FN = len(T_match[T_match == 0])
                return TP, FP, FN


class base_evaluator(object):
    def __init__(self, csv_path):
        """
        csv_path: the path to annotation csv file. Assuming to have the following structure
        img_path, x1, y1, x2, y2, class
        """
        self.num_classes = 1
        self.class_map = {'car': 0, 'truck': 0, 'pedestrian': -1}
        self.db = pd.read_csv(
            csv_path, names=['Frame', 'x1', 'y1', 'x2', 'y2', 'Label'])
        self.db['Label'] = self.db['Label'].apply(lambda x: self.class_map[x])
        self.db = self.db[self.db['Label'] >= 0]
        self.detector = None
        self.frames = []

    def run_detection(self):
        for fname in self.db['Frame'].unique():
            print('Processing frame{}'.format(fname))
            frame = img_frame(fname)
            frame.get_bb_true(self.db)
            frame.get_bb_pred(self.detector)
            self.frames.append(frame)

    def evaluate_class(self, class_id):
        if not self.frames:
            self.run_detection()
        res = []
        for frame in self.frames:
            tp, fp, fn = frame.score(class_id=class_id)
            res.append([tp, fp, fn])
        res = np.array(res)
        res_sum = res.sum(axis=0)
        print('precision:{}'.format(res_sum[0] / (res_sum[0] + res_sum[1])))
        print('recall:{}'.format(res_sum[0] / (res_sum[0] + res_sum[2])))


class slide_window_eval(base_evaluator):
    def __init__(self, csv_path):
        super(slide_window_eval, self).__init__(csv_path)
        self.detector = vehicle_detector(resize_scale=0.5).detect


class rtna_coco_eval(base_evaluator):
    def __init__(self, csv_path):
        self.num_classes = 3
        self.class_map = {'car': 2, 'truck': 7, 'pedestrian': 0}
        self.db = pd.read_csv(
            csv_path, names=['Frame', 'x1', 'y1', 'x2', 'y2', 'Label'])
        self.db['Label'] = self.db['Label'].apply(lambda x: self.class_map[x])
        self.db = self.db[self.db['Label'] >= 0]
        self.detector = self.detect
        self.frames = []
        self.coco_model = None

    def load_model(self):
        self.coco_model = keras.models.load_model(
            'snapshots/resnet50_coco_best.h5', custom_objects=custom_objects)

    def detect(self, img):
        if not self.coco_model:
            self.load_model()
        scale = 0.5
        bb_list = []
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale)
        _, _, detections = self.coco_model.predict_on_batch(
            np.expand_dims(img_scaled, axis=0))
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(
            detections.shape[1]), 4 + predicted_labels]
        detections[:, :4] /= scale
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < 0.5:
                continue
            b = detections[0, idx, :4].astype(int)
            bb_list.append([b[0], b[1], b[2], b[3], label])
        return np.array(bb_list)


class rtna_resnet_eval(base_evaluator):
    def __init__(self, csv_path):
        self.num_classes = 3
        self.class_map = {'car': 0, 'truck': 1, 'pedestrian': 2}
        self.db = pd.read_csv(
            csv_path, names=['Frame', 'x1', 'y1', 'x2', 'y2', 'Label'])
        self.db['Label'] = self.db['Label'].apply(lambda x: self.class_map[x])
        self.db = self.db[self.db['Label'] >= 0]
        self.detector = self.detect
        self.frames = []
        self.coco_model = None

    def load_model(self):
        self.coco_model = keras.models.load_model(
            'snapshots/resnet18_best_bn.h5', custom_objects=custom_objects)

    def detect(self, img):
        if not self.coco_model:
            self.load_model()
        scale = 0.5
        bb_list = []
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale)
        _, _, detections = self.coco_model.predict_on_batch(
            np.expand_dims(img_scaled, axis=0))
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(
            detections.shape[1]), 4 + predicted_labels]
        detections[:, :4] /= scale
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < 0.5:
                continue
            b = detections[0, idx, :4].astype(int)
            bb_list.append([b[0], b[1], b[2], b[3], label])
        return np.array(bb_list)
        

