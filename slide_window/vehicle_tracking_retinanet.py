import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.ndimage.measurements import label
from tqdm import tqdm
from moviepy.editor import VideoFileClip, ImageSequenceClip
import keras
from keras_retinanet.models.resnet import custom_objects


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def generate_heat_map(map_size, bbox_list):
    """
    generate heat map and probability map after performning slide window search.
    heat map counts the number of times a region was recognized as vehicle with probability 0.7 or higher
    probability map stores the maximum probability the region received in slide window search
    """
    heatmap = np.zeros(map_size)
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    color = (100, 100, 250)
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
 
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    # Return the image
    return img





class car_tracker(object):
    """
    class for tracking driving lanes in video
    """

    def __init__(self, video=0):
        self.frame_ct = 0
        self.detected = False
        self.decay_factor = 0.2
        self.scale = 1.0
        self.heatmap_thresh = 0.8
        self.detection_threshold = 0.6
        self.recent_heatmap = []
        self.current_heatmap = None
        self.img_shape = []
        self.ystart = 360
        self.ystop = 720
        self.model = None 
        self.label_list= [0,1]
        self.antn_img = []


    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)

    def car_search_retinanet(self, img):
        scale = self.scale
        bb_list = []
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale)
        _, _, detections = self.model.predict_on_batch(
            np.expand_dims(img_scaled, axis=0))
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(
            detections.shape[1]), 4 + predicted_labels]
        detections[:, :4] /= scale
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < self.detection_threshold:
                continue
            if label not in self.label_list:
                continue
            b = detections[0, idx, :4].astype(int)
            bb_list.append(((b[0], b[1]), (b[2], b[3])))
        
        img_size = (img.shape[0], img.shape[1])
        heatmap = generate_heat_map(img_size, bb_list)
        return heatmap

    def detect_car_from_img(self, img):
        """
        apply the pipeline to find driving lanes in each frame
        """
        self.frame_ct += 1

        self.current_heatmap= self.car_search_retinanet(img)
        self.update_history()  # update lane parameters
        # save the annotated images
        heatmap = apply_threshold(self.current_heatmap, self.heatmap_thresh)
        labels = label(heatmap)
        imcopy = np.copy(img)
        draw_labeled_bboxes(imcopy, labels)
        self.antn_img.append(imcopy)

    def update_history(self):
        if self.detected == False:
            self.recent_heatmap.append(np.copy(self.current_heatmap))
            self.detected = True
        else:
            self.current_heatmap = self.decay_factor * self.current_heatmap + \
                (1 - self.decay_factor) * self.recent_heatmap[-1]
            self.recent_heatmap.append(np.copy(self.current_heatmap))


if __name__ == "__main__":
    clip = VideoFileClip('project_video.mp4')
    clip = [frame for frame in clip.iter_frames()]
    ct = car_tracker()
    for img in tqdm(clip):
        ct.detect_car_from_img(img)
    gif_clip = ImageSequenceClip(ct.antn_img, fps=25)
    gif_clip.write_videofile('video_output.mp4')
