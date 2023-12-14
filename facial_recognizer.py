import os
import cv2
import numpy as np
import onnxruntime as ort

import utils

class FacialRecognizer():
    def __init__(self, detection_model, recognition_model):
        self.detection_model = ort.InferenceSession(detection_model)
        self.recognition_model = ort.InferenceSession(recognition_model)
        
        self.feature_dataset = list()
    
    def initial_face_dataset(self, dataset_path, labels):
        dataset = []
        for person in labels:
            folder = os.path.join(dataset_path, person)
            for path in os.listdir(folder):
                img_path = os.path.join(folder, path)
                
                img = cv2.imread(img_path)
                ret, face, _ = self.detect_face(original_image=img)
                if ret is False:
                    continue

                print(f" - Loading {img_path}, {face.shape}, {person}")
                face_feature = self.extract_face_feature(face)
                
                dataset.append((person, face_feature))
        self.feature_dataset = dataset
        
    def detect_face(self, original_image, threshold=0.7) -> list:
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).astype(np.float32)

        input_name = self.detection_model.get_inputs()[0].name
        confidences, raw_boxes = self.detection_model.run(None, {input_name: image})
        
        boxes, _, _ = utils.choose_boxes(original_image.shape[1], 
                                         original_image.shape[0], 
                                         confidences, 
                                         raw_boxes, 
                                         threshold)
        ret, face = utils.crop_face(boxes=boxes, image=original_image)
        return ret, face, boxes
    
    def extract_face_feature(self, face_image):
        image = utils.facenet_preprocess(face_image)
        input_name = self.recognition_model.get_inputs()[0].name
        result = self.recognition_model.run(None, {input_name: image})
        
        return np.array(result)[0]
    
    def recognize_face(self, face_image, threshold=7, debug=False):
        feature = self.extract_face_feature(face_image)
    
        min_dist = -1
        name = ""
        for data in self.feature_dataset:
            dist = np.linalg.norm(data[1] - feature)
            if debug:
                print(data[0], dist)
            if min_dist>dist or min_dist==-1:
                min_dist = dist
                name = data[0]
                
        if min_dist<threshold:
            return name
        else:
            return "Guest"