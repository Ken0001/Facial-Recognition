import os
import cv2
import numpy as np

import onnxruntime as ort

import utils

class FaceDetectorONNX():
    def __init__(self, path) -> None:
        self.model = ort.InferenceSession(path)
    
    def preprocess_input(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image = (image - np.array([127, 127, 127])) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image
    
    def inference(self, original_image, threshold=0.7) -> list:
        image = self.preprocess_input(original_image)
        input_name = self.model.get_inputs()[0].name
        confidences, raw_boxes = self.model.run(None, {input_name: image})
        
        boxes, _, _ = utils.choose_boxes(original_image.shape[1], 
                                         original_image.shape[0], 
                                         confidences, 
                                         raw_boxes, 
                                         threshold)
        ret, face = utils.crop_face(boxes=boxes, image=original_image)
        return ret, face, boxes
    
class FaceRecognizerONNX():
    """
        Return: 1x512 vector
    """
    def __init__(self, path) -> None:
        self.model = ort.InferenceSession(path)
    
    def preprocess_input(self, image):
        image = cv2.resize(image, (160,160))
        image = np.transpose(image, (2, 0, 1))
        norm_img_data = np.zeros(image.shape).astype('float32')
        for i in range(image.shape[0]):
            norm_img_data[i, :, :] = (image[i, :, :] - 127.5) / 128
        norm_img_data = np.expand_dims(norm_img_data, axis=0).astype(np.float32)
        return norm_img_data
    
    def inference(self, face_image):
        image = self.preprocess_input(face_image)
        input_name = self.model.get_inputs()[0].name
        result = self.model.run(None, {input_name: image})
        
        return np.array(result)[0]
    

    
    
class Model():
    def __init__(self, config, mode="lite") -> None:
        self.config = config
        self.face_feature_dataset = None
        self.face_detector = FaceDetectorONNX(self.config['model']['detector_path'])
        self.face_extractor = None
        if mode == "lite":
            self.face_extractor = FaceRecognizerONNX(self.config['model']['recognizer_path'])
        else:
            from facenet import Facenet
            self.face_extractor = Facenet()
            pass
        
    def initial_face_dataset(self, dataset_path, labels):
        dataset = []
        for person in labels:
            folder = os.path.join(dataset_path, person)
            for path in os.listdir(folder):
                img_path = os.path.join(folder, path)
                
                img = cv2.imread(img_path)
                ret, face, _ = self.face_detector.inference(original_image=img)
                if ret is False:
                    continue

                print(f" - Loading {img_path}, {face.shape}, {person}")
                face_feature = self.face_extractor.inference(face)
                
                dataset.append((person, face_feature))
        self.face_feature_dataset = dataset
        
    def recognize_face(self, image, threshold=7, debug=False):
        ret, face, boxes = self.face_detector.inference(original_image=image)
        if ret is False:
            return None, [(0,0,0,0)]
        feature = self.face_extractor.inference(face)
    
        min_dist = -1
        name = ""
        for data in self.face_feature_dataset:
            dist = np.linalg.norm(data[1] - feature)
            if debug:
                print(data[0], dist)
            if min_dist>dist or min_dist==-1:
                min_dist = dist
                name = data[0]
                
        if min_dist<threshold:
            return name, boxes
        else:
            return "Guest", boxes
        
        
if __name__ == "__main__":
    
    from facenet import Facenet
    
    image1 = cv2.imread("dataset/Ken/S__24674311.jpg")
    # image2 = cv2.imread("dataset/Ken/S__24674312.jpg")
    image2 = cv2.imread("dataset/Mark/S__25403461.jpg")
    
    detector_path = "models/face_detector.onnx"
    recognizer_path = "models/FaceNet_vggface2_optimized.onnx"
    
    face_detector = FaceDetectorONNX(detector_path)
    # face_extractor = FaceRecognizerONNX(recognizer_path)
    face_extractor = Facenet()
    
    _, face1, _ = face_detector.inference(image1)
    _, face2, _ = face_detector.inference(image2)
    
    
    output1 = face_extractor.inference(face1)
    output2 = face_extractor.inference(face2)
    
    print(output1.shape)
    print(output2.shape)
    
    print("---")
    print(np.linalg.norm(output1 - output2, axis=1))
    print(np.linalg.norm(output1 - output2))