import os
import cv2
import time
import json
import argparse

from model import Model
from door_controller import DoorController
from logger import Logger

log = Logger().get_log()

parser = argparse.ArgumentParser()
parser.add_argument('--run_mode', default="lite", help='lite, normal, or incredible')
parser.add_argument('--dataset_path', default="dataset/", help='path to dataset')
parser.add_argument('--camera', default=0, help='camera index')
parser.add_argument('--door', default=False, help='door controller')
parser.add_argument('--show', default=False, help='view mode')
parser.add_argument('--threshold', default=False, help='recognition threshold')
parser.add_argument('--debug', default=False, help='debug mode')


def run_inference(frame, model, debug)->(str, list):
    _, face, boxes = model.detect_face(frame)
    if len(boxes) == 0:
        return None, [(0,0,0,0)]
    else:
        result = model.recognize_face(face, debug=debug)
        return result, boxes

def start_streaming(camera, model, show_result=True, debug=False):
    while True:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            log.warning("Cannot open camera")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Can't receive frame.")
                
                try:
                    start_time = time.time()
                    # result, boxes = run_inference(frame, model, debug)
                    result, boxes = model.recognize_face(frame, threshold=threshold, debug=debug)
                    log.debug(f"Inference time: {time.time() - start_time}")
                    if args.door:
                        ret, name = door.visit(result)
                        if ret == True:
                            log.info(f"Hello, {name}.")
                        elif name != "No Person":
                            log.info(f"Found guest!")
                except Exception as e:
                    log.warning(f"Got Exception: {e}")
                    result = None

                if show_result == False:
                    continue
                
                if result != None:
                    x1 = boxes[0][0]
                    y1 = boxes[0][1]
                    x2 = boxes[0][2]
                    y2 = boxes[0][3]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (214, 217, 8), 2, cv2.LINE_AA)
                    cv2.putText(frame, result, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (214, 217, 8), 2, cv2.LINE_AA)

                try:
                    image = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
                    cv2.imshow("Result", image)
                    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
                        break
                except:
                    log.warning("Can't show image")
            
            cap.release()       
            cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parser.parse_args()
    
    config = json.load(open("config.json"))
    
    camera = config["video"] if args.camera == "CYL" else int(args.camera)
    
    if args.door:
        door = DoorController(url = config["open"])
    
    model = Model(config=config, mode=args.run_mode)
    model.initial_face_dataset(dataset_path=args.dataset_path,
                                labels=os.listdir(args.dataset_path))
    
    threshold = float(args.threshold)
    
    start_streaming(camera=camera, model=model, show_result=args.show, debug=args.debug)