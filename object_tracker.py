from models import *
from utils import utils

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image
import cv2
from sortnew import *
from math import ceil

def main():
    with open(approved_classes_file) as file:
        approved_classes = [row.strip() for row in file]

    model = Darknet(config_path, img_size=img_size)
    model.load_darknet_weights(weights_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    classes = utils.load_classes(class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def detect_image(img):
        # scale and pad image
        ratio = min(img_size/img.size[0], img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
             transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                            (128,128,128)),
             transforms.ToTensor(),
             ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(Tensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = model(input_img)
            detections = utils.non_max_suppression(detections, conf_thres, nms_thres)
        return detections[0]


    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

    cap = cv2.VideoCapture(source)
    mot_tracker = Sort(max_age=10)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()
    if ret is True:
        vw = frame.shape[1]
        vh = frame.shape[0]
        print("Video ", source, " size ", vw, 'x', vh)
    else:
        print("Can't read video ", source)
        exit()

    cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stream', (vw, vh))

    if video_write_output is True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outvideo = cv2.VideoWriter(source.replace(".mp4", "-det.avi"), fourcc, fps, (vw, vh))

    frames = 0
    start_time = time.time()
    while True:
        frame_detection_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        objects_in_frame = 0
        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())
            objects_in_frame = tracked_objects.shape[0]

            unique_labels = detections[:, -1].cpu().unique()
            scores = detections[:, 4].cpu().numpy()
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                cls = classes[int(cls_pred)]
                if cls not in approved_classes:
                    tracked_objects -= 1
                    break

                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                color = colors[int(obj_id) % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*15+80, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)) + ' ', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.imshow('Stream', frame)
        total_time = time.time() - start_time
        print("frame", frames, ",", objects_in_frame, "tracked objects, time", time.time() - frame_detection_time)
        if video_write_output:
            outvideo.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if video_in_realtime and frames < total_time * fps:
            while frames < total_time * fps:
                cap.read()
                frames += 1

    totaltime = time.time() - start_time
    print(frames, "frames", totaltime/frames, "average sec/frame")
    cv2.destroyAllWindows()
    cap.release()
    if video_write_output:
        outvideo.release()
    return


if __name__ == '__main__':
    config_path = 'yolofiles/yolov3.cfg'
    weights_path = 'yolofiles/yolov3.weights'  # download it via https://pjreddie.com/darknet/yolo/
    class_path = 'yolofiles/coco.names'
    approved_classes_file = 'yolofiles/approved.names'  # define classes to predict

    img_size = 320  # 320 416 512, num must be % 32
    conf_thres = 0.8
    nms_thres = 0.4  # non max supression

    source = 'traffic.mp4'  # set to 0 for web camera
    video_in_realtime = False  # skipping frames if system too slow
    video_write_output = False
    if video_write_output:
        video_output_path = source.replace(".mp4", "-detected.avi")

    # source = 'http://85.237.63.165/mjpg/video.mjpg'
    main()
