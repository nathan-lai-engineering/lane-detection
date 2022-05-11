import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import True_, random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
import statistics as stats
import math
import csv
import json

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result, plot_img_and_mask, get_lane, get_center
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if opt.save_dir != "":
      if os.path.exists(opt.save_dir):  # output dir
          shutil.rmtree(opt.save_dir)  # delete dir
      os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    if not opt.save_csv is None:
      if not (os.path.exists(opt.save_csv) and os.path.isdir(opt.save_csv)):
        os.mkdir(opt.save_csv)

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, opt.detect_fps, img_size=opt.img_size, save_csv=opt.save_csv)
        bs = 1  # batch_size


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    names = ["car"]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    img_centers = {}
    json_data = [None, None]

    # stores row data for each video/img
    # file name : list of rows
    # row formatted as (frame number, "lane number x1 y1 x2, y2")
    csv_data = {}
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        if not os.path.basename(path) in csv_data:
          csv_data[os.path.basename(path)] = {}
        #print(opt.external_boxes)

        # json data loader
        if opt.external_boxes and os.path.exists(opt.external_boxes) and os.path.isdir(opt.external_boxes):
          json_filename = str(os.path.basename(path)) + '.json'
          if json_filename in os.listdir(opt.external_boxes):
            if json_filename != json_data[0]:
              json_data[0] = json_filename
              with open(os.path.join(opt.external_boxes, json_filename), 'r') as json_file:
                print("\n", 'loading', json_filename)
                json_data[1] = json.load(json_file)
                print("\n", json_filename, "loaded")
          else:
            json_data = [None, None]
            if opt.enforce_external == True:
              print("Error, no external bounding boxes loaded 1")
              continue
        else:
          json_data = [None, None]
          
          if opt.enforce_external == True:
            print("Error, no external bounding boxes loaded 2")
            continue
        #print(len(json_data[1]))

        detections_list = []
        use_yolop = False
        current_frame_number = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not json_data[1] is None and current_frame_number < len(json_data[1]):
          search_list = json_data[1]
          target = None
          while target is None:
            list_midpoint = int(len(search_list) / 2)
            json_index = int(search_list[list_midpoint]['frame'] / 30)
            if len(search_list) < 2:
              target = search_list[list_midpoint]
            if int(json_index) == current_frame_number:
              target = search_list[list_midpoint]
            if int(json_index) < current_frame_number:
              search_list = search_list[list_midpoint + 1:]
            else:
              search_list = search_list[:list_midpoint]
            #print(target)
          print(current_frame_number, target)
          if(not target is None):
            for k in range(len(target['rois'])):
              if target['class_ids'][k] in [3,4,6,8]:
                x_factor = 1280 / 1920
                y_factor = 720 / 1080
                json_xyxy = [int(target['rois'][k][1] * y_factor), int(target['rois'][k][0] * x_factor), int(target['rois'][k][3] * y_factor), int(target['rois'][k][2] * x_factor)]
                #print(target['frame'])
                detections_list.append(json_xyxy + [target['scores'][k]] + [target['class_ids'][k]])
        
        else:
          if opt.enforce_external is True:
            print("Error, no external bounding boxes loaded on load")
            continue
          else:
            use_yolop = True

        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()
        #if i == 0:
        #    print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]


        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        #plot_img_and_mask(img_det, ll_seg_mask, 1, 1, opt.save_dir)
        
        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()

            if opt.img_center is None:
              if not path in img_centers:
                img_centers[path] = []
              if vid_cap is not None and len(img_centers[path]) <= 0.5 * vid_cap.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_center, points_to_graph = get_center(ll_seg_mask)
                img_centers[path].append(frame_center)

              frame_center_mean = int(stats.mean(img_centers[path]))
            else:
              frame_center_mean = opt.img_center
            

            minimum_diff = None
            lane_numbers = {}

            if use_yolop is True:
              for *xyxy,conf,cls in reversed(det):
                detections_list.append(xyxy + [conf])

            # get lane of current box
            for xyxy in detections_list:
              lane_number, points_to_graph = get_lane(xyxy, ll_seg_mask, inp_center=frame_center_mean)
              coords = []
              for coord in xyxy[0:4]:
                coords.append(str(coord))
              current_box = ";".join(coords)
              lane_numbers[current_box] = (lane_number, points_to_graph)

            for xyxy in detections_list:
                coords = []
                for coord in xyxy[0:4]:
                  coords.append(str(coord))
                current_box = ";".join(coords)
                if current_box in lane_numbers:
                  lane_number, points_to_graph = lane_numbers[current_box]
                else:
                  lane_number, points_to_graph = get_lane(xyxy, ll_seg_mask, inp_center=frame_center_mean)
                
                if not opt.save_csv is None:
                  if vid_cap.get(cv2.CAP_PROP_POS_FRAMES) >= 0:
                    if not current_frame_number in csv_data[os.path.basename(path)]:
                      csv_data[os.path.basename(path)][current_frame_number] = []
                    detection_string = f'{lane_number} {xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]} {xyxy[4]} {frame_center_mean}'
                    if json_data[0] != None and len(xyxy) >= 6:
                      detection_string += f' {xyxy[5]}'
                    csv_data[os.path.basename(path)][current_frame_number].append(detection_string)

                #label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                if not lane_number is None:
                  #label_det_pred += f' {lane_number}'
                  label_det_pred = f'{lane_number}'
                else:
                  label_det_pred = None
                
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[0], line_thickness=2)
                
                # draw points from lane detection
                for lane_edge in points_to_graph:
                  img_det = cv2.circle(img_det, lane_edge, radius=3, color=(0,255,0), thickness=-1)
                
        #print(csv_data)
        if opt.save_dir != '':
          save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")
          if dataset.mode == 'images':
              cv2.imwrite(save_path,img_det)

          elif dataset.mode == 'video':
              if vid_path != save_path:  # new video
                  vid_path = save_path

                  if isinstance(vid_writer, cv2.VideoWriter):
                      vid_writer.release()  # release previous video writer

                  fourcc = 'mp4v'  # output video codec
                  fps = vid_cap.get(cv2.CAP_PROP_FPS)
                  h,w,_=img_det.shape
                  vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
              vid_writer.write(img_det)
          
          else:
              cv2.imshow('image', img_det)
              cv2.waitKey(1)  # 1 millisecond
      


        new_csv_path = os.path.join(opt.save_csv, os.path.basename(path) + ".csv")
        with open(new_csv_path, "w", newline='') as new_csv:
          csv_writer = csv.writer(new_csv)
          for row in sorted(csv_data[os.path.basename(path)].keys()):
            csv_writer.writerow([row] + csv_data[os.path.basename(path)][row])



    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--detect-fps', type=int, default=None, help='fps to perform detections at')
    parser.add_argument('--img-center', type=int, default=None, help='Manually add a center')
    parser.add_argument('--save-csv', type=str, default=None, help='path to save csv output')
    parser.add_argument('--external-boxes', type=str, default=None, help='path to externally generated bounding boxes')
    parser.add_argument('--enforce-external', default=False, help='enforce externally generated bounding boxes')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
