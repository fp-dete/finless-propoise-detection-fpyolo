# limit the number of cpus used by high performance libraries
import os
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

import sys

sys.path.insert(0, './yolov5')
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                                  colorstr, cv2,
                                  increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer,
                                  xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box


def detect(opt):
    out, source, yolo_model, deep_sort_model, project, name, show_vid, save_vid, save_txt, imgsz, evaluate, half, exist_ok = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.project, opt.name, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.exist_ok

    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.MODEL_TYPE,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    (save_dir / 'yolo_detections').mkdir(parents=True, exist_ok=True)
    (save_dir / 'tracking_results').mkdir(parents=True, exist_ok=True)
    (save_dir / 'labeled_tracking').mkdir(parents=True, exist_ok=True)

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    if webcam:
        txt_file_name = 'webcam'
    else:
        source_path = Path(source)
        if source_path.is_file():
            txt_file_name = source_path.stem
        else:
            txt_file_name = source_path.name

    txt_path = save_dir / 'tracks' / f'{txt_file_name}.txt'
    video_save_path = save_dir / f'{txt_file_name}.mp4'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    dt, seen = [0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string

            # 创建三个图片保存路径
            yolo_detection_path = save_dir / 'yolo_detections' / f'{p.stem}_{frame_idx:06d}.jpg'
            tracking_result_path = save_dir / 'tracking_results' / f'{p.stem}_{frame_idx:06d}.jpg'
            labeled_tracking_path = save_dir / 'labeled_tracking' / f'{p.stem}_{frame_idx:06d}.jpg'

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            has_detections = False
            has_tracking_results = False

            if det is not None and len(det):
                has_detections = True
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                if len(det) > 0:
                    yolo_im0 = im0.copy()
                    yolo_annotator = Annotator(yolo_im0, line_width=2, pil=not ascii)

                    for *xyxy, conf, cls in det:
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        yolo_annotator.box_label(xyxy, label, color=colors(c, True))

                    cv2.imwrite(str(yolo_detection_path), yolo_annotator.result())

                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()

                if len(outputs) > 0:
                    has_tracking_results = True
                    cv2.imwrite(str(tracking_result_path), im0)

                if len(outputs) > 0:
                    labeled_im0 = im0.copy()
                    labeled_annotator = Annotator(labeled_im0, line_width=2, pil=not ascii)

                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        labeled_annotator.box_label(bboxes, label, color=colors(c, True))

                    cv2.imwrite(str(labeled_tracking_path), labeled_annotator.result())

                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
            else:
                deepsort.increment_ages()
                if save_txt:
                    with open(txt_path, 'a') as f:
                        f.write(f'{frame_idx + 1} -1 -1 -1 -1 -1 -1 -1 -1 -1\n')
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s)')

            if not has_detections and len(deepsort.tracker.tracks) > 0:
                active_tracks = [track for track in deepsort.tracker.tracks if
                                 track.is_confirmed() or track.time_since_update < 1]
                if len(active_tracks) > 0:
                    cv2.imwrite(str(tracking_result_path), im0)
                    labeled_im0 = im0.copy()
                    labeled_annotator = Annotator(labeled_im0, line_width=2, pil=not ascii)
                    for track in active_tracks:
                        if track.is_confirmed() or track.time_since_update < 1:
                            bbox = track.to_tlbr()
                            id = track.track_id
                            cls = track.get_class() if hasattr(track, 'get_class') else 0

                            c = int(cls)  # integer class
                            label = f'{id} {names[c]}'
                            labeled_annotator.box_label(bbox, label, color=colors(c, True))

                    cv2.imwrite(str(labeled_tracking_path), labeled_annotator.result())

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != str(video_save_path):  # new video
                    vid_path = str(video_save_path)
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(str(video_save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''

        yolo_detection_count = len(list(save_dir.glob('yolo_detections/*.jpg')))
        tracking_result_count = len(list(save_dir.glob('tracking_results/*.jpg')))
        labeled_tracking_count = len(list(save_dir.glob('labeled_tracking/*.jpg')))

        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        LOGGER.info(f"YOLO detection frames: {yolo_detection_count}")
        LOGGER.info(f"Tracking result frames (no labels): {tracking_result_count}")
        LOGGER.info(f"Labeled tracking frames: {labeled_tracking_count}")

        if platform == 'darwin':  # MacOS
            os.system('open ' + str(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='', help='source')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default=False, action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', default=False, action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=10, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
