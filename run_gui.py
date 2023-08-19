import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import argparse
import pandas as pd
import os
# (the rest of your imports)
import argparse
import pandas as pd
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import time
# from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov7 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
    
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box,put_line_middle
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

from PyTorch_UnderwaterImageEnhancement.model import PhysicalNN
from torchvision import transforms


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Add your YOLOv7 implementation and classes here

class YOLOv7App:
    def __init__(self, root):
        self.root = root
        self.model = None

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.btn_open_video = tk.Button(root, text="Open Video", command=self.open_video)
        self.btn_open_video.pack()

        self.btn_start = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.btn_start.pack()

        self.btn_quit = tk.Button(root, text="Quit", command=self.quit)
        self.btn_quit.pack()

        self.video_source = 0
        self.cap = None
        self.is_detecting = False

    def open_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_path:
            self.video_source = file_path

    def start_detection(self):
        if self.model is None:
            
            # Load YOLOv7 model
            # Implement the loading of YOLOv7 model here
            pass

        self.cap = cv2.VideoCapture(self.video_source)
        self.is_detecting = True
        self.detect_objects()

    def stop_detection(self):
        self.is_detecting = False

    def detect_objects(self):
        if not self.is_detecting:
            return
        yolo_weights = Path(ROOT,"best.pt")
        strong_sort_weights = "osnet_x0_25_msmt17.pt"
        config_strongsort = "strong_sort/configs/strong_sort.yaml"
        imgsz = (640, 640)
        conf_thres = 0.25
        iou_thres = 0.45
        max_det = 1000
        device = "0"
        show_vid = True
        save_txt = False
        save_conf = False
        save_crop = False
        save_vid = False
        nosave = False
        classes = None
        agnostic_nms = False
        augment = False
        visualize = False
        update = False
        project = "runs/track"
        name = "exp"
        exist_ok = False
        line_thickness = 2
        hide_labels = False
        hide_conf = False
        hide_class = False
        half = False
        dnn = False
        source = 0
        ret, frame = self.cap.read()
        if ret:
            source = str(source)
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            is_file = Path(source).suffix[1:] in (VID_FORMATS)
            is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
            webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
            if is_url and is_file:
                source = check_file(source)  # download

            # Directories
            if not isinstance(yolo_weights, list):  # single yolo model
                exp_name = yolo_weights.stem
            elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
                exp_name = Path(yolo_weights[0]).stem
                yolo_weights = Path(yolo_weights[0])
            else:  # multiple models after --yolo_weights
                exp_name = 'ensemble'
            exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
            save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
            save_dir = Path(save_dir)
            (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

            # Load model
            device = select_device(device)
            
            WEIGHTS.mkdir(parents=True, exist_ok=True)
            model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model
            names, = model.names,
            stride = model.stride.max()  # model stride
            imgsz = check_img_size(imgsz[0], s=stride.cpu().numpy())  # check image size

            # Dataloader
            if webcam:
                show_vid = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride.cpu().numpy())
                nr_sources = 1
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride)
                nr_sources = 1
            vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

            # initialize StrongSORT
            cfg = get_config()
            cfg.merge_from_file(config_strongsort)
            
            # ImageEnhancement
            # Load model
            # checkpoint = "./PyTorch_UnderwaterImageEnhancement/checkpoints/model_best_2842.pth.tar"
            # model_enc = PhysicalNN()
            # model_enc = torch.nn.DataParallel(model_enc).to(device)
            # print("=> loading trained model")
            # checkpoint = torch.load(checkpoint, map_location=device)
            # model_enc.load_state_dict(checkpoint['state_dict'])
            # print("=> loaded model at epoch {}".format(checkpoint['epoch']))
            # model_enc = model_enc.module
            # model_enc.eval()

            # testtransform = transforms.Compose([
            #             transforms.ToTensor(),
            #         ])
            # unloader = transforms.ToPILImage()


            # Create as many strong sort instances as there are video sources
            strongsort_list = []
            for i in range(nr_sources):
                strongsort_list.append(
                    StrongSORT(
                        strong_sort_weights,
                        device,
                        half,
                        max_dist=cfg.STRONGSORT.MAX_DIST,
                        max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.STRONGSORT.MAX_AGE,
                        n_init=cfg.STRONGSORT.N_INIT,
                        nn_budget=cfg.STRONGSORT.NN_BUDGET,
                        mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                        ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

                    )
                )
                strongsort_list[i].model.warmup()
            outputs = [None] * nr_sources
            
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run tracking
            dt, seen = [0.0, 0.0, 0.0, 0.0], 0
            curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
            t0 = time.time()
            graph_fps = []
            for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
                s = ''
                t1 = time_synchronized()
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_synchronized()
                dt[0] += t2 - t1

                # Inference
                visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
                pred = model(im)
                t3 = time_synchronized()
                dt[1] += t3 - t2

                # Apply NMS
                pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)
                dt[2] += time_synchronized() - t3
                
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    seen += 1
                    if webcam:  # nr_sources >= 1
                        p, im0, _ = path[i], im0s[i].copy(), dataset.count
                        p = Path(p)  # to Path
                        s += f'{i}: '
                        txt_file_name = p.name
                        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                    else:
                        p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                        p = Path(p)  # to Path
                        # video file
                        if source.endswith(VID_FORMATS):
                            txt_file_name = p.stem
                            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                        # folder with imgs
                        else:
                            txt_file_name = p.parent.name  # get folder name containing current img
                            save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                            
                    # inp = testtransform(im0).unsqueeze(0)
                    # inp = inp.to(device)
                    # out = model_enc(inp)
                    # corrected = unloader(out.cpu().squeeze(0))
                    # im0 = np.array(corrected)
                    curr_frames[i] = im0

                    txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    imc = im0.copy() if save_crop else im0  # for save_crop

                    if cfg.STRONGSORT.ECC:  # camera motion compensation
                        strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        
                        xywhs = xyxy2xywh(det[:, 0:4])
                        confs = det[:, 4]
                        clss = det[:, 5]

                        # pass detections to strongsort
                        t4 = time_synchronized()
                        outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                        t5 = time_synchronized()
                        dt[3] += t5 - t4
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # draw boxes for visualization
                        if len(outputs[i]) > 0:
                            for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                                bboxes = output[0:4]
                                cv2.circle(img=im0,center=(int((bboxes[0]+bboxes[2])/2),int((bboxes[1]+bboxes[3])/2)),radius=3,color=(0, 0, 255), thickness=1)
                                id = output[4]
                                cls = output[5]
                                if save_txt:
                                    # to MOT format
                                    bbox_left = output[0]
                                    bbox_top = output[1]
                                    bbox_w = output[2] - output[0]
                                    bbox_h = output[3] - output[1]
                                    # Write MOT compliant results to file
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                                if save_vid or save_crop or show_vid:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    id = int(id)  # integer id
                                    label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                        (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                    plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=2)
                                    
                                    if save_crop:
                                        txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                        save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                        # box_left = [x for x in outputs[i][0:4] if (int(x[0]+x[2]/2) <= ((im0.shape[1]/2)+75))]
                        # box_right = [x for x in outputs[i][0:4] if (int(x[0]+x[2]/2) > ((im0.shape[1]/2)+75))]
                        # total = len(box_left)+len(box_right)
                        # cv2.putText(im0, f'KIRI : '+str(len(box_left)),(480,60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255),2)
                        # cv2.putText(im0, f'KANAN : '+str(len(box_right)), (480,85), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255),2)
                        # cv2.putText(im0, f'TOTAL : '+str(total), (480,110), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255),2)
                    
                    else:
                        strongsort_list[i].increment_ages() 
                        print('No detections')
                    
                    if dataset.mode != 'image':
                        currentTime= time.time()
                        
                        fps = 1/(currentTime-t0)
                        t0= currentTime
                        total_det = str(0) if len(det) == 0 else str(len(det))
                        cv2.putText(im0, f'FPS : '+str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)
                        cv2.putText(im0,f"Total : {total_det}",(im0.shape[1]-190,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)   
                        graph_fps.append([total_det,fps])
                        # put_line_middle(im0,im0.shape)
                        # if outputs[i] is None:
                        #     box_left = []
                        #     box_right = []
                        # else :
                        #     box_left = [x for x in outputs[i] if (int(x[0]+x[2]/2) <= ((im0.shape[1]/2)+75))]
                        #     box_right = [x for x in outputs[i] if (int(x[0]+x[2]/2) > ((im0.shape[1]/2)+75))]
                        # total = len(box_left)+len(box_right)
                        # cv2.putText(im0, f'KIRI : '+str(len(box_left)),(480,60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255),2)
                        # cv2.putText(im0, f'KANAN : '+str(len(box_right)), (480,85), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255),2)
                        # cv2.putText(im0, f'TOTAL : '+str(total), (480,110), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255),2)
                        
                    # Stream resultss
                    if show_vid:
                        cv2.imshow(f"Layar Monitoring", im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_vid:
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
                
                    prev_frames[i] = curr_frames[i]
                    # end_time = time.time()
                df = pd.DataFrame(graph_fps,columns=['num_detect','fps'])
                df.to_csv('hasil_fps_akhir.csv',index=False)
                # cv2.putText(im0,str(f"FPS : {1/(time.time()-t0):.1f}"),(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
            if save_txt or save_vid:
                s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
                print(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
                
            
            # Perform YOLOv7 object detection on the frame
            # Implement the YOLOv7 object detection function here

            # Convert the frame to an ImageTk.PhotoImage
            image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image=image)

            # Update the Tkinter canvas with the new frame
            self.canvas.create_image(0, 0, image=image_tk, anchor=tk.NW)
            self.canvas.image = image_tk

        # Schedule the next detection if self.is_detecting is still True
        if self.is_detecting:
            self.root.after(30, self.detect_objects)
        else:
            # Stop the video capture when detection is stopped
            self.cap.release()

    def quit(self):
        self.is_detecting = False
        self.root.quit()

if __name__ == "__main__":
    # opt = parse_opt()
    root = tk.Tk()
    root.title("LoCo")
    app = YOLOv7App(root)
    root.mainloop()





# class YOLOTrackingApp(tk.Tk):
#     def __init__(self):
#         super().__init__()

#         self.video_source = 0
#         self.vid = cv2.VideoCapture(self.video_source)
#         self.title("YOLO Tracking App")
#         self.geometry("1000x1000")

#         # Create and place widgets
#         self.create_widgets()

#     def update(self):
#         if self.is_playing:
#             ret, frame = self.vid.read()
#             if ret:
#                 self.photo = self.convert_frame(frame)
#                 self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
#             else:
#                 # Video ended, stop playing
#                 self.is_playing = False
#                 self.btn_toggle.config(text="Play")
#         self.root.after(30, self.update)
    
#     def update(self):
#         if self.is_playing:
#             ret, frame = self.vid.read()
#             if ret:
#                 self.photo = self.convert_frame(frame)
#                 self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
#             else:
#                 # Video ended, stop playing
#                 self.is_playing = False
#                 self.btn_toggle.config(text="Play")
#         self.root.after(30, self.update)

#     def create_widgets(self):
#         # Labels
#         self.source_label = tk.Label(self, text="Source (file/dir/URL/glob, 0 for webcam):")
#         self.source_label.pack()

#         # Source Entry
#         self.source_entry = tk.Entry(self, width=50)
#         self.source_entry.pack()

#         # Buttons
#         self.browse_button = tk.Button(self, text="Browse", command=self.browse_source)
#         self.browse_button.pack()

#         self.run_button = tk.Button(self, text="Run Tracking", command=self.run_tracking)
#         self.run_button.pack()
        
#         self.btn_stop = tk.Button(self, text="Stop", command=self.stop_video)
#         self.btn_stop.pack()

#     def stop_video(self):
#         self.is_playing = False
#         self.btn_toggle.config(text="Play")
#         self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
#     def browse_source(self):
#         filename = filedialog.askopenfilename()
#         if filename:
#             self.source_entry.delete(0, tk.END)
#             self.source_entry.insert(tk.END, filename)

#     def run_tracking(self):
#         try:
#             source = self.source_entry.get()

#             # Collect other parameters from GUI
#             yolo_weights = Path(ROOT,"best.pt")
#             strong_sort_weights = "osnet_x0_25_msmt17.pt"
#             config_strongsort = "strong_sort/configs/strong_sort.yaml"
#             imgsz = (640, 640)
#             conf_thres = 0.25
#             iou_thres = 0.45
#             max_det = 1000
#             device = "0"
#             show_vid = True
#             save_txt = False
#             save_conf = False
#             save_crop = False
#             save_vid = False
#             nosave = False
#             classes = None
#             agnostic_nms = False
#             augment = False
#             visualize = False
#             update = False
#             project = "runs/track"
#             name = "exp"
#             exist_ok = False
#             line_thickness = 2
#             hide_labels = False
#             hide_conf = False
#             hide_class = False
#             half = False
#             dnn = False

#             # Run the tracking function
#             run(
#                 source=source,
#                 yolo_weights=yolo_weights,
#                 strong_sort_weights=strong_sort_weights,
#                 config_strongsort=config_strongsort,
#                 imgsz=imgsz,
#                 conf_thres=conf_thres,
#                 iou_thres=iou_thres,
#                 max_det=max_det,
#                 device=device,
#                 show_vid=show_vid,
#                 save_txt=save_txt,
#                 save_conf=save_conf,
#                 save_crop=save_crop,
#                 save_vid=save_vid,
#                 nosave=nosave,
#                 classes=classes,
#                 agnostic_nms=agnostic_nms,
#                 augment=augment,
#                 visualize=visualize,
#                 update=update,
#                 project=project,
#                 name=name,
#                 exist_ok=exist_ok,
#                 line_thickness=line_thickness,
#                 hide_labels=hide_labels,
#                 hide_conf=hide_conf,
#                 hide_class=hide_class,
#                 half=half,
#                 dnn=dnn
#             )

#             messagebox.showinfo("YOLO Tracking App", "Tracking Completed!")

#         except Exception as e:
#             messagebox.showerror("YOLO Tracking App", str(e))


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
#     parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
#     parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
#     parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

#     return opt


# def main(opt):
#     check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
#     run(**vars(opt))


# # if __name__ == "__main__":
# #     opt = parse_opt()
# #     app = YOLOTrackingApp()
# #     app.mainloop()
#     # main(opt)
