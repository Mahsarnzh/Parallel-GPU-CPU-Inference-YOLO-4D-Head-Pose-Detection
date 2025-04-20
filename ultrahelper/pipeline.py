from pathlib import Path

# this file’s directory
BASE_DIR = Path(__file__).resolve().parent

# video file sits next to this script
source      = BASE_DIR / "IMG_5692.mov"

# outputs/ subfolder next to this script
output_dir  = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)  # create if missing

# if you need them as strings:
source = str(source)
output_path = str(output_dir)


import cv2
import time
import torch
from multiprocessing import Process, Queue
from ultrahelper.load import load_deployment_model, load_postprocessor
from ultralytics.utils.ops import non_max_suppression
import math 

COCO_SKELETON = [
    (0, 1),  # nose → left eye
    (0, 2),  # nose → right eye
    (1, 3),  # left eye → left ear
    (2, 4),  # right eye → right ear
    (5, 6),  # left shoulder → right shoulder
    (5, 7),  # left shoulder → left elbow
    (7, 9),  # left elbow → left wrist
    (6, 8),  # right shoulder → right elbow
    (8,10),  # right elbow → right wrist
    (5,11),  # left shoulder → left hip
    (6,12),  # right shoulder → right hip
    (11,12), # left hip → right hip
    (11,13), # left hip → left knee
    (13,15), # left knee → left ankle
    (12,14), # right hip → right knee
    (14,16)  # right knee → right ankle
]

def to_cpu(obj):
    """
    Recursively move any torch.Tensor inside obj to CPU for pickling.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(o) for o in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    else:
        return obj
    

def draw_detections(frame, dets, threshold=0.5):
    """
    dets: Tensor[N, 6]  →  x1,y1,x2,y2,conf,class_id
    """
    # print(len(arr).shape)
    arr = dets
    # arr = arr.reshape(-1, arr.shape[-1])
    # print(len(arr[0][0]))    # -- > 1 --> preds
    # print(len(arr[1][0]))   ##--> 51
    # print(len(arr[0][1]))  # --> 1
    # print(len(arr[0]))  # -- > 3
    # print(len(arr))    # --> 2
    # print(len(arr[0])) #--> 17 --> decode_keypoint_heatmaps
    
    
    arr = dets.numpy()
    if arr.ndim == 1:                  # single box → make it batch
        arr = arr[None, :]

    for row in arr:
        x1, y1, x2, y2, conf = row[:5]
        if conf < threshold:
            continue
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame,
                    f"{conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    1)
    return frame
def decode_keypoint_heatmaps(head_feats, threshold=0.5):
    """
    head_feats: Tensor of shape [1, 3*K, N] where K=num_keypoints, N=H*W
    Returns: a Python list [[(x1,y1,c1), …, (xK,yK,cK)]] for the batch
    """
    B, num_ch, N = head_feats.shape        # B==1, num_ch==3*K, N==H*W
    K = num_ch // 3
    # assume square grid:
    H = W = int(math.sqrt(N))

    feats = head_feats.squeeze(0)          # now [3*K, N]
    keypoints = []

    for i in range(K):
        x_ch = feats[3*i + 0]            # [N]
        y_ch = feats[3*i + 1]
        c_ch = feats[3*i + 2]

        # index of maximum confidence
        idx = int(c_ch.argmax().item())
        conf = float(c_ch[idx].item())

        # map flat idx → grid coords
        y_idx = idx // W
        x_idx = idx %  W

        # normalize to [0,1]
        x_norm = x_idx / (W - 1)
        y_norm = y_idx / (H - 1)

        keypoints.append((x_norm, y_norm, conf))

    return [keypoints]  # batch style


def _hardware_worker(frame_q: Queue, result_q: Queue, device: torch.device):
    hw_model = load_deployment_model().to(device)
    while True:
        frame = frame_q.get()
        if frame is None:
            # signal postprocessor to exit, then stop
            result_q.put(None)
            break

        img = cv2.resize(frame, (640, 640))
        tensor = (
            torch.from_numpy(img)
                 .permute(2,0,1)
                 .unsqueeze(0)
                 .float()
                 .div(255.0)
                 .to(device)
        )

        t0 = time.time()
        with torch.no_grad():
            feats = hw_model(tensor)
        gpu_latency = (time.time() - t0)*1000  # ms

        # move everything to CPU so it can be sent over the queue
        result_q.put((frame, to_cpu(feats), gpu_latency))

def _postprocessor_worker(result_q: Queue, IMG_Q: Queue):
    pp_model = load_postprocessor()
    while True:
        item = result_q.get()
        if item is None:
            IMG_Q.put(None)
            break

        frame, feats_cpu, gpu_ms = item

        t0 = time.time()
        with torch.no_grad():
            dets, raw_kpts = pp_model(feats_cpu)
            # print('len(dets), len(raw_kpts)', len(dets), len(raw_kpts[0]))

        h, w = frame.shape[:2]  # e.g. both 640 if you resized to 640×640

        # 3) Decode heatmaps → list of 17 (x,y,conf) tuples
        kpt_list = decode_keypoint_heatmaps(raw_kpts, threshold=0.65)

        h_px, w_px = frame.shape[:2]
        pts = kpt_list[0]  # list of K tuples

        for i1, i2 in COCO_SKELETON:
            x1n, y1n, c1 = pts[i1]
            x2n, y2n, c2 = pts[i2]
            if c1 >= 0.1 and c2 >= 0.1:
                # scale into pixels
                pt1 = (int(x1n * w_px), int(y1n * h_px))
                pt2 = (int(x2n * w_px), int(y2n * h_px))
                cv2.line(frame, pt1, pt2, (100, 0 , 255), 2)





        preds = [x.flatten(2).permute(0, 2, 1) for x in dets]

        # 2) concat → (1, 9600, 65)
        pred = torch.cat(preds, dim=1)
        boxes = non_max_suppression(pred,
                                conf_thres=0.1,
                                iou_thres=0.1)[0]

       
        h_px, w_px = frame.shape[:2]

        if boxes is not None:
            for *xyxy, conf, cls in boxes.tolist():
                x1_norm, y1_norm, x2_norm, y2_norm = xyxy

                # detect whether coords are normalized (≤1) or already in pixels (>1)
                if 0.0 <= x1_norm <= 1.0 and 0.0 <= y1_norm <= 1.0 and \
                0.0 <= x2_norm <= 1.0 and 0.0 <= y2_norm <= 1.0:
                    # scale to pixel coords
                    x1 = int(x1_norm * w_px)
                    y1 = int(y1_norm * h_px)
                    x2 = int(x2_norm * w_px)
                    y2 = int(y2_norm * h_px)
                else:
                    # already pixel coords
                    x1, y1, x2, y2 = map(int, (x1_norm, y1_norm, x2_norm, y2_norm))

                # draw rectangle and optional confidence
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 10)
                cv2.putText(
                    frame,
                    f"{conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    10
                )

        IMG_Q.put(frame)

        # 4) Draw keypoints
        cpu_latency = (time.time() - t0)*1000  # ms
        fps = 1/ (cpu_latency + gpu_ms)
        # result_q.put((frame, to_cpu(feats), gpu_ms))
    #     # 6) Overlay metrics
        cv2.putText(frame, f"FPS: {fps:.8f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"GPU: {gpu_ms:.8f}ms", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"CPU: {cpu_latency:.8f}ms", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cv2.destroyAllWindows()

class ParallelInferencePipeline:
    def __init__(self, source=0, queue_size=4):
        # pick best device: CUDA → MPS → CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.source   = source
        self.frame_q  = Queue(maxsize=queue_size)
        self.result_q = Queue(maxsize=queue_size)
        self.IMG_Q = Queue(maxsize=queue_size)

    def run(self):
        # launch GPU and CPU workers
        cap = cv2.VideoCapture(source)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        hw = Process(target=_hardware_worker,
                     args=(self.frame_q, self.result_q, self.device),
                     daemon=True)
        pp = Process(target=_postprocessor_worker,
                     args=(self.result_q, self.IMG_Q),
                     daemon=True)
        hw.start()
        pp.start()

    
        while True:
            ret, frame = cap.read()
            if not ret:
                self.frame_q.put(None)
                break
            self.frame_q.put(frame)
            processed = self.IMG_Q.get()
            if processed is None:
                break
            writer.write(processed)
            cv2.imshow('Output', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.frame_q.put(None)
                break
        hw.join()
        pp.join()
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    













