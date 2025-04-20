from pathlib import Path
import cv2
import time
import torch
import numpy as np
from multiprocessing import Process, Queue
from ultrahelper.load import load_deployment_model, load_postprocessor
from ultralytics.utils.ops import non_max_suppression
import math
import queue
import threading

# I/O Path
current_dir = Path(__file__).resolve().parent
SOURCE     = str(current_dir / "IMG_5692.mov")
OUTPUT_DIR = current_dir / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = str(OUTPUT_DIR / "output.mp4")

ultrahelper_root = current_dir

class Postprocessor:
    def __init__(self):
        # a small queue of frames for display
        self.frame_queue = queue.Queue(maxsize=2)
        # ensure ultrahelper is on the path
        self.current_dir = current_dir

    def _tracker_worker(self, source):
        # load CPU postprocessor
        model = load_postprocessor()

        # stream=True yields one `r` at a time
        for r in model.tracker(source=source, save=False, stream=True):
            frame = r.plot()  # annotated BGR numpy array
            try:
                # drop frames if we fall behind
                self.frame_queue.put(frame, timeout=0.01)
            except queue.Full:
                pass

    def run(self, source=1):
        """
        source: integer index for webcam (e.g. 0 or 1) or path string for a file
        """
        # start the tracker thread
        tracker_thread = threading.Thread(
            target=self._tracker_worker,
            args=(source,),
            daemon=True
        )
        tracker_thread.start()

        # main display loop (must run on main thread on macOS)
        try:
            while True:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    cv2.imshow(f"Postprocessor ({source})", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            tracker_thread.join(timeout=0.1)


# ─── drawing utilities ───
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8,10): 'c', (5, 6): 'y', (5,11): 'm',
    (6,12): 'c', (11,12): 'y', (11,13): 'm', (13,15): 'm',
    (12,14): 'c', (14,16): 'c'
}


# shaped now has shape [K, 3], with y,x in pixel coords and confidence unchanged
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


# ─── helper to move tensors to CPU ───
def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(o) for o in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    else:
        return obj

# ─── decode raw heatmaps → [(x_norm,y_norm,conf), …] ───
def decode_keypoint_heatmaps(head_feats):
    B, num_ch, N = head_feats.shape
    K = num_ch // 3
    H = W = int(math.sqrt(N))
    feats = head_feats.squeeze(0)
    keypoints = []
    for i in range(K):
        x_ch = feats[3*i + 0]
        y_ch = feats[3*i + 1]
        c_ch = feats[3*i + 2]
        idx  = int(c_ch.argmax().item())
        conf = float(c_ch[idx].item())
        y_i, x_i = divmod(idx, W)
        keypoints.append((x_i/(W-1), y_i/(H-1), conf))
    return [keypoints]

# ─── worker #1: GPU/deployment model ───
def _hardware_worker(frame_q: Queue, result_q: Queue, device: torch.device):
    hw_model = load_deployment_model().to(device)
    while True:
        frame = frame_q.get()
        if frame is None:
            result_q.put(None)
            break
        img = cv2.resize(frame, (640, 640))
        tensor = (torch.from_numpy(img)
                      .permute(2,0,1)
                      .unsqueeze(0)
                      .float()
                      .div(255.0)
                      .to(device))
        t0 = time.time()
        with torch.no_grad():
            feats = hw_model(tensor)
        gpu_ms = (time.time() - t0) * 1000
        result_q.put((frame, to_cpu(feats), gpu_ms))

# ─── worker #2: CPU/postprocessing model + drawing ───
def _postprocessor_worker(result_q: Queue, img_q: Queue):
    pp_model = load_postprocessor()
    while True:
        item = result_q.get()
        if item is None:
            img_q.put(None)
            break

        frame, feats_cpu, gpu_ms = item
        t0 = time.time()
        with torch.no_grad():
            dets, raw_kpts = pp_model(feats_cpu)    # raw_kpts: [1,3K,N]
        cpu_ms = (time.time() - t0) * 1000

        # ─── run NMS to get boxes ───
        preds = [x.flatten(2).permute(0,2,1) for x in dets]
        pred  = torch.cat(preds, dim=1)                          
        boxes = non_max_suppression(pred, 0.25, 0.65)[0]

        # ─── only if at least one box survived ───
        if boxes is not None and len(boxes):
            # a) decode heatmaps → list of [(x_norm,y_norm,conf),…]
            kpt_list = decode_keypoint_heatmaps(raw_kpts)    
            norm_kps = np.array(kpt_list[0])                 # shape (K,3)

            # b) swap to (y_norm, x_norm, conf) for draw fns
            kps = norm_kps[:, [1, 0, 2]]                     # (K,3)

            # c) draw skeleton & keypoints
            draw_connections(frame, kps, EDGES, 0)
            draw_keypoints   (frame, kps, 0)

            # d) draw boxes
            h_px, w_px = frame.shape[:2]
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                cv2.putText(frame, f"{conf:.6f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

            
            for *xyxy, conf, cls in boxes.tolist():
                x1n, y1n, x2n, y2n = xyxy
                if 0.0 <= x2n <= 1.0:
                    x1, y1 = int(x1n * w_px), int(y1n * h_px)
                    x2, y2 = int(x2n * w_px), int(y2n * h_px)
                else:
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,100), 2)
                cv2.putText(frame, f"{conf:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)


        # ─── overlay performance & emit ───
        fps = 1000.0 / (gpu_ms + cpu_ms)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"GPU: {gpu_ms:.1f}ms", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"CPU: {cpu_ms:.1f}ms", (10,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        img_q.put(frame)




# ─── pipeline orchestrator ───
class ParallelInferencePipeline:
    def __init__(self,):
        queue_size=4
        # pick best device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.source      = SOURCE
        self.output_path = OUTPUT_PATH
        self.frame_q     = Queue(maxsize=queue_size)
        self.result_q    = Queue(maxsize=queue_size)
        self.img_q       = Queue(maxsize=queue_size)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        hw = Process(target=_hardware_worker,
                     args=(self.frame_q, self.result_q, self.device),
                     daemon=True)
        pp = Process(target=_postprocessor_worker,
                     args=(self.result_q, self.img_q),
                     daemon=True)
        hw.start(); pp.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                self.frame_q.put(None)
                break
            self.frame_q.put(frame)
            processed = self.img_q.get()
            if processed is None:
                break
            writer.write(processed)
            cv2.imshow('Output', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.frame_q.put(None)
                break

        hw.join(); pp.join()
        cap.release(); writer.release()
        cv2.destroyAllWindows()
        post = Postprocessor()
        post.run(SOURCE)


















