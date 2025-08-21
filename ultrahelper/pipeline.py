
from pathlib import Path
import cv2
import time
import torch
import queue
import numpy as np
from multiprocessing import Process, Queue
from ultrahelper.load import load_deployment_model, load_postprocessor
from ultralytics.utils.ops import non_max_suppression
import threading




# I/O Path
current_dir = Path(__file__).resolve().parent
SOURCE     = str(current_dir / "IMG_5692.mov")
OUTPUT_DIR = current_dir / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = str(OUTPUT_DIR / "output.mp4")




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


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(o) for o in obj)
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    else:
        return obj


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
        strides = hw_model.model[-1].stride.cpu().tolist()
        result_q.put((frame, to_cpu(feats), gpu_ms, strides))


# ─── worker #2: CPU/postprocessing model ───
def _postprocessor_worker(result_q: Queue, img_q: Queue):
    pp_model = load_postprocessor()
    prev_time = time.time()

    while True:
        item = result_q.get()
        if item is None:
            img_q.put(None)
            break

        frame, feats_cpu, gpu_ms, feat_stride = item

        t0 = time.time()
        with torch.no_grad():
            dets, raw_kpts = pp_model(feats_cpu)
        cpu_ms = (time.time() - t0) * 1000

        # ─── comment out all keypoint & box drawing ───
        # if dets is not None and len(dets[0]):
        #     # decode and draw...
        #     pass

        # compute actual FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # just print performance stats
        print(f"FPS: {fps:.1f}, GPU: {gpu_ms:.1f}ms, CPU: {cpu_ms:.1f}ms")

        img_q.put(frame)


class ParallelInferencePipeline:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.source      = SOURCE
        self.output_path = OUTPUT_PATH
        self.frame_q     = Queue(maxsize=4)
        self.result_q    = Queue(maxsize=4)
        self.img_q       = Queue(maxsize=4)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps_input,
            (w, h)
        )

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
            # cv2.imshow('Output', processed)      # optional display
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.frame_q.put(None)
                break

        hw.join(); pp.join()
        cap.release(); writer.release()
        cv2.destroyAllWindows()
        post = Postprocessor()
        post.run(SOURCE)