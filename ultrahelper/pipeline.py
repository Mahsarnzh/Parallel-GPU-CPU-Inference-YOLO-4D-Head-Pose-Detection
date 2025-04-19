# import multiprocessing
# import time
# import cv2
# import torch
# from ultrahelper.load import load_deployment_model, load_postprocessor

# class ParallelInferencePipeline:
#     def __init__(self):
#         self.return_dict = {}
#         self.initial_time = time.time()

#     def hardware_worker(self, frame_queue, feat_queue):
#         """ This worker runs on the GPU for model inference. """
#         model = load_deployment_model()  # Load the hardware (GPU) model inside the worker
#         cap = cv2.VideoCapture(0)  # Use webcam for input
#         if not cap.isOpened():
#             print("[ERROR] Webcam not accessible")
#             return

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 continue

#             # Preprocess the frame
#             img = cv2.resize(frame, (640, 640))
#             img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

#             start_time = time.time()
#             with torch.no_grad():
#                 feats = model(img_tensor)  # Only the pose head
#             inference_time = time.time() - start_time

#             # Put the feature and frame into the feature queue for postprocessing
#             feat_queue.put((feats, frame))
#             frame_rate = 1 / inference_time  # FPS calculation
#             print(f"GPU FPS: {frame_rate:.2f}, Inference Latency: {inference_time:.4f}s")

#     def cpu_worker(self, feat_queue, result_queue):
#         """ This worker runs on the CPU for postprocessing the results. """
#         postprocessor = load_postprocessor()  # Load the postprocessor (CPU)

#         while True:
#             feats, frame = feat_queue.get()  # Get the features from the feature queue

#             start_time = time.time()
#             processed_output = postprocessor(feats)  # Process the features
#             postprocess_time = time.time() - start_time

#             # Put the processed output into the result queue
#             result_queue.put(processed_output)

#             print(f"CPU Postprocessing Time: {postprocess_time:.4f}s")

#     def run(self):
#         """Main method to run the parallel pipeline using multiprocessing and threading."""
#         # Queues for passing data between the workers
#         frame_queue = multiprocessing.Queue(maxsize=1)
#         feat_queue = multiprocessing.Queue(maxsize=1)
#         result_queue = multiprocessing.Queue(maxsize=1)

#         # Start the GPU (hardware) inference in a separate process
#         gpu_process = multiprocessing.Process(target=self.hardware_worker, args=(frame_queue, feat_queue))
#         gpu_process.start()

#         # Start the CPU (postprocessing) in a separate process
#         cpu_process = multiprocessing.Process(target=self.cpu_worker, args=(feat_queue, result_queue))
#         cpu_process.start()

#         # Keep the pipeline running
#         try:
#             while True:
#                 if not result_queue.empty():
#                     result = result_queue.get()
#                     # print(f"Processed Output: {result}")

#         except KeyboardInterrupt:
#             # Graceful shutdown when interrupted
#             gpu_process.terminate()
#             cpu_process.terminate()
#             print("Pipeline terminated")







import multiprocessing
import time
import cv2
import torch
from threading import Thread
from ultrahelper.load import load_deployment_model, load_postprocessor


class ParallelInferencePipeline:
    def __init__(self):
        self.return_dict = {}
        self.initial_time = time.time()

    def thread_safe_predict(self, model_path, img_tensor):
        """Performs thread-safe prediction on an image using a locally instantiated model."""
        model = load_deployment_model()  # Ensure each thread uses its own model instance
        with torch.no_grad():
            feats = model(img_tensor)  # Perform inference on the image
        return feats

    def hardware_worker(self, frame_queue, feat_queue):
        """ This worker runs on the GPU for model inference, with thread safety ensured. """
        cap = cv2.VideoCapture(0)  # Use webcam for input
        if not cap.isOpened():
            print("[ERROR] Webcam not accessible")
            return

        model_path = 'path_to_your_model.pt'  # You can pass different models here if needed
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Preprocess the frame
            img = cv2.resize(frame, (640, 640))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            start_time = time.time()

            # Use a thread to handle the prediction independently
            prediction_thread = Thread(target=self.thread_safe_predict, args=(model_path, img_tensor))
            prediction_thread.start()
            prediction_thread.join()  # Wait for the thread to finish and return the result

            inference_time = time.time() - start_time
            # Put the feature and frame into the feature queue for postprocessing
            feat_queue.put((img_tensor, frame))
            frame_rate = 1 / inference_time  # FPS calculation
            print(f"GPU FPS: {frame_rate:.2f}, Inference Latency: {inference_time:.4f}s")

    def cpu_worker(self, feat_queue, result_queue):
        """ This worker runs on the CPU for postprocessing the results. """
        postprocessor = load_postprocessor()  # Load the postprocessor (CPU)

        while True:
            feats, frame = feat_queue.get()  # Get the features from the feature queue

            start_time = time.time()
            processed_output = postprocessor(feats)  # Process the features
            postprocess_time = time.time() - start_time

            # Put the processed output into the result queue
            result_queue.put(processed_output)

            print(f"CPU Postprocessing Time: {postprocess_time:.4f}s")

    def run(self):
        """Main method to run the parallel pipeline using multiprocessing and threading."""
        # Queues for passing data between the workers
        frame_queue = multiprocessing.Queue(maxsize=1)
        feat_queue = multiprocessing.Queue(maxsize=1)
        result_queue = multiprocessing.Queue(maxsize=1)

        # Start the GPU (hardware) inference in a separate process
        gpu_process = multiprocessing.Process(target=self.hardware_worker, args=(frame_queue, feat_queue))
        gpu_process.start()

        # Start the CPU (postprocessing) in a separate process
        cpu_process = multiprocessing.Process(target=self.cpu_worker, args=(feat_queue, result_queue))
        cpu_process.start()

        # Keep the pipeline running
        try:
            while True:
                if not result_queue.empty():
                    result = result_queue.get()
                    # print(f"Processed Output: {result}")

        except KeyboardInterrupt:
            # Graceful shutdown when interrupted
            gpu_process.terminate()
            cpu_process.terminate()
            print("Pipeline terminated")

