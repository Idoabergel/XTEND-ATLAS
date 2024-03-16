# =========================================
# =============== IMPORTS =================
# =========================================
import torch
import cv2
import numpy as np
from torchvision.ops import nms

# =========================================
# =============== MISSION =================
# =========================================

"""
# Implement collision detection logic
# Your task is to write code that analyzes the detected people's positions
# and determines if the robot is on a collision path with any of them.
# You can find the detected people's bounding box coordinates in the 'people_boxes' list.
# Use this information to calculate the distance between the robot and each person,
# and trigger an alert (e.g., play a sound, display a message) if a collision is imminent.
# Happy coding!
"""

# =========================================
# =========== GLOBAL PARAMS ===============
# =========================================
# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Full path for video to process.
video_path = '/home/ido/Work/code-repos/cvsandbox_mark3/peopleTracker/people_yoloV8/videos/lab01.mp4'


# TODO: change folder to sub-dir videos

# =========================================
# ========== HELPER FUNCTION ==============
# =========================================
def download_model():
    # This function is a placeholder. YOLOv5 models can be loaded using PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to(device)
    return model


# =========================================
# ========= MODEL PREDICTIONS =============
# =========================================


def process_detections(detections, conf_threshold=0.25, nms_threshold=0.45):
    """
    Process raw detections to extract bounding box coordinates, apply NMS, and filter by confidence.

    Args:
        detections (torch.Tensor): Raw model output tensor of shape [1, 6300, 85].
        conf_threshold (float): Confidence threshold to filter detections.
        nms_threshold (float): IoU threshold for NMS filtering.

    Returns:
        List of filtered and NMS-processed detections, each formatted as [x1, y1, x2, y2, confidence, class_id].
    """
    # Filter out detections below the confidence threshold
    conf_mask = detections[..., 4] > conf_threshold
    detections = detections[conf_mask]

    # Extract class scores, calculate max class score and its corresponding class id
    class_scores = detections[..., 5:]
    max_scores, class_ids = torch.max(class_scores, dim=-1)
    confidences = detections[..., 4] * max_scores

    # Filter out detections with low class-specific confidence
    conf_mask = confidences > conf_threshold
    detections = detections[conf_mask]
    class_ids = class_ids[conf_mask]
    confidences = confidences[conf_mask]

    # Convert detections to [x1, y1, x2, y2] format
    boxes = detections[..., :4]
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    # Apply NMS
    keep = nms(boxes_xyxy, confidences, nms_threshold)
    final_boxes = boxes_xyxy[keep]
    final_scores = confidences[keep]
    final_class_ids = class_ids[keep]

    # Convert everything back to numpy arrays (if necessary) to facilitate further processing
    final_boxes_np = final_boxes.cpu().numpy()
    final_scores_np = final_scores.cpu().numpy()
    final_class_ids_np = final_class_ids.cpu().numpy()

    # Combine boxes with their confidences and class IDs
    detections_processed = np.concatenate([
        final_boxes_np,
        final_scores_np[:, None],
        final_class_ids_np[:, None]], axis=-1)

    return detections_processed.tolist()


def resize_detections_to_original(detections, original_size, inference_size=(640, 480)):
    """
    Resize detection bounding boxes back to the original frame size.

    Args:
        detections (list): List of detections, each detection is [x1, y1, x2, y2, confidence, class_id].
        original_size (tuple): The original size of the frame as (width, height).
        inference_size (tuple): The size used for inference as (width, height).

    Returns:
        List of resized detections.
    """
    original_width, original_height = original_size
    inference_width, inference_height = inference_size

    # Calculate scale factors
    x_scale = original_width / inference_width
    y_scale = original_height / inference_height

    resized_detections = []
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection

        # Scale bounding box coordinates back to original frame size
        x1 = x1 * x_scale
        y1 = y1 * y_scale
        x2 = x2 * x_scale
        y2 = y2 * y_scale

        resized_detection = [x1, y1, x2, y2, confidence, class_id]
        resized_detections.append(resized_detection)

    return resized_detections


def process_video(video_path, model):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Extract original frame size
    original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_size = (original_frame_width, original_frame_height)

    # Size of the frame that is fed into the YOLO
    inference_size = (640, 480)

    # Inference loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to model's expected size
        frame_resized = cv2.resize(frame, inference_size)

        # Convert frame to format expected by the model
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize to [0, 1]
        img = torch.tensor(img, dtype=torch.float32).to(device)

        with torch.no_grad():
            results = model(img)

        # Convert detections to numpy array
        # Each detection has the format: x1, y1, x2, y2, confidence, class
        processed_detections = process_detections(results, conf_threshold=0.5)
        resized_detections = resize_detections_to_original(processed_detections, original_size, inference_size)

        # Iterate through detections
        for det in resized_detections:
            x1, y1, x2, y2 = map(int, det[:4])
            conf, cls = map(float, det[4:])
            if cls == 0:  # Assuming '0' is the class for person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Visualization
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all windows
    cap.release()
    cv2.destroyAllWindows()


# =========================================
# ========= COMPUTE COLLISION =============
# =========================================
# TODO: Implement collision detection logic
def is_collision():
    # Code your solution here
    pass


# =========================================
# ================= MAIN ==================
# =========================================
def main():
    model = download_model()
    process_video(video_path, model)


if __name__ == '__main__':
    main()
