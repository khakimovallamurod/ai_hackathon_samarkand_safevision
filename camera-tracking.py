import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
from supervision.draw.color import Color
from telegram import Bot
import io
from config import get_token, get_chat_id
import asyncio


CLASS_COLORS = {
    'Helmet': Color.GREEN,
    'No-Helmet': Color.RED,
    'Vest': Color.GREEN,
    'No-Vest': Color.RED,
    'Person': Color.BLUE,
    'Person-Fall': Color.YELLOW,
    'Fire': Color.RED,
    'Smoke': Color.BLACK,
    'fire': Color.RED,
    'smoke': Color.BLACK,
}

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

def load_model(model_path):
    return YOLO(model_path)

def setup_tracking(fps):
    return sv.ByteTrack(frame_rate=fps)

def draw_corner_lines(frame, x1, y1, x2, y2, color, thickness):
    line_len_x = (x2 - x1) // 4
    line_len_y = (y2 - y1) // 4
    corners = [
        ((x1, y1), (x1 + line_len_x, y1), (x1, y1 + line_len_y)),
        ((x2, y1), (x2 - line_len_x, y1), (x2, y1 + line_len_y)),
        ((x1, y2), (x1 + line_len_x, y2), (x1, y2 - line_len_y)),
        ((x2, y2), (x2 - line_len_x, y2), (x2, y2 - line_len_y))
    ]
    for start, end_x, end_y in corners:
        cv2.line(frame, start, end_x, color, thickness)
        cv2.line(frame, start, end_y, color, thickness)
    return frame

def get_class_name(model, class_id):
    return model.model.names[int(class_id)]

def count_people(detections, model):
    return sum(1 for cid in detections.class_id if get_class_name(model, cid) == 'Person')

def annotate_frame(frame, model1, model2, model3, detections):
    annotated_frame = frame.copy()
    labels = []

    for xyxy, tracker_id, class_id in zip(detections.xyxy, detections.tracker_id, detections.class_id):
        x1, y1, x2, y2 = map(int, xyxy)
        class_id = int(class_id)

        class_name = (
            model1.model.names.get(class_id)
            or model2.model.names.get(class_id)
            or model3.model.names.get(class_id)
            or f"ID:{class_id}"
        )

        color = CLASS_COLORS.get(class_name, Color.WHITE)
        labels.append(class_name)

        if class_name == 'Person':
            annotated_frame = draw_corner_lines(annotated_frame, x1, y1, x2, y2, color.as_bgr(), 2)
        else:
            detection_box = sv.Detections(
                xyxy=np.array([[x1, y1, x2, y2]]),
                confidence=np.array([1.0]),
                class_id=np.array([class_id]),
                tracker_id=np.array([tracker_id])
            )
            detection_box.color = [color]
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame,
                detections=detection_box
            )
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame, count_people(detections, model2)

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def merge_detections(det1, model1, det2, model2, det3, model3):
    final_xyxy = []
    final_conf = []
    final_class_id = []
    final_tracker_id = []

    priority_classes = ["Helmet", "No-Helmet"]
    other_classes = ["Vest", "No-Vest", "Person-Fall", "Fire", "Smoke"]

    for xyxy, conf, cls in zip(det1.xyxy, det1.confidence, det1.class_id):
        if model1.model.names[cls] == "Person":
            final_xyxy.append(xyxy)
            final_conf.append(conf)
            final_class_id.append(cls)
            final_tracker_id.append(None)

    for cls_name in priority_classes:
        detections_by_pos = []

        for xyxy, conf, cls in zip(det1.xyxy, det1.confidence, det1.class_id):
            if model1.model.names[cls] == cls_name:
                detections_by_pos.append((xyxy, conf, cls))

        for xyxy, conf, cls in zip(det2.xyxy, det2.confidence, det2.class_id):
            if model2.model.names[cls] == cls_name:
                detections_by_pos.append((xyxy, conf, cls))

        for xyxy, conf, cls in zip(det3.xyxy, det3.confidence, det3.class_id):
            if model3.model.names[cls] == cls_name:
                detections_by_pos.append((xyxy, conf, cls))

        kept = []
        for i, (xyxy, conf, cls) in enumerate(detections_by_pos):
            keep = True
            for j, (other_xyxy, other_conf, other_cls) in enumerate(detections_by_pos):
                if i != j:
                    iou = compute_iou(xyxy, other_xyxy)
                    if iou > 0.5 and other_conf > conf:
                        keep = False
                        break
            if keep:
                kept.append((xyxy, conf, cls))

        for xyxy, conf, cls in kept:
            final_xyxy.append(xyxy)
            final_conf.append(conf)
            final_class_id.append(cls)
            final_tracker_id.append(None)

    for cls_name in other_classes:
        detections_by_pos = []

        for xyxy, conf, cls in zip(det1.xyxy, det1.confidence, det1.class_id):
            if model1.model.names[cls] == cls_name:
                detections_by_pos.append((xyxy, conf, cls))

        for xyxy, conf, cls in zip(det2.xyxy, det2.confidence, det2.class_id):
            if model2.model.names[cls] == cls_name:
                detections_by_pos.append((xyxy, conf, cls))

        for xyxy, conf, cls in zip(det3.xyxy, det3.confidence, det3.class_id):
            if model3.model.names[cls] == cls_name:
                detections_by_pos.append((xyxy, conf, cls))

        kept = []
        for i, (xyxy, conf, cls) in enumerate(detections_by_pos):
            keep = True
            for j, (other_xyxy, other_conf, other_cls) in enumerate(detections_by_pos):
                if i != j:
                    iou = compute_iou(xyxy, other_xyxy)
                    if iou > 0.5 and other_conf > conf:
                        keep = False
                        break
            if keep:
                kept.append((xyxy, conf, cls))

        for xyxy, conf, cls in kept:
            final_xyxy.append(xyxy)
            final_conf.append(conf)
            final_class_id.append(cls)
            final_tracker_id.append(None)

    if len(final_xyxy) == 0:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(final_xyxy),
        confidence=np.array(final_conf),
        class_id=np.array(final_class_id),
        tracker_id=np.array(final_tracker_id)
    )

async def send_to_telegram(bot, chat_id, frame):
 
    _, buffer = cv2.imencode('.jpg', frame)
    img_byte = buffer.tobytes()
    await bot.send_photo(chat_id=chat_id, photo=img_byte)
sent_tracker_ids = set()
async def main(camera_index, output_path, bot_token, chat_id):
    MODEL_NAME = "models/person20k.pt"
    MODEL_NAME_TWO = "models/build25k.pt"
    MODEL_NAME_THREE = "models/fire-smoke-model.pt" 
    CONFIDENCE_THRESHOLD = 0.5
    NMS_IOU_THRESHOLD = 0.4

    model1 = load_model(MODEL_NAME)
    model2 = load_model(MODEL_NAME_TWO)
    model3 = load_model(MODEL_NAME_THREE)  

    bot = Bot(token=bot_token)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Kamera ochilmadi!")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    if frame_width == 0 or frame_height == 0:
        print("Frame o'lchamlari noto'g'ri. Chiqyapti...")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    tracker = setup_tracking(fps)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result1 = model1(frame, imgsz=(640, 640), verbose=False)[0]
            result2 = model2(frame, imgsz=(640, 640), verbose=False)[0]
            result3 = model3(frame, imgsz=(640, 640), verbose=False)[0]  

            detections1 = sv.Detections.from_ultralytics(result1)
            detections2 = sv.Detections.from_ultralytics(result2)
            detections3 = sv.Detections.from_ultralytics(result3) 

            detections1 = detections1[detections1.confidence > CONFIDENCE_THRESHOLD]
            detections2 = detections2[detections2.confidence > CONFIDENCE_THRESHOLD]
            detections3 = detections3[detections3.confidence > CONFIDENCE_THRESHOLD]

            merged_detections = merge_detections(detections1, model1, detections2, model2, detections3, model3)
            merged_detections = merged_detections.with_nms(NMS_IOU_THRESHOLD)
            merged_detections = tracker.update_with_detections(merged_detections)

            annotated_frame, person_count = annotate_frame(frame, model1, model2, model3, merged_detections)

            for class_id, tracker_id in zip(merged_detections.class_id, merged_detections.tracker_id):
                if tracker_id is None:
                    continue  
                label = (
                    model1.model.names.get(class_id) or
                    model2.model.names.get(class_id) or
                    model3.model.names.get(class_id)
                )
                if label in ["Fire", "Smoke", "fire", "smoke"] and tracker_id not in sent_tracker_ids:
                    print(f"Detected: {label}, Tracker ID: {tracker_id}")
                    await send_to_telegram(bot, chat_id, annotated_frame)
                    sent_tracker_ids.add(tracker_id)
                    break
                
            cv2.putText(annotated_frame, f"Ishchilar soni: {person_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Construction Monitoring", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # writer.write(annotated_frame)

    finally:
        cap.release()
        writer.release()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    token = get_token()
    chat_id = get_chat_id()
    output_path = 'video_2025-11-23_09-14-31.mp4'
    camera_idx = 0
    asyncio.run(main(camera_index = camera_idx, 
                     output_path = output_path, 
                     bot_token = token, 
                     chat_id = chat_id)) 