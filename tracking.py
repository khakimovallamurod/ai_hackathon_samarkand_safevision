import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from supervision.draw.color import Color
import threading
import time

CLASS_NAMES_UZ = {
    'oddiy_harakat': 'Oddiy Harakat',
    'shubhali_harakat': 'Shubhali Harakat',
    'jabrlangan_shaxs': 'Jabrlangan Shaxs',
    'qurol_aslahasi': 'Qurol-Aslaha',
}

CLASS_COLORS = {
    'oddiy_harakat': Color.from_hex('#00FF00'),
    'shubhali_harakat': Color.from_hex('#FFA500'),
    'jabrlangan_shaxs': Color.from_hex('#FF0000'),
    'qurol_aslahasi': Color.from_hex('#8A2BE2'),
}

TRACK_CLASSES = ['oddiy_harakat', 'shubhali_harakat', 'jabrlangan_shaxs']

class VideoProcessor:
    def __init__(self):
        self.model = None
        self.tracker = None
        self.camera_cap = None
        self.processing_active = False
        self.current_camera_index = None
        self.camera_lock = threading.Lock()
        self.sent_tracker_ids = set()
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_LEFT,
            text_scale=0.5,
            text_padding=5
        )
        
    def load_model(self):
        if self.model is None:
            print("Model yuklanmoqda...")
            self.model = YOLO("models/zakladchik_model.pt")
            print(f"Model yuklandi! {len(self.model.names)} ta class")
    
    def set_camera(self, camera_index):
        with self.camera_lock:
            if self.camera_cap is not None:
                self.camera_cap.release()
            
            self.current_camera_index = camera_index
            self.sent_tracker_ids.clear()
            
            self.camera_cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            
            if not self.camera_cap.isOpened():
                print(f"Kamera ochilmadi: {camera_index}")
                return False
            
            self.camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            fps = self.camera_cap.get(cv2.CAP_PROP_FPS) or 30
            self.tracker = sv.ByteTrack(frame_rate=fps)
            
            self.processing_active = True
            print(f"Kamera ochildi: index={camera_index}, FPS={fps}")
            return True
    
    def stop_camera(self):
        with self.camera_lock:
            self.processing_active = False
            if self.camera_cap is not None:
                self.camera_cap.release()
                self.camera_cap = None
            self.current_camera_index = None
            print("Kamera to'xtatildi")
    
    def get_available_cameras(self):
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def draw_corner_lines(self, frame, x1, y1, x2, y2, color, thickness):
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
    
    def create_empty_detections(self):
        """Bo'sh detections obyekti yaratish"""
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
            tracker_id=np.empty((0,), dtype=np.int32)
        )
    
    def annotate_frame(self, frame, detections):
        # Agar detections bo'sh bo'lsa, faqat asl frameni qaytar
        if len(detections) == 0 or detections is None:
            return frame.copy()
            
        annotated_frame = frame.copy()
        labels = []
        colors = []
        
        # Track IDlar uchun massiv yaratish (agar mavjud bo'lmasa)
        if detections.tracker_id is None:
            detections.tracker_id = np.array([None] * len(detections.xyxy))
        
        for i in range(len(detections.xyxy)):
            xyxy = detections.xyxy[i]
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            class_id = int(detections.class_id[i])
            confidence = detections.confidence[i]
            
            x1, y1, x2, y2 = map(int, xyxy)
            
            class_name = self.model.names[class_id]
            uz_name = CLASS_NAMES_UZ.get(class_name, class_name)
            conf_percent = confidence * 100
            
            if tracker_id is not None:
                label_text = f"{uz_name} {conf_percent:.0f}% ID:{tracker_id}"
            else:
                label_text = f"{uz_name} {conf_percent:.0f}%"
            
            labels.append(label_text)
            
            color = CLASS_COLORS.get(class_name, Color.WHITE)
            colors.append(color)
            
            if class_name in TRACK_CLASSES:
                annotated_frame = self.draw_corner_lines(
                    annotated_frame, x1, y1, x2, y2, color.as_bgr(), 2
                )
        
        detections.color = colors
        
        # Annotatsiya qilish
        if len(detections.xyxy) > 0:
            try:
                annotated_frame = self.bounding_box_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections
                )
                
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )
            except Exception as e:
                print(f"Annotatsiya xatolik: {e}")
        
        return annotated_frame
    
    def count_objects_by_class(self, detections):
        counts = {class_name: 0 for class_name in CLASS_NAMES_UZ.values()}
        
        if len(detections) == 0:
            return counts
            
        for class_id in detections.class_id:
            class_id = int(class_id)
            class_name = self.model.names[class_id]
            uz_name = CLASS_NAMES_UZ.get(class_name, class_name)
            if uz_name in counts:
                counts[uz_name] += 1
        
        return counts
    
    def generate_frames(self):
        if self.current_camera_index is None:
            print("Kamera tanlanmagan!")
            return
        
        self.load_model()
        
        CONFIDENCE_THRESHOLD = 0.35
        NMS_IOU_THRESHOLD = 0.3
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.processing_active:
                with self.camera_lock:
                    if self.camera_cap is None or not self.camera_cap.isOpened():
                        break
                    
                    ret, frame = self.camera_cap.read()
                
                if not ret:
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # YOLO modeli orqali detection
                try:
                    results = self.model(frame, imgsz=(640, 640), verbose=False)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    
                    # Confidence threshold qo'llash
                    if len(detections) > 0:
                        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
                    
                    # NMS qo'llash
                    if len(detections) > 0:
                        detections = detections.with_nms(NMS_IOU_THRESHOLD)
                    
                    # Tracking qilish
                    if len(detections) > 0:
                        track_mask = np.array([
                            self.model.names[int(class_id)] in TRACK_CLASSES 
                            for class_id in detections.class_id
                        ])
                        
                        if np.any(track_mask):
                            track_detections = detections[track_mask]
                            
                            # Track qilish uchun XYXY formatida koordinatalar kerak
                            if len(track_detections.xyxy) > 0:
                                try:
                                    tracked_detections = self.tracker.update_with_detections(track_detections)
                                    
                                    # Track IDlarni asosiy detections ga qo'shish
                                    if detections.tracker_id is None:
                                        detections.tracker_id = np.array([None] * len(detections))
                                    
                                    track_idx = 0
                                    for i in range(len(detections)):
                                        if track_mask[i] and track_idx < len(tracked_detections):
                                            if tracked_detections.tracker_id is not None:
                                                detections.tracker_id[i] = tracked_detections.tracker_id[track_idx]
                                            track_idx += 1
                                except Exception as e:
                                    print(f"Tracking xatolik: {e}")
                                    detections.tracker_id = np.array([None] * len(detections))
                    else:
                        # Agar detections bo'sh bo'lsa, bo'sh detections yaratish
                        detections = self.create_empty_detections()
                        
                except Exception as e:
                    print(f"Detection xatolik: {e}")
                    detections = self.create_empty_detections()
                
                # Annotatsiya qilish
                try:
                    annotated_frame = self.annotate_frame(frame, detections)
                except Exception as e:
                    print(f"Annotatsiya xatolik: {e}")
                    annotated_frame = frame.copy()
                
                # Statistik ma'lumotlar
                counts = self.count_objects_by_class(detections)
                
                elapsed_time = time.time() - start_time
                fps_current = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # FPS va ma'lumotlarni chizish
                y_pos = 30
                cv2.putText(annotated_frame, f"Kamera: {self.current_camera_index}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_pos += 30
                cv2.putText(annotated_frame, f"FPS: {fps_current:.1f}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Ob'ektlar sonini ko'rsatish
                y_pos += 40
                for class_name_uz, count in counts.items():
                    if count > 0:
                        color_key = next((k for k, v in CLASS_NAMES_UZ.items() if v == class_name_uz), class_name_uz)
                        color = CLASS_COLORS.get(color_key, Color.WHITE).as_bgr()
                        
                        cv2.putText(annotated_frame, f"{class_name_uz}: {count}", (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_pos += 25
                
                # Frame'ni encode qilish
                ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except Exception as e:
            print(f"Generate frames xatolik: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.processing_active = False
            print("Video qayta ishlash tugadi!")