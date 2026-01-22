from ultralytics import YOLO
import cv2

model = YOLO("train7/weights/best.pt")

def image_prediction(image_path):
    image = cv2.imread(image_path)
    results = model.predict(image, conf=0.5, save=True)
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.model.names[int(class_id)]}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Predicted Image", image)
    cv2.imwrite("results/predicted_image01.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_prediction("test/ppe_test/images/image_181_jpg.rf.5c76357ca11f55470a77dadd458d2545.jpg")
# image_prediction('image copy.png')
