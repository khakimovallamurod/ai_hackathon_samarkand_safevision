from flask import Flask, Response, render_template, jsonify, request
from tracking import VideoProcessor
import cv2

app = Flask(__name__)
video_processor = VideoProcessor()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        video_processor.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/get_cameras")
def get_cameras():
    cameras = video_processor.get_available_cameras()
    return jsonify({"cameras": cameras})

@app.route("/set_camera", methods=["POST"])
def set_camera():
    data = request.get_json()
    camera_index = data.get("camera_index", 0)
    
    success = video_processor.set_camera(camera_index)
    
    if success:
        return jsonify({"status": "success", "message": f"Kamera {camera_index} ochildi"})
    else:
        return jsonify({"status": "error", "message": f"Kamera {camera_index} ochilmadi"}), 400

@app.route("/stop_camera")
def stop_camera():
    video_processor.stop_camera()
    return jsonify({"status": "success", "message": "Kamera to'xtatildi"})

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )