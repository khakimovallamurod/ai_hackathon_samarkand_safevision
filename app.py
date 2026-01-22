from flask import Flask, Response, render_template, request, jsonify
import tracking
import os
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)

# Video yuklash uchun papka
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit

def allowed_file(filename):
    """Fayl formatini tekshirish"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Asosiy sahifa"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video stream - kamera yoki yuklangan video"""
    try:
        return Response(tracking.generate_frames(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Video feed xatolik: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Kamerani ishga tushirish"""
    try:
        # Kamera index 0 dan boshlash
        tracking.set_video(0)
        
        return jsonify({
            'status': 'success', 
            'message': 'Kamera ishga tushdi',
            'source': 'camera'
        })
    except Exception as e:
        print(f"Kamera xatolik: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error', 
            'message': f'Xatolik: {str(e)}'
        }), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Video yuklash va qayta ishlashni boshlash"""
    try:
        # Faylni tekshirish
        if 'video' not in request.files:
            return jsonify({
                'status': 'error', 
                'message': 'Video fayl topilmadi'
            }), 400
        
        file = request.files['video']
        
        # Fayl tanlanganligini tekshirish
        if file.filename == '':
            return jsonify({
                'status': 'error', 
                'message': 'Fayl tanlanmagan'
            }), 400
        
        # Fayl formatini tekshirish
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error', 
                'message': f'Noto\'g\'ri fayl formati. Ruxsat etilgan: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Faylni saqlash
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"Video saqlanmoqda: {filepath}")
        file.save(filepath)
        print(f"Video saqlandi: {filepath}")
        
        # Fayl mavjudligini tekshirish
        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': 'Video saqlashda xatolik'
            }), 500
        
        # Videoni qayta ishlashni boshlash
        tracking.set_video(filepath)
        
        return jsonify({
            'status': 'success', 
            'message': 'Video yuklandi va qayta ishlanmoqda',
            'filename': filename,
            'filepath': filepath,
            'source': 'file'
        })
    
    except Exception as e:
        print(f"Upload xatolik: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error', 
            'message': f'Xatolik: {str(e)}'
        }), 500

@app.route('/stop_video', methods=['POST'])
def stop_video():
    """Videoni to'xtatish"""
    try:
        tracking.stop_video()
        return jsonify({
            'status': 'success', 
            'message': 'Video qayta ishlash to\'xtatildi'
        })
    except Exception as e:
        print(f"Stop video xatolik: {e}")
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500

@app.route('/check_status', methods=['GET'])
def check_status():
    """Video qayta ishlash holatini tekshirish"""
    try:
        is_active = tracking.processing_active
        current_video = tracking.current_video
        
        # Agar current_video 0 bo'lsa, bu kamera ekanligini ko'rsatish
        source_type = 'camera' if current_video == 0 else 'file' if current_video else None
        
        return jsonify({
            'status': 'success',
            'is_processing': is_active,
            'current_video': current_video,
            'source_type': source_type
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/uploaded_videos', methods=['GET'])
def uploaded_videos():
    """Yuklangan videolarni ro'yxatini qaytarish"""
    try:
        videos = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if allowed_file(filename):
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    size = os.path.getsize(filepath)
                    videos.append({
                        'filename': filename,
                        'size': size,
                        'size_mb': round(size / (1024 * 1024), 2)
                    })
        
        return jsonify({
            'status': 'success',
            'videos': videos
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/delete_video/<filename>', methods=['DELETE'])
def delete_video(filename):
    """Yuklangan videoni o'chirish"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({
                'status': 'success',
                'message': f'{filename} o\'chirildi'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Fayl topilmadi'
            }), 404
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Fayl hajmi katta bo'lganda"""
    return jsonify({
        'status': 'error',
        'message': 'Fayl hajmi juda katta. Maksimal: 500MB'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Server xatoligi"""
    return jsonify({
        'status': 'error',
        'message': 'Server xatoligi'
    }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Flask Server ishga tushmoqda...")
    print("📡 Server manzili: http://0.0.0.0:5000")
    print("📡 Lokal: http://localhost:5000")
    print("📹 Kamera index: 0")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)