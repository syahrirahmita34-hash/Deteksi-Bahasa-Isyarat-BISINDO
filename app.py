from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)

# LOAD MODEL
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# CAMERA
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MEDIAPIPE
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# LABEL
labels_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',
    5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',
    15:'P',16:'Q',17:'R',18:'S',19:'T',
    20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',

    26:'MAAF',
    27:'TERIMAKASIH',
    28:'SEKOLAH',
    29:'BAGAIMANA',
    30:'SENANG',

    31:'1',
    32:'2',
    33:'3',
    34:'4',
    35:'5',
    36:'6',
    37:'7',
    38:'8',
    39:'9',
    40:'10'
}

# HTML

HTML = '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deteksi Bahasa Isyarat Indonesia</title>

    <style>

        *{
            margin:0;
            padding:0;
            box-sizing:border-box;
            font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body{
            background:#E3F2FD;
        }

        header{
            background:linear-gradient(135deg,#42A5F5,#1565C0);
            color:white;
            text-align:center;
            padding:25px;
            box-shadow:0 4px 12px rgba(0,0,0,0.15);
        }

        header h1{
            font-size:34px;
            margin-bottom:8px;
        }

        header p{
            font-size:16px;
            opacity:0.95;
        }

        .container{
            width:90%;
            max-width:1200px;
            margin:30px auto;
        }

        .camera-box{
            background:#FFFFFF;
            padding:25px;
            border-radius:20px;
            text-align:center;
            box-shadow:0 6px 20px rgba(0,0,0,0.1);
        }

        .camera-box h2{
            color:#1565C0;
            margin-bottom:20px;
        }

        .camera-box img{
            width:100%;
            max-width:850px;
            border-radius:15px;
            border:5px solid #42A5F5;
            box-shadow:0 4px 12px rgba(0,0,0,0.1);
        }

        .status{
            margin-top:15px;
            font-size:18px;
            font-weight:bold;
            color:#1565C0;
        }

        .info-container{
            display:flex;
            gap:20px;
            margin-top:25px;
            flex-wrap:wrap;
        }

        .card{
            flex:1;
            min-width:280px;
            background:#FFFFFF;
            padding:20px;
            border-radius:20px;
            box-shadow:0 4px 15px rgba(0,0,0,0.08);
            border-top:6px solid #42A5F5;
            transition:0.3s;
        }

        .card:hover{
            transform:translateY(-5px);
        }

        .card h3{
            color:#1565C0;
            margin-bottom:15px;
        }

        .card ul{
            padding-left:20px;
        }

        .card li{
            margin-bottom:10px;
            color:#444;
        }

        footer{
            text-align:center;
            margin-top:35px;
            padding:20px;
            color:#1565C0;
            font-weight:bold;
        }

        @media(max-width:768px){

            header h1{
                font-size:26px;
            }

            .info-container{
                flex-direction:column;
            }
        }

    </style>
</head>

<body>

    <header>
        <h1>🤟 Sistem Deteksi Bahasa Isyarat Indonesia</h1>
        <p>Berbasis MediaPipe Hands dan Random Forest Classifier</p>
    </header>

    <div class="container">

        <div class="camera-box">

            <h2>Deteksi Bahasa Isyarat Secara Real-Time</h2>

            <img src="/video_feed" alt="Video Streaming">

            <div class="status">
                📷 Kamera Aktif
            </div>

        </div>

        <div class="info-container">

            <div class="card">

                <h3>📌 Informasi Sistem</h3>

                <ul>
                    <li>Jumlah Kelas : 41</li>
                    <li>Dataset : 150 gambar per kelas</li>
                    <li>Input : Webcam</li>
                    <li>Output : Huruf, Angka, dan Kata</li>
                </ul>

            </div>

            <div class="card">

                <h3>🧠 Algoritma</h3>

                <ul>
                    <li>MediaPipe Hands</li>
                    <li>21 Landmark Tangan</li>
                    <li>84 Fitur Landmark</li>
                    <li>Random Forest Classifier</li>
                </ul>

            </div>

            <div class="card">

                <h3>📖 Cara Penggunaan</h3>

                <ul>
                    <li>Posisikan tangan di depan kamera</li>
                    <li>Lakukan gesture bahasa isyarat</li>
                    <li>Tunggu proses deteksi</li>
                    <li>Hasil akan tampil secara otomatis</li>
                </ul>

            </div>

        </div>

    </div>

</body>
</html>
'''

# VIDEO DETECTION
def generate_frames():
    while True:
        success, frame = cap.read()

        if not success:
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            # padding agar selalu 84 fitur
            while len(data_aux) < 84:
                data_aux.append(0)

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 4)

            cv2.putText(
                frame,
                predicted_character,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0,255,0),
                3,
                cv2.LINE_AA
            )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ROUTES
# =========================
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)