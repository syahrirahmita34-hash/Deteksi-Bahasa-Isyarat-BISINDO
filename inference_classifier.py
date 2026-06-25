import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:

        data_aux = []
        x_ = []
        y_ = []

        # 🔥 urutkan kiri → kanan
        hands_sorted = sorted(
            results.multi_hand_landmarks,
            key=lambda hand: min([lm.x for lm in hand.landmark])
        )

        for hand_landmarks in hands_sorted:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

        for hand_landmarks in hands_sorted:
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        # 🔥 padding kalau 1 tangan
        if len(data_aux) == 42:
            data_aux.extend([0]*42)

        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 3)
            cv2.putText(frame, predicted_character, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()