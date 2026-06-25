import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            # 🔥 urutkan kiri → kanan
            hands_sorted = sorted(
                results.multi_hand_landmarks,
                key=lambda hand: min([lm.x for lm in hand.landmark])
            )

            for hand_landmarks in hands_sorted:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

            for hand_landmarks in hands_sorted:
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            # 🔥 padding kalau 1 tangan (42 → 84)
            if len(data_aux) == 42:
                data_aux.extend([0] * 42)

            # hanya ambil kalau 84 fitur
            if len(data_aux) == 84:
                data.append(data_aux)
                labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Jumlah data:", len(data))