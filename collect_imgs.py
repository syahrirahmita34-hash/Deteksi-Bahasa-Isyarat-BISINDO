import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 41
dataset_size = 150

cap = cv2.VideoCapture(0)

for j in range(31, 41):
    class_path = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Tekan Q', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(50)

        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()