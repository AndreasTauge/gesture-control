import cv2 as cv 
import os 
import time 

cap = cv.VideoCapture(2) 
counter = 0

gesture_name = "thumbs_up"
save_dir = f"data/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)
last_capture = time.time()
capture_delay = 0.5 

if not cap.isOpened():
    print("Could not open camera")
    exit()

while counter < 200:
    ret, frame = cap.read()

    if not ret:
        print("Cant receive frame. Exiting")
        break

    current_time = time.time()

    if current_time - last_capture > capture_delay:
        filename = f"{save_dir}/img_{counter}.jpg"
        cv.imwrite(filename, frame)
        counter += 1
        last_capture = current_time
        print(f"Saved {filename}")
    
    cv.putText(frame, f"Captured: {counter}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("frame", frame)

    if cv.waitKey(1) == ord("q"):
        break 

cap.release()
cv.destroyAllWindows()