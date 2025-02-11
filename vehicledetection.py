import cv2
import time

haar_cascade = 'cars.xml'
video = 'video.avi'

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
car_cascade = cv2.CascadeClassifier(haar_cascade)

startTime = time.time()
print('startTime: ', startTime)
carsDetected = 0

while True:
    ret,frames = cap.read()

    # convert frames to gray scale 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
      
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.15, 1)

    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        carsDetected += 1
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
  
        # Display frames in a window 
    currentTime = time.time()
    timeElapsed = currentTime - startTime
    print('timeElapsed: ', timeElapsed)
    detectionRate = round(float(carsDetected / timeElapsed), 2)
    print('detectionRate: ', detectionRate)
    countText = 'Cars detected: ' + str(carsDetected) + '(' + str(detectionRate) + '/sec)'
    org = (795,700)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(frames, countText, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('video', frames)

    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()

