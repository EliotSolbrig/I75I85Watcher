import cv2
import time
import os
import shutil

def whichDirection(x1, y1, x2, mDivider, bDivider):
    y2Temp = (mDivider * x2) + bDivider
    print('y2Temp: ', y2Temp)
    tempSlope = (y2Temp - y1) / (x2 - x1)
    print('tempSlope: ', tempSlope)
    if y1 < y2Temp:
        return "North"
    return "South"

haar_cascade = 'cars.xml'
# haar_cascade 20'cars2.xml'
video = 'video.avi'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
car_cascade = cv2.CascadeClassifier(haar_cascade)

try:
    shutil.rmtree('tmp')
except FileExistsError:
    print('error removing tmp dir')
try:
    os.mkdir('tmp')
except FileExistsError:
    print('error making tmp dir')

startTime = time.time()
print('startTime: ', startTime)
carsDetected = 0
southCount = 0
northCount = 0

mDivider = (995 - 1045) / (1100 - 250)
bDivider = 1045

while True:
    ret,frames = cap.read()

    frames =  cv2.rotate(frames, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # convert frames to gray scale 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    scale = 1.1
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, scale, 0, minSize=(75,75))#, maxSize=(600,600))

    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        carsDetected += 1
        # print('x: ', + x
        tempImg = cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)#[(x+w,y+h),(x-w,y-h)]
        direction = whichDirection(x, y, x, mDivider, bDivider)
        print('direction: ', direction)
        if direction == "North":
            northCount += 1
        elif direction == "South":
            southCount += 1
        cv2.imwrite('tmp/car' + str(carsDetected) +' .png', tempImg)
  
        # Display frames in a window 
    currentTime = time.time()
    timeElapsed = currentTime - startTime
    print('timeElapsed: ', timeElapsed)
    detectionRate = round(float(carsDetected / timeElapsed), 2)
    print('detectionRate: ', detectionRate)
    southRate = round(float(southCount / timeElapsed), 2)
    northRate = round(float(northCount/ timeElapsed), 2)
    countText = 'Cars detected: ' + str(carsDetected) + '(' + str(detectionRate) + '/sec)'
    southCountText = 'Northbound cars detected: ' + str(southCount) + '(' + str(southRate) + '/sec)'
    northCountText = 'Southbound cars detected: ' + str(northCount) + '(' + str(northRate) + '/sec)'
    # org = (795,700)
    org = (400,1800)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(frames, countText, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frames, northCountText, (org[0],org[1]-200), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frames, southCountText, (org[0],org[1]-400), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.line(frames, (250,1045),(1100,995), color, thickness)
    cv2.imshow('video', frames)

    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()

