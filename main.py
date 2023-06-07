import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
import os

cap = cv2.VideoCapture("Resources/videos/1.mp4") #(0)
detector = PoseDetector() # method to run poseDetector module

shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)#to import shirts
#print(listShirts)
fixedRatio = 262/190  # widthOfShirt/widthOfPoints11to12(ie shoulder)
shirtRatioHeightWidth = 581/440
imageNumber = 1

while True:
    success, img = cap.read() #it will capture our image and will tell is it success or not
    img = detector.findPose(img) #getting the pose from the image
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False) #bounding box of pose module
    if lmList: #if body is detected then import shirt
        #print(lmList)
        lm11 = lmList[11][1:3]
        lm12 = lmList[12][1:3]
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
        #overlaying image on pose

        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)#widthofShirt is in pixles
        print(widthOfShirt)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))#resize/scaling shirt on width and height
        currentScale = (lm11[0] - lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)

        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
        except:
            pass

    cv2.imshow("Image", img) #to display image(name of window, which img wanna to display)
    cv2.waitKey(1) #to add 1 millisec delay