import cv2 as cv 
import cvzone
import numpy as np 

cap = cv.VideoCapture(0)    # starting camera   0 means your inuilt camera 
                # this can change according to the number of camera used 


cap.set(3,640)    # setting width 
cap.set(4,480)      # setting height 

total_money = 0 

def empty(s):
    pass


cv.namedWindow("settings")
cv.resizeWindow('settings',640,240)
cv.createTrackbar("Threshold1", 'settings' , 135, 255, empty)
cv.createTrackbar("Threshold2", 'settings' , 67 , 255, empty)












def preprocessing(img):

    # step 1 : have to add some blur bcoz if the image is too sharp the contours will be too edgy 
    
    imgPre = cv.GaussianBlur(img , (5,5) , 3)
    
    thresh1 = cv.getTrackbarPos('Threshold1', 'settings')
    thresh2 = cv.getTrackbarPos('Threshold2', 'settings')
    

    #  finding edges  using canny fnx 

    imgPre = cv.Canny(imgPre,thresh1 , thresh2)


    # after canny we have to dilate it , for filling gaps after canny 
        # and using closing function to close any contverse that are open 

    # dilate : thicken and makint the shape of corners more prominent 

    kernel = np.ones((1,1), np.uint8)

    imgPre = cv.dilate(imgPre, kernel , iterations= 1 )

    # closing opens 

    imgPre = cv.morphologyEx(imgPre , cv.MORPH_CLOSE, kernel)



    return imgPre


while True:
    # boiler plate for logging the web cam 
    success,img = cap.read()
    imgPre = preprocessing(img)


    #contours are used for getting the positions 
    imgContours , confound = cvzone.findContours(img,imgPre, minArea= 30)


    # differentiating based on area ----------


    total_money = 0 
    if confound :
        for contour in confound:
            peri = cv.arcLength(contour['cnt'],True)
            approx = cv.approxPolyDP( contour['cnt'],0.02 * peri , True)
            if(len(approx) > 5):
                area = contour['area']

            if area < 2050 :
                total_money += 5 

            elif 2050 < area < 2500 :
                total_money += 1 
            else:
                total_money += 2 
    print(total_money)
            

                
















    # lets put both the images together so we have to use cvzone stackImage func 

    imgStacked = cvzone.stackImages([img, imgPre],2,1)

    cvzone.putTextRect(imgStacked , f'rs.{total_money}',(50,50))   

    cv.imshow("Image",imgStacked)   # for showing the image 


    # cv.imshow("ImgPre" , imgPre)

    cv.imshow("contour image" , imgContours)
    cv.waitKey(1)      # 1 delay 
