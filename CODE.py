
import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import os
from sklearn.neighbors import KNeighborsClassifier

train_folder = 'Train'
train_data = []
labels = []
real_label = {}
for class_ in os.listdir(train_folder):
    [this_class,this_label] = class_.split('.')
    class_folder = os.path.join(train_folder,class_)
    real_label[int(this_class)]=this_label
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder,img_name)
        img = cv2.imread(img_path,0)
        img = np.reshape(np.array(img),-1)
        train_data.append(img)
        labels.append(int(this_class))

print('loading Data Done !!!')



neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(train_data,labels)


# In[8]:


def predict(so):
    if len(so.shape)==3:
        this_img = cv2.cvtColor(so,cv2.COLOR_BGR2GRAY)
    else:
        this_img = so
    row,col = this_img.shape
    
    this_img = cv2.resize(this_img,(int(50/row*col),50))
#     this_img = cv2.bitwise_not(this_img)
    
    row,col = this_img.shape
    img_50 = np.zeros((50,50),np.uint8)
    if col>50:
        this_img = cv2.resize(this_img,(50,(int(50/col*row))))

    row,col = this_img.shape
    this_img = np.array(this_img,np.uint8)
    img_50[25-int(row/2):25-int(row/2)+row,25-int(col/2):25-int(col/2)+col]=this_img

    img_50 = np.reshape(img_50,-1)

    return real_label[neigh.predict([img_50])[0]]


# In[4]:



 
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# In[5]:


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# In[36]:


import cv2
import numpy as np
import math
import os
import datetime


# img_name= "bien_so_thay.png"

for img_name in os.listdir('Test'):
    time_start = datetime.datetime.now()
    img = cv2.imread('Test\\'+img_name)
    # #     img = cv2.imread(img_name)
    #     cv2.imshow('img',img)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask_h = img_hsv[:,:,2]
    mask = cv2.inRange(mask_h,160,255)

    row,col = mask.shape
    mask_1 = np.zeros((row+2,col+2),np.uint8)
    mask = cv2.bitwise_not(mask)
    cv2.floodFill(mask, mask_1, (0,0), 255)
    mask = cv2.bitwise_not(mask)




    # cv2.imshow('asdad',mask_h)
    # mask = cv2.adaptiveThreshold(mask_h,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,7)
    # mask = cv2.erode(mask,kernel = None,iterations = 2)
    im2, contours, hierarchy = cv2.findContours(mask,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    row,col = mask.shape
    # print(row,col)
    bienso= []
    for i,found_contour in enumerate(contours):
        x,y,w,h = cv2.boundingRect(contours[i])
        approx = cv2.approxPolyDP(np.array(found_contour),0.1*cv2.arcLength(found_contour,True),True)

    #     print(len(approx))
    #     cv2.drawContours(img,[box],0,(255,100,0),2)
        if len(approx>=4) and w>row*.1 and h>col*.1:
            rec = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rec)
            box = order_points(box)
    #         print(box)
            h = math.sqrt((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)
            w = math.sqrt((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2)
    #         print(h,w)
            if h>w:
                a = h
                h = w
                w = a
            if h>row*.12 and w>col*.12 and h<row*.95 and w<col*.95:

                # cv2.drawContours(img, [np.int0(box)], -1, (0,255,0), 3)
                bienso.append(box)
                # print(w,h)
                # cv2.imshow('img',img)
                # cv2.waitKey()
                # cv2.destroyAllWindows()


    #             print(box)

    # for i,found_contour in enumerate(contours):
    #     rec = cv2.minAreaRect(contours[i])
    #     box = np.int0(cv2.boxPoints(rec))
    #     cv2.drawContours(img,[box],0,(255,100,0),2)
    for i,box in enumerate(bienso):
        pts= order_points(box)
        new_img = four_point_transform(img,pts)
        new_img = cv2.resize(new_img,(190,140),interpolation = cv2.INTER_AREA)
        new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
        new_img = cv2.GaussianBlur(new_img,(5,5),0)
        new_img = cv2.adaptiveThreshold(new_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,3)
        new_img = cv2.bitwise_not(new_img)
        _, contours, _ = cv2.findContours(new_img,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        row,col = new_img.shape
        
        foundnum = []
        bot_right = box[0]
        for i1,cont in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cont)
            if w>15 and h>30 and h<row/2:
                this_img = np.array(new_img[y:y+h,x:x+w])
                foundnum.append([y,x,h,w,this_img])
                # if x>bot_right[0]: bot_right[0]=x
                # if y>bot_right[1]: bot_right[1]=y

        
        if len(foundnum)>=8:
            # plt.figure()
            # plt.imshow(new_img)
            # plt.show(block = True)

            x = [i[0] for i in foundnum]
            w = [i[2] for i in foundnum]

           
            x_max = min(x)+max(w)


            hang_tren = []
            for i in foundnum:
                if i[0]<x_max:
                    hang_tren.append(i)

            hang_tren = sorted(hang_tren,key = lambda hang_tren:hang_tren[1])

            hang_tren = np.array([i[4] for i in hang_tren])

            hang_duoi = []
            for i in foundnum:
                if i[0]>x_max:
                    hang_duoi.append(i)

            hang_duoi = sorted(hang_duoi,key = lambda hang_duoi:hang_duoi[1])
            hang_duoi =np.array( [i[4] for i in hang_duoi])

            if (len(hang_tren)*len(hang_duoi)!=0):
                foundnum = np.concatenate((hang_tren,hang_duoi))
            else:
                foundnum = hang_tren if len(hang_tren!=0) else hang_duoi if (len(hang_duoi)!=0) else None    

            so_bien_so =''
            for num in foundnum:
                label = predict(num)
                so_bien_so+=label

            cv2.putText(img,so_bien_so,(int(bot_right[0]),int(bot_right[1])),cv2.FONT_HERSHEY_COMPLEX,.7,(0,0,255),2)
            print(so_bien_so)  

    cv2.imshow('img',img)
    time_stop = datetime.datetime.now()
    
    print((time_stop-time_start))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
            

            


# In[ ]:





# In[ ]:




