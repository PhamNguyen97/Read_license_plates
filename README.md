# Read_license_plates
License plate reading - Image Processing Technique and KNN Classifier

# Introduction
License plate reading is an interest task with numerous uses in real world. Though there are many image processing algorithms to this task, the accuracy may varied due to changes of surrounding environment, expecially light. The compination of old image processing algorithms and mordern maching learning model may solve the problem. 

KNN Classifier is a simple algorithm but it can bring surprisingly high accuracy output. In this situation, KNN is used for character classification.

![Alt text](img1.png?raw=true "img1")
![Alt text](img2.png?raw=true "img2")

# Algorithm
- Step1. Prepare training data (image of characters in license plates (0,1,2,3,....,9,A,B,C,...) with labels)
- Step2. Because the license plates are brighter than other image areas, so this can be a solution for segmentation in HSV color space.
- Step3. Make license plates in the previous step rectangle (using def four_point_transform(image, pts)).
- Step4. Find contours of characters in license plates and take them out of the image.
- Step5. Pass them through KNN model.

# Output
![Alt text](output/output1.PNG?raw=true "output1")
![Alt text](output/output2.PNG?raw=true "output2")

# Dependencies
- opencv
- numpy
- matplotlib
- os
- sklearn
- math
