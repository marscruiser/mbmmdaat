import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('flower.jpg')

#---------->>>
#---------->>>
#section 1
'''
#how to show an image
cv.imshow("image",img)

cv.waitKey(0)
#resizing an image
img = cv.resize(img,(int(img.shape[1]*0.5),int(img.shape[0]*0.5)),interpolation=cv.INTER_AREA)
cv.imshow("image",img)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#section 2
#drawing shapes
#empty image
blank = np.ones((500,500,3),dtype="uint8")*255
'''

box_size = 400
half_size = box_size//2
cv.rectangle(blank,(250-half_size,250-half_size),(250+half_size,250+half_size),(255,0,0),thickness=cv.FILLED)
cv.circle(blank,(250,250),50,(255,255,255),thickness=cv.FILLED)
cv.line(blank,(250-half_size,250-half_size),(250,250),(0,0,255),thickness=3)
cv.putText(blank,"Hello",(250,250),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=1.0,thickness=1,color=(255,0,0))
cv.imshow("Blank",blank)
cv.waitKey(5000)
cv.destroyAllWindows()
'''
'''
#------>shapes section 2
#------>
#drawing four rectangles at the corners of size 50
size = 200
cv.rectangle(blank,(0,0),(size,size),color=(255,0,0),thickness=cv.FILLED)
cv.rectangle(blank,(blank.shape[1]-size,0),(blank.shape[1],size),color=(255,0,0),thickness=cv.FILLED)
cv.rectangle(blank,(0,blank.shape[0]-size),(size,blank.shape[0]),color=(255,0,0),thickness=cv.FILLED)
cv.rectangle(blank,(blank.shape[1]-size,blank.shape[0]-size),(blank.shape[1],blank.shape[0]),color=(255,0,0),thickness=cv.FILLED)
cv.imshow("Blank",blank)
cv.waitKey(5000)
cv.destroyAllWindows()
'''
'''
#------>>> shapes section 3
#------>>>
#drawing n number of lines to turn the blank to a grid
num_lines = 4
for i in range(1,num_lines+1):
    cv.line(blank,(blank.shape[1]//(num_lines+1)*i,0),(blank.shape[1]//(num_lines+1)*i,blank.shape[0]),color=(0,0,0),thickness=1)
    cv.line(blank,(0,(blank.shape[0]//(num_lines+1))*i),(blank.shape[0],(blank.shape[0]//(num_lines+1))*i),color=(0,0,0),thickness=1)
cv.imshow("image",blank)
cv.waitKey(5000)
cv.destroyAllWindows()
'''
#------>>> section 3 essential functions
'''
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)
cv.waitKey(5000)
cv.destroyAllWindows()
'''
'''
#blur
blur = cv.GaussianBlur(img,(31,31),cv.BORDER_REFLECT)
cv.imshow("Blurred",blur)

#edge detection using canny
canny = cv.Canny(img,175,175)
cv.imshow("Edges",canny)
cv.waitKey(5000)
cv.destroyAllWindows()
'''

#----->essential function cropping
'''
cropped = img[100:400,400:700]
cv.imshow("Cropped",cropped)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#-----> section 4 geometric transformation
'''
#we use something called warpAffine
#it takes a 2x3 matrix [a b tx][c d ty] allows all transformations
def translate(src,tx,ty):
    transMat = np.float32([[1,0,tx],[0,1,ty]])
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)
translatedImage = translate(img,200,200)
cv.imshow("Translated Image",translatedImage)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#rotation
'''
def rotation(src,angle,rotPoint=None):
    (height,width) = src.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2,height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
    return cv.warpAffine(src,rotMat,(width,height))
rotatedImage = rotation(img,45)
cv.imshow("Rotated Image",rotatedImage)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#resize
'''
def resize(src,scale):
    return cv.resize(src,(int(img.shape[1]*scale),int(img.shape[0]*scale)),interpolation=cv.INTER_CUBIC)
resizedImage = resize(img,0.5)
cv.imshow("Resized",resizedImage)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#flip
'''
flippedImage = cv.flip(img,0)
cv.imshow("Flipped",flippedImage)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#----->section 5 color space
#transformation to other scale
'''
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)
cv.imshow("LAB",lab)
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
#splitting into channels
b,g,r = cv.split(img)
blank = np.zeros((img.shape[:2]),dtype=np.uint8)
blue = cv.merge([b,blank,blank])
cv.imshow("blue",blue)#to see the channel in its own color
cv.imshow("green",g)
cv.imshow("red",r)
cv.waitKey(0)
cv.destroyAllWindows()

#merge these channels back 
mergedImage = cv.merge([b,g,r])
cv.imshow('originalCopy',mergedImage)
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
ycbcr = cv.cvtColor(img,cv.COLOR_BGR2YCrCb)
y,cr,cb = cv.split(ycbcr)
cv.imshow("luminance",y)
cv.imshow("chrominance red",cr)
cv.imshow("chrominance blue",cb)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#------>>section 6 blurring
#------>>
'''
#averaging blur
blurredImage = cv.blur(img,(7,7))
cv.imshow("blurred",blurredImage)
#gaussian blurred
gaussed = cv.GaussianBlur(img,(7,7),0)
cv.imshow("gaussed",gaussed)
cv.waitKey(0)
cv.destroyAllWindows()
'''
#--------->>section 7 bitwise operations
#--------->>
'''
blank = np.zeros((500,500),dtype=np.uint8)
rectangle = cv.rectangle(blank.copy(),(50,50),(475,475),255,-1)
circle = cv.circle(blank.copy(),(200,200),(200),255,-1)
#intersecting region
bit_and = cv.bitwise_and(rectangle,circle)
#intersection and non intersecting
bit_or = cv.bitwise_or(rectangle,circle)
#only non intersecting
bit_xor = cv.bitwise_xor(rectangle,circle)
cv.imshow("Bitwise XOR",bit_xor)
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
#---------->>section 8 histogram operations
#-------->>
#histogram for only the grayscale
blank = np.zeros(img.shape[:2],dtype="uint8")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
mask = cv.bitwise_and(gray,gray,mask=cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1))
cv.imshow("Masked img",mask)
hist = cv.calcHist([gray],[0],None,[256],[0,256])
#the parameters for hist are: 
#src in a list
#what element to calculate hist for
#set it to None if not required it is mask
#number of bins
#range of these bins 
plt.figure()

plt.xlabel("bins")
plt.xlim((0,256))
plt.ylabel("no of pixel")
plt.plot(hist)
plt.show()
'''
'''
#histogram for bgr space
plt.figure()
color = ("b","g","r")
for i,c in enumerate(color):
    hist = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color=c)
    plt.xlim([0,256])
plt.show()
'''
'''
#histogram equalisation
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
hist = cv.calcHist([gray],[0],None,[256],[0,256])
cdf = hist.cumsum()
cdfNorm = cdf*(hist.max())/cdf.max()
plt.figure()
plt.plot(hist)
plt.plot(cdf,color="r")
plt.show()

equImg = cv.equalizeHist(gray)
cv.imshow("original",gray)
cv.imshow("equalised",equImg)
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
#------------>> section 9 Thresholding
#--_-----_____->> binary thresholding
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
threshold,thresh = cv.threshold(gray,150,255,cv.THRESH_BINARY)
#inverse binary thresholding
threshold,thresh_inv = cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
#adaptive thresholding
adaptive_threshold = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,5)
cv.imshow("Adaptive",adaptive_threshold)
cv.imshow("INV THRESHOLDED IMAGE",thresh_inv)
cv.imshow("THRESHOLDED IMAGE",thresh)
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
#---------->>>> section 10 edge detection
#--------->>>
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Original: ",gray)
laplacian = cv.Laplacian(gray,cv.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
cv.imshow("edges",laplacian)
cv.waitKey(0)
cv.destroyAllWindows()
'''