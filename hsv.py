import cv2
import numpy
red = numpy.uint8([[[50,183,75]]])
hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
print(hsv_red)