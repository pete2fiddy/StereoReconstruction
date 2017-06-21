from math import cos, sin, pi
import cv2
import numpy as np

'''angle is in radians'''
def rotate_image(image, origin, angle_radians):
    rot_mat = cv2.getRotationMatrix2D(origin, 360.0 * angle_radians/(2.0 * pi), 1.0)

    #rotated_size = tuple(image.shape * np.array([sin(angle), cos(angle)]))
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape)
    return rotated_image
