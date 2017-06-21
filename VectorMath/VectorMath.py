from math import acos
import numpy as np

def dot_angle_between(v1, v2):

    return acos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
