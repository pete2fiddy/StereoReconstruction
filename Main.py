from PIL import Image
import numpy as np
import cv2
import scipy
import timeit
from math import atan2, pi, tan, cos, sin
from MosaicMaker import MosaicMaker2
import Mosaic.MatchedPointLoader as MatchedPointLoader

mosaic_path = "/Users/phusisian/Desktop/DZYNE/Mosaicing Data/Toy Data/Black-Square-Translation-And-Rotation/"
mosaic_imgs = [Image.open(mosaic_path + "Before.png"), Image.open(mosaic_path + "After.png")]
base_image = cv2.cvtColor(np.array(mosaic_imgs[0]), cv2.COLOR_RGB2GRAY)
compare_image = cv2.cvtColor(np.array(mosaic_imgs[1]), cv2.COLOR_RGB2GRAY)
#mosaic_maker = MosaicMaker2(mosaic_imgs)
img_features = MatchedPointLoader.load_matched_points(mosaic_path + "feature_matches.txt", np.array(mosaic_imgs[0]), np.array(mosaic_imgs[1]))
img_features.draw_point_averages(base_image, compare_image)
img_features.fit_compare_to_base(compare_image)
