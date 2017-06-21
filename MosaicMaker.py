import numpy as np
import cv2
from PIL import Image
from scipy.signal import argrelextrema
from Mosaic.ImgFeature import ImgFeature, ImgFeatures

class MosaicMaker2:
    NUM_FEATURES_PER_IMAGE = 4#10
    FEATURE_QUALITY = 0.2
    FEATURE_MIN_DISTANCE = 10#100
    CROSS_CORRELATION_WINDOW = 15
    CROSS_CORRELATION_WINDOW_MARGIN = (CROSS_CORRELATION_WINDOW-1)//2
    def __init__(self, imgs):
        self.imgs = imgs
        self.images = []
        for i in range(0, len(imgs)):
            self.images.append(cv2.cvtColor(np.array(imgs[i]), cv2.COLOR_RGB2GRAY))
        self.init_image_feature_points()
        self.connect_image_feature_points()

    def init_image_feature_points(self):
        '''feature_points is an i x j array where i represents the index of the image that the features correspond to
        and j represents the jth feature of that image at i'''
        self.feature_points = []
        for image_index in range(0, len(self.imgs)):
            feature_image = self.images[image_index]
            image_features = cv2.goodFeaturesToTrack(feature_image, MosaicMaker2.NUM_FEATURES_PER_IMAGE, MosaicMaker2.FEATURE_QUALITY, MosaicMaker2.FEATURE_MIN_DISTANCE)[:, 0, :]
            iter_feature_points = []
            for i in range(0, image_features.shape[0]):
                iter_feature_points.append(FeaturePoint(self.images[image_index], image_index, image_features[i]))
            self.feature_points.append(iter_feature_points)



    '''all features in each image that will be used for mosaicing are already found. Each feature must be connected with
    its immediate neighbors feature by cross correlation or some other method'''
    def connect_image_feature_points(self):
        self.img_features = []
        for image_index in range(0, len(self.feature_points)-1):
            unmatched_feature_points = list(self.feature_points[image_index+1])

            iter_img_features = []

            for feature_index in range(0, len(self.feature_points[image_index])):
                if len(unmatched_feature_points) <= 0:
                    break
                feature_point1 = self.feature_points[image_index][feature_index]
                #print("feature point 1: ", feature_point1)
                '''matches feature_point1 to one of the feature points of the neighboring image remaining in unmatched_feature_points'''
                best_match_to_feature_point1_index = self.get_best_match_index_for_feature_point(feature_point1, unmatched_feature_points)
                iter_img_features.append(ImgFeature.init_with_image_xy(feature_point1.feature_xy_int, unmatched_feature_points[best_match_to_feature_point1_index].feature_xy_int, self.images[image_index], self.images[image_index + 1]))
                del unmatched_feature_points[best_match_to_feature_point1_index]
            '''is temporary to test toy data. Manually inputted feature matches'''
            self.img_features.append(ImgFeatures([ ImgFeature.init_with_image_xy(np.array([52, 48]), np.array([176, 138]), self.images[0], self.images[1]),
            ImgFeature.init_with_image_xy(np.array([119, 78]), np.array([226, 192]), self.images[0], self.images[1]),
            ImgFeature.init_with_image_xy(np.array([90, 145]), np.array([171, 242]), self.images[0], self.images[1]),
            ImgFeature.init_with_image_xy(np.array([22, 116]), np.array([122, 187]), self.images[0], self.images[1]), ]))
            #self.img_features.append(ImgFeatures(iter_img_features))
        #print("img features: ", self.img_features)
        self.img_features[0].draw_point_averages(self.images[0], self.images[1])


    '''returns the index of the feature point in feature_point_candidates that best matches the feature point of match_feature_point.
    Cross correlation has highest response at best match, so keep this in mind if using a different algorithm that it will sort
    responses in the wrong order if the best match is low'''
    def get_best_match_index_for_feature_point(self, match_feature_point, feature_point_candidates):
        best_match_index = 0
        best_match_score = self.get_match_score_between_points(match_feature_point, feature_point_candidates[0])
        for i in range(0, len(feature_point_candidates)):
            candidate_score = self.get_match_score_between_points(match_feature_point, feature_point_candidates[i])
            if candidate_score != None and candidate_score > best_match_score:
                best_match_score = candidate_score
                best_match_index = i
        return best_match_index

    '''gives the match score of the point at feature_point2 using feature_point1 as a comparator'''
    def get_match_score_between_points(self, feature_point1, feature_point2):

        feature_point1_template = feature_point1.get_feature_template(self.CROSS_CORRELATION_WINDOW)
        feature_point2_template = feature_point1.get_feature_template(self.CROSS_CORRELATION_WINDOW)
        '''template matching is not invariant to scale. Scale first or choose a different template match algorithm'''
        try:
            feature_point2_match_score = (cv2.matchTemplate(feature_point2_template, feature_point1_template, cv2.TM_CCORR_NORMED))[0,0]
            return feature_point2_match_score
        except:
            '''feature window is chopped off and a comparison can't be made'''
            return None



class FeaturePoint:
    def __init__(self, image, img_index, feature_xy):
        self.image = image
        self.img_index = img_index
        self.feature_xy = feature_xy
        self.feature_xy_int = self.feature_xy.astype(np.int)

    '''def shift_vec(self, dvec):
        self.vec += dvec

    def transform_vec_2d(self, mat):
        self.vec = mat.dot(self.vec)'''

    def get_feature_template(self, template_size):
        template_margin = (template_size-1)//2
        return self.image[self.feature_xy_int[0] - template_margin : self.feature_xy_int[0] + template_margin, self.feature_xy_int[1] - template_margin : self.feature_xy_int[1] + template_margin]

    def __repr__(self):
        return str(self.feature_xy)

    '''@staticmethod
    def average(feature_points):
        sum = np.zeros((2))
        for i in range(0, len(feature_points)):
            sum += feature_points[i].vec
        return sum/float(len(feature_points))

    @staticmethod
    def multi_shift_vec(feature_points, dvec):
        for i in range(0, len(feature_points)):
            feature_points[i].shift_vec(dvec)'''

'''
class ImgFeatures:
    def __init__(self, image, img_features):
        self.image = image
        self.img_features = img_features

    def __repr__(self):
        return str(self.img_features)


class ImgFeature:
    def __init__(self, feature_point1, feature_point2):
        self.feature_point1 = feature_point1
        self.feature_point2 = feature_point2



    def __repr__(self):
        return "Img feature: " +str(self.feature_point1) + ", " + str(self.feature_point2)
'''
