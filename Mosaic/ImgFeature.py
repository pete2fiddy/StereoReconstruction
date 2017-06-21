import numpy as np
import cv2
from PIL import Image
import VectorMath.VectorMath as VectorMath
from math import pi, cos, sin
from random import randint
import ImageOp.ImageFunctions as ImageFunctions

class ImgFeature:
    '''takes img1_xy and img2_xy, where img1_xy and img2_xy are vectors relative to the center point of their respective images.
    both variables belong to vectors that point to features that are the same across two images.'''
    def __init__(self, img1_xy, img2_xy):
        self.base_vec = img1_xy
        self.compare_vec = img2_xy

    def draw(self, base_image, compare_image, radius):
        base_point = ImgFeature.vec_to_image_xy(self.base_vec, base_image)
        compare_point = ImgFeature.vec_to_image_xy(self.compare_vec, compare_image)
        rand_matching_color = (randint(50, 254), randint(50, 254), randint(50, 254))
        cv2.circle(base_image, base_point, radius, (rand_matching_color), thickness = 5)
        cv2.circle(compare_image, compare_point, radius, rand_matching_color, thickness = 5)

    '''where base_centroid is the average vector of all base_vecs, and compare_centroid is the average vector of all
    compare_vecs'''
    def angle_between(self, base_centroid, compare_centroid):
        sub_base_vec = self.base_vec - base_centroid
        sub_compare_vec = self.compare_vec - compare_centroid
        #print("sub base vec: ", sub_base_vec)
        #print("sub compare vec: ", sub_compare_vec)
        dot_angle_between = VectorMath.dot_angle_between(sub_base_vec, sub_compare_vec)
        #print("dot angle between: ", dot_angle_between)
        return dot_angle_between

    @classmethod
    def init_with_image_xy(self, xy1, xy2, image1, image2):
        img1_xy = ImgFeature.image_xy_to_vec(xy1, image1)
        img2_xy = ImgFeature.image_xy_to_vec(xy2, image2)
        return ImgFeature(img1_xy, img2_xy)

    @staticmethod
    def image_xy_to_vec(image_xy, image):
        out_vec = np.array(image_xy) - np.array(np.array([image.shape[1], image.shape[0]]))/2.0
        out_vec[1] = -out_vec[1]
        return out_vec

    @staticmethod
    def vec_to_image_xy(vec, image):
        image_midpoint = np.array([image.shape[1], image.shape[0]])/2.0
        #print("image midpoint: ", image_midpoint)
        xy_point = tuple((image_midpoint + np.array([vec[0], -vec[1]])).astype(np.int).tolist())
        return xy_point



    def displacement(self):
        return self.compare_vec - self.base_vec

    def __repr__(self):
        return "Img Feature Vec 1: " + str(self.base_vec) + ", " + str(self.compare_vec)

class ImgFeatures:
    def __init__(self, linked_features):
        self.linked_features = linked_features
        self.init_feature_points()
        '''has to be init'd after rotation as rotation will mess up average displacement'''
        self.init_image_centroids()
        self.init_avg_angle_between()
        self.rotate_compare_features_by_avg_theta()
        self.init_average_displacement()



    '''for operations where it is more convenient to have two sets of vectors, one set for the first, base img, and another for the compare,
    2nd img.'''
    def init_feature_points(self):
        self.base_img_vecs = np.zeros((len(self.linked_features), 2))
        self.compare_img_vecs = np.zeros((len(self.linked_features), 2))
        for i in range(0, len(self.linked_features)):
            append_base_img_vec = self.linked_features[i].base_vec
            append_compare_img_vec = self.linked_features[i].compare_vec
            self.base_img_vecs[i] = append_base_img_vec
            self.compare_img_vecs[i] = append_compare_img_vec

    def init_image_centroids(self):
        base_vec_sum = np.zeros((2))
        compare_vec_sum = np.zeros((2))
        for i in range(0, len(self.linked_features)):
            base_vec_sum += self.linked_features[i].base_vec
            compare_vec_sum += self.linked_features[i].compare_vec
        self.base_centroid = base_vec_sum / float(len(self.linked_features))
        self.compare_centroid = compare_vec_sum / float(len(self.linked_features))

    '''initializes the average point of the base_img vecs and the compare_img vecs.'''
    def init_average_displacement(self):
        displacement_sum = np.zeros((2))
        for i in range(0, len(self.linked_features)):
            displacement_sum += self.linked_features[i].displacement()
        self.avg_displacement = displacement_sum / float(len(self.linked_features))
        print("avg displacement: ", self.avg_displacement)

    def init_avg_angle_between(self):
        theta_sum = 0
        for i in range(0, len(self.linked_features)):
            theta_sum += self.linked_features[i].angle_between(self.base_centroid, self.compare_centroid)
        self.avg_theta = theta_sum / float(len(self.linked_features))
        print("avg theta: ", 360.0 * (self.avg_theta/(2.0*pi)))

    def rotate_compare_features_by_avg_theta(self):
        rot_matrix1 = np.array([[cos(self.avg_theta), -sin(self.avg_theta)],
                              [sin(self.avg_theta), cos(self.avg_theta)]])
        '''since the dot angle between two vectors does not give the correct direction to rotate a to meet b,
        the rotation matrix is applied twice and the sum of the dot angle between is reassed. The smaller dot angle
        sum rotation is chosen and kept'''
        rot_matrix2 = np.array([[cos(-self.avg_theta), -sin(-self.avg_theta)],
                              [sin(-self.avg_theta), cos(-self.avg_theta)]])
        compare_img_vecs1 = []
        compare_img_vecs2 = []
        for i in range(0, len(self.compare_img_vecs)):
            compare_img_vecs1.append(self.compare_centroid + rot_matrix1.dot(self.compare_img_vecs[i] - self.compare_centroid))
            compare_img_vecs2.append(self.compare_centroid + rot_matrix2.dot(self.compare_img_vecs[i] - self.compare_centroid))

        dot_angle_sum1 = 0
        dot_angle_sum2 = 0
        for i in range(0, len(compare_img_vecs1)):
            '''both dot products do not rotate around their centroids but does not matter since it is only measuring angle between'''
            dot_angle_sum1 += VectorMath.dot_angle_between(compare_img_vecs1[i] - self.compare_centroid, self.base_img_vecs[i] - self.base_centroid)
            dot_angle_sum2 += VectorMath.dot_angle_between(compare_img_vecs2[i] - self.compare_centroid, self.base_img_vecs[i] - self.base_centroid)

        self.compare_img_vecs = compare_img_vecs1 if dot_angle_sum1 < dot_angle_sum2 else compare_img_vecs2
            #self.compare_img_vecs[i] = rot_matrix.dot(self.compare_img_vecs[i])

    def draw_point_averages(self, base_image, compare_image):
        base_out = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
        compare_out = cv2.cvtColor(compare_image, cv2.COLOR_GRAY2RGB)

        base_point = ImgFeature.vec_to_image_xy(np.zeros((2)), base_image)#(base_image.shape[0]//2, base_image.shape[1]//2)
        #print("base point: ", base_point)
        compare_point = ImgFeature.vec_to_image_xy(self.avg_displacement, base_image)
        #compare_point = (int(base_point[0] + self.avg_displacement[0]), int(base_point[1] - self.avg_displacement[1]))#(int(base_point[0] + self.avg_displacem), int(base_point[1] - diff_between_avgs[0]))

        base_centroid_point = ImgFeature.vec_to_image_xy(self.base_centroid, base_image)
        compare_centroid_point = ImgFeature.vec_to_image_xy(self.compare_centroid, compare_image)

        cv2.circle(base_out, base_point, 10, (255,0,0), thickness = 5)
        cv2.circle(base_out, base_centroid_point, 10, (255, 255, 0), thickness = 5)
        cv2.circle(compare_out, compare_centroid_point, 10, (0,255,255), thickness = 5)
        cv2.circle(compare_out, base_point, 10, (0,255,0), thickness = 5)
        cv2.circle(compare_out, compare_point, 10, (255,255,255), thickness = 5)

        for i in range(0, len(self.linked_features)):
            self.linked_features[i].draw(base_out, compare_out, 5)



        Image.fromarray(base_out).show()
        Image.fromarray(compare_out).show()

    def fit_compare_to_base(self, compare_image):
        out_image = ImageFunctions.rotate_image(compare_image, ImgFeature.vec_to_image_xy(self.compare_centroid, compare_image), self.avg_theta)
        Image.fromarray(out_image).show()


    def __repr__(self):
        out_str = "Img Features: \n"
        for i in range(0, len(self.linked_features)):
            out_str += str(self.linked_features[i])
            if i < len(self.linked_features)-1:
                out_str += "\n"
        return out_str
