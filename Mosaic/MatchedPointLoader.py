from Mosaic.ImgFeature import ImgFeature, ImgFeatures
import numpy as np

'''loads a txt file in the format:

x1,0 ... x1,n \n
y1,0 ... y1,n \n
x2,0 ... x2,n \n
y2,0 ... y2,n

(with spaces separating numbers)
'''
def load_matched_points(path, base_image, compare_image):
    matched_vectors = np.loadtxt(open(path))
    print("matched vectors: ", matched_vectors)
    print("matched vectors shape: ", matched_vectors.shape)
    img_features = []
    for i in range(0, matched_vectors.shape[0]):
        base_xy = np.array([matched_vectors[0][i], matched_vectors[1][i]])
        compare_xy = np.array([matched_vectors[2][i], matched_vectors[3][i]])
        img_features.append(ImgFeature.init_with_image_xy(base_xy, compare_xy, base_image, compare_image))

    return ImgFeatures(img_features)
