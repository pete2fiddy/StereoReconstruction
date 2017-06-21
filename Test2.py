from PIL import Image
import cv2
import numpy as np
import timeit

WINDOW_SIZE = 5
WINDOW_MARGIN = (WINDOW_SIZE-1)//2
RESIZE_FACTOR = 1
POINT_SEARCH_MARGIN = 160//RESIZE_FACTOR
TOP_MATCH_CUTOFF = 0.01#0.005#.01
NUM_SIMILAR_MATCH_THRESHOLD = 75#25

FOCAL_LENGTH_PIXELS = 5299.313/float(RESIZE_FACTOR)
BASELINE_MM = 177.288
D_OFFS=174.186
PPMM = D_OFFS/BASELINE_MM
MMPP = 1.0/PPMM
FOCAL_LENGTH_MM = FOCAL_LENGTH_PIXELS * MMPP

'''returns the tuple (x,y) of the identical feature in the right image'''
def match_feature(left_xy, l_image, r_image):
    sub_image = r_image[:left_xy[0], left_xy[1] - WINDOW_MARGIN : left_xy[1] + WINDOW_MARGIN + 1]
    left_template = l_image[left_xy[0] - WINDOW_MARGIN : left_xy[0] + WINDOW_MARGIN, left_xy[1] - WINDOW_MARGIN: left_xy[1] + WINDOW_MARGIN + 1]
    template_match_image = cv2.matchTemplate(sub_image, left_template, method = cv2.TM_CCORR_NORMED)

    sorted_template_matches = sorted(template_match_image.squeeze().tolist(), reverse = True)
    num_matches = 0
    for i in range(0, len(sorted_template_matches)):
        if sorted_template_matches[0] - sorted_template_matches[i] < TOP_MATCH_CUTOFF:
            num_matches += 1
        else:
            break
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match_image)
    return tuple((np.array(tuple(reversed(max_loc))) + np.array([WINDOW_MARGIN, left_xy[1]])).tolist()), num_matches

def create_depth_image(l_image, r_image):
    out_image = np.zeros(l_image.shape)
    principal_point = np.array([l_image.shape[0]//2, l_image.shape[1]//2])
    print("out image shape: ", out_image.shape)
    points = []
    for x in range(POINT_SEARCH_MARGIN, l_image.shape[0]-POINT_SEARCH_MARGIN):
        for y in range(WINDOW_MARGIN, l_image.shape[1] - WINDOW_MARGIN):
            feature_in_r, num_similar_responses = match_feature((x,y), l_image, r_image)
            if feature_in_r[0] != x and num_similar_responses < NUM_SIMILAR_MATCH_THRESHOLD:
                #depth = 1.0/abs(feature_in_r[0] - x)
                feature_xyz = point_of_same_feature((x,y), feature_in_r, principal_point)
                points.append((feature_xyz/100.0).tolist())
                out_image[x,y] = feature_xyz[2]
        print("column traversed")
    out_image_min = np.amin(out_image)
    out_image_max = np.amax(out_image)

    points_with_y_as_depth = []
    for i in range(0, len(points)):
        points_with_y_as_depth.append(np.array([points[i][0], points[i][2], points[i][1]]))

    with open(base_path + "calculated points", 'w') as point_output:
        point_output.write(str(str(points_with_y_as_depth)[1:len(str(points_with_y_as_depth))-1]))
        point_output.close()


    out_image = (out_image - out_image_min)/(out_image_max - out_image_min)
    return out_image

def point_of_same_feature(left_xy, right_xy, principal_point):
    z = BASELINE_MM*FOCAL_LENGTH_PIXELS/((left_xy[0]-right_xy[0]) + D_OFFS)#FOCAL_LENGTH_MM * BASELINE_MM/float(left_xy[0] - right_xy[0])
    x = float(left_xy[0] - principal_point[0]) * z/FOCAL_LENGTH_MM
    y = float(principal_point[1] - left_xy[1]) * z/FOCAL_LENGTH_MM
    return np.array([x,y,z])


base_path = "/Users/phusisian/Desktop/DZYNE/Python/Stereo Reconstruction/Test Images/Middlebury/2014/Bicycle/"

l_img = Image.open(base_path + "im0.png")
r_img = Image.open(base_path + "im1.png")

resize_dim = (l_img.size[0]//RESIZE_FACTOR, l_img.size[1]//RESIZE_FACTOR)
l_img = l_img.resize(resize_dim)
r_img = r_img.resize(resize_dim)

l_image = cv2.cvtColor(np.array(l_img), cv2.COLOR_RGB2GRAY).T
r_image = cv2.cvtColor(np.array(r_img), cv2.COLOR_RGB2GRAY).T

test_xy = (2243,144)
start_time = timeit.default_timer()
Image.fromarray(255*create_depth_image(l_image, r_image).T).show()
print("time elapsed: ", timeit.default_timer() - start_time)
