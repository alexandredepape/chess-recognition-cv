# clear the ouput folder
import math
import os
import traceback

import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull


# put the code above in a function
def order_points(convex_hull_points):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = convex_hull_points.sum(axis=1)
    rect[0] = convex_hull_points[np.argmin(s)]
    rect[2] = convex_hull_points[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(convex_hull_points, axis=1)
    rect[1] = convex_hull_points[np.argmin(diff)]
    rect[3] = convex_hull_points[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(original_image, convex_hull_points):
    # obtain a consistent order of the points and unpack them
    # individually
    convex_hull_points = np.array(convex_hull_points)
    convex_hull_points = convex_hull_points.reshape(4, 2)
    rect = order_points(convex_hull_points)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(original_image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def cluster_1d_data(data):
    from sklearn.cluster import DBSCAN
    import numpy as np
    X = np.array(data).reshape(-1, 1)
    db = DBSCAN(eps=5, min_samples=1).fit(X)
    return db.labels_


def detect_2dChessboard(original_image, filename):
    print("Detecting 2D chessboard...")
    STEPS = [greyscale,
             # equalisation,
             # gaussianblur,
             cannyedge,
             # gaussianblur,
             ]

    step_images = [original_image]
    try:

        image = original_image.copy()
        for step in STEPS:
            image = step(image, step_images)

        contours, contours_image = find_contours(image, original_image, step_images)
        contours, contours_image = find_contours(contours_image, original_image, step_images)

        # find the largest contour
        largest_contour = max(contours, key=cv.contourArea)
        # find the convex hull of the largest contour
        biggest_contour_image = get_black_image(original_image)
        cv.drawContours(biggest_contour_image, [largest_contour], -1, (0, 255, 0), 2)
        step_images.append(biggest_contour_image)

        hough_lines_image, lines = hough_lines(biggest_contour_image, original_image, step_images)
        merged_lines, bundled_image = bundle_lines(lines, original_image, step_images, min_distance=100)

        # sort the lines by groups of 10 degress orientation
        orientations = {}
        for line in merged_lines:
            orientation = HoughBundler().get_orientation(line[0])
            if orientation not in orientations:
                orientations[orientation] = []
            orientations[orientation].append(line)

        labels = cluster_1d_data(list(orientations.keys()))
        orientations_clusters = {}
        for i, label in enumerate(labels):
            if label not in orientations_clusters:
                orientations_clusters[label] = []
            orientations_clusters[label].append(orientations[list(orientations.keys())[i]][0])
        # create a dict with the two orientations clusters with the most lines
        orientations_clusters = dict(
            sorted(orientations_clusters.items(), key=lambda item: len(item[1]), reverse=True)[:2])


        # find the intersection of the lines
        black_image = get_black_image(original_image)
        for orientation_group, lines in orientations_clusters.items():
            # draw the lines
            random_color = list(np.random.randint(0, 255, 3))
            color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
            for line in lines:
                # line is of the form [[x1, y1, x2, y2]]
                cv.line(black_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), color, 3)
        step_images.append(black_image)

        merged_lines1, _ = bundle_lines(list(orientations_clusters.values())[0], original_image, step_images,
                                        min_distance=100)
        merged_lines2, _ = bundle_lines(list(orientations_clusters.values())[1], original_image, step_images,
                                        min_distance=100)

        clustered_lines = list(merged_lines1) + list(merged_lines2)
        extend_lines_image, extended_lines = extend_lines(clustered_lines, hough_lines_image, step_images)

        merged_lines, bundled_image = bundle_lines(extended_lines, original_image, step_images, min_distance=100)

        merged_lines = [((line[0][0], line[0][1]), (line[0][2], line[0][3])) for line in merged_lines]
        line_intersections = get_line_intersections(merged_lines, original_image)

        black_image = bundled_image.copy()
        for coord in line_intersections:
            cv.circle(black_image, (coord[0], coord[1]), 20, (0, 0, 255), -1)
        step_images.append(black_image)
        # filter out points that are not in the image
        convex_hull = ConvexHull(line_intersections)
        convex_hull_image = get_black_image(original_image)
        # find the 4 points of the convex hull
        # draw the convex hull
        line_intersections = np.array(line_intersections)
        cv.drawContours(convex_hull_image, [line_intersections[convex_hull.vertices]], -1, (0, 255, 0), 2)
        step_images.append(convex_hull_image)

        hough_lines_image, lines = hough_lines(convex_hull_image, original_image, step_images)

        merged_lines, bundled_image = bundle_lines(lines, original_image, step_images)

        extend_lines_image, extended_lines = extend_lines(merged_lines, hough_lines_image, step_images)
        extended_lines = [((line[0][0], line[0][1]), (line[0][2], line[0][3])) for line in extended_lines]
        line_intersections = get_line_intersections(extended_lines, original_image)

        black_image = get_black_image(original_image)
        draw_points(black_image, line_intersections, step_images)

        black_image = original_image.copy()
        draw_points(black_image, line_intersections, step_images)

        cropped = four_point_transform(original_image, line_intersections)
        # integrate the cropped image into the step images by putting it in the middle of a black image
        black_image = get_black_image(original_image)
        black_image[0:cropped.shape[0], 0:cropped.shape[1]] = cropped
        step_images.append(black_image)
        save_images(filename, step_images)
        return cropped
    except Exception as e:
        # print the exception tracback
        traceback.print_exc()
        save_images(filename, step_images)


def draw_points(black_image, line_intersections, step_images):
    for coord in line_intersections:
        cv.circle(black_image, (coord[0], coord[1]), 20, (0, 255, 0), -1)
    step_images.append(black_image)


def get_line_intersection(line1, line2):
    from shapely.geometry import LineString

    line1 = LineString(line1)
    line2 = LineString(line2)

    int_pt = line1.intersection(line2)
    if int_pt.is_empty:
        return None

    return int_pt.x, int_pt.y


def get_intersections(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            # find the intersection of the two lines
            intersection = get_line_intersection(lines[i], lines[j])
            if intersection is not None:
                intersections.append(intersection)
    return intersections


def get_line_intersections(lines, original_image):
    try:
        line_intersections = get_intersections(lines)
        line_intersections = np.round(line_intersections).astype(int)
        line_intersections = [coord for coord in line_intersections if
                              0 <= coord[0] < original_image.shape[1] and 0 <= coord[1] < original_image.shape[0]]
        return line_intersections
    except ValueError:
        lines = [((line[0][0], line[0][1]), (line[0][2], line[0][3])) for line in lines]
        return get_line_intersections(lines, original_image)


def bundle_lines(lines, image, step_images, min_distance=10):
    # bundle the lines
    lines = np.array(lines)
    merged_lines = HoughBundler(min_distance=min_distance, min_angle=10).process_lines(lines)
    output = drawLines(merged_lines, image, step_images)
    return merged_lines, output


def dist2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def fuse(points, d):
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point1 = points[i]
            point1_coordinates = points[i].copy()
            taken[i] = True
            for j in range(i + 1, n):
                point2 = points[j]
                dist_squared = dist2(point1_coordinates, point2)
                if dist_squared < d2:
                    point1[0] += point2[0]
                    point1[1] += point2[1]
                    count += 1
                    taken[j] = True
            point1[0] /= count
            point1[1] /= count
            point1[0] = round(point1[0])
            point1[1] = round(point1[1])
            ret.append((point1[0], point1[1]))
    return ret


def draw_contours(contours, original_image, step_images):
    # contours = sorted(contours, key=cv.contourArea, reverse=True)[:1]
    contours_image = get_black_image(original_image)
    for contour in contours:
        random_color = list(np.random.randint(0, 255, 3))
        color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
        cv.drawContours(contours_image, [contour], -1, color, 3)
    step_images.append(contours_image)
    return contours


def save_images(filename, step_images):
    # save a concatenation of all the images with a max with of 5 images
    # and a max height of 5 images
    max_width = 5
    max_height = 5
    height, width, channels = step_images[0].shape
    final_image = np.zeros((height * max_height, width * max_width, channels), np.uint8)
    for i in range(len(step_images)):
        x = i % max_width
        y = i // max_width
        # make the image 80% of the original size and center the image

        resized_image = cv.resize(step_images[i], (0, 0), fx=0.8, fy=0.8)
        resized_height, resized_width, channels = resized_image.shape
        x_offset = int((width - resized_width) / 2)
        y_offset = int((height - resized_height) / 2)
        final_image[y * height + y_offset:y * height + y_offset + resized_height,
        x * width + x_offset:x * width + x_offset + resized_width] = resized_image
    name, extension = os.path.splitext(filename)
    cv.imwrite("output/" + name + "_steps" + extension, final_image)


def find_contours(image, original_image, step_images):
    try:
        contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_image = get_black_image(original_image)
        for contour in contours:
            random_color = list(np.random.randint(0, 255, 3))
            color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
            cv.drawContours(contours_image, [contour], -1, color, 5)
        step_images.append(contours_image)
        return contours, contours_image
    except cv.error:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return find_contours(image, original_image, step_images)


def hough_lines(image, original_image, step_images):
    try:
        theta = np.pi / 180
        rho = 1
        threshold = 50
        lines = None
        minLineLength = 200
        maxLineGap = 10
        p = cv.HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap)
        lines = p
    except cv.error:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return hough_lines(image, original_image, step_images)
    black_image = get_black_image(original_image)
    drawLines(lines, original_image, step_images)
    return cv.cvtColor(black_image, cv.COLOR_BGR2GRAY), lines


def extend_lines(lines, original_image, step_images):
    if lines is not None:
        # extend lines :
        extended_lines = []
        for line in lines:
            line = line[0]
            p1, p2 = extend_line((line[0], line[1]), (line[2], line[3]), distance=original_image.shape[0])
            extended_lines.append([[p1[0], p1[1], p2[0], p2[1]]])
        image_with_lines = drawLines(extended_lines, original_image, step_images, on_origin_image=False)
        return image_with_lines, extended_lines


def cornerharris(image, original_image, step_images):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corner_harris = cv.cornerHarris(image, 2, 3, 0.04)
    result = get_black_image(original_image)
    result[corner_harris > 0.01 * corner_harris.max()] = [0, 255, 0]
    step_images.append(result)
    return cv.cvtColor(result, cv.COLOR_BGR2GRAY)


def drawLines(lines, original_image, step_images, on_origin_image=False):
    black_image = get_black_image(original_image)
    if on_origin_image:
        black_image = original_image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # x1, y1, x2, y2 = line
            random_color = list(np.random.randint(0, 255, 3))
            color = (int(random_color[0]), int(random_color[1]), int(random_color[2]))
            cv.line(black_image, (x1, y1), (x2, y2), color, 5, cv.LINE_AA)
    step_images.append(black_image)
    return black_image


def get_black_image(original_image):
    # get a grey scale image
    black_image = np.zeros((original_image.shape[0], original_image.shape[1], 3), np.uint8)

    return black_image


def greyscale(original_image, step_images):
    # convert to grayscale
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    # step_images.append(cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR))

    return gray_image


def cannyedge(blur_image, step_images):
    # apply canny edge detection
    canny_image = cv.Canny(blur_image, 100, 200)
    step_images.append(cv.cvtColor(canny_image, cv.COLOR_GRAY2BGR))
    return canny_image


def gaussianblur(equalised, step_images):
    # apply gaussian blur
    blur_image = cv.GaussianBlur(equalised, (3, 3), 0)
    step_images.append(cv.cvtColor(blur_image, cv.COLOR_GRAY2BGR))
    return blur_image


def equalisation(gray_image, step_images):
    # apply histogram equalization
    equalised = cv.equalizeHist(gray_image)
    step_images.append(cv.cvtColor(equalised, cv.COLOR_GRAY2BGR))
    return equalised


def extend_line(p1, p2, distance=10000):
    diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    p3_x = int(p1[0] + distance * np.cos(diff))
    p3_y = int(p1[1] + distance * np.sin(diff))
    p4_x = int(p1[0] - distance * np.cos(diff))
    p4_y = int(p1[1] - distance * np.sin(diff))
    return (p3_x, p3_y), (p4_x, p4_y)


def get_orientation(line: [int, int, int, int]):
    orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
    return math.degrees(orientation)


def get_average_orientation(lines):
    orientations = []
    for line in lines:
        orientations.append(get_orientation(line))
    print(orientations)
    mean = np.mean(orientations)
    print(mean)
    return np.average(orientations)


def merge_line_segments(lines):
    orientation = get_orientation(lines[0])
    if len(lines) == 1:
        return np.block([[lines[0][:2], lines[0][2:]]])

    points = []
    for line in lines:
        points.append(line[:2])
        points.append(line[2:])
    if 45 < orientation <= 90:
        # sort by y
        points = sorted(points, key=lambda point: point[1])
    else:
        # sort by x
        points = sorted(points, key=lambda point: point[0])

    return np.block([[points[0], points[-1]]])


class HoughBundler:
    # https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
    def __init__(self, min_distance=5, min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle

    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < self.min_distance:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < self.min_angle:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing groups, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups):
                groups.append([line_new])

        return groups

    def process_lines(self, lines):
        if lines is None:
            print("No lines found to merge")
            return lines
        lines_horizontal = []
        lines_vertical = []

        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical = sorted(lines_vertical, key=lambda line: line[1])
        lines_horizontal = sorted(lines_horizontal, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(merge_line_segments(group))
                merged_lines_all.extend(merged_lines)

        return np.asarray(merged_lines_all)



def clean_ouput_folder():
    os.system("rm -rf output/*")
    # create the ouput folder if it doesn't exist:
    if not os.path.exists("output"):
        os.makedirs("output")
