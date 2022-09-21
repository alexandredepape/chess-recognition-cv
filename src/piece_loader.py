# create an single image containing the first 20 black bishops images from assets/chess_pieces/black_bishop/ on a single image

import os
import random
import sys
import cv2 as cv

black_queen = "../assets/chess_pieces/black_king/"
bb_image_filenames = os.listdir(black_queen)
# take 100 random queens from the filenames
bb_image_filenames = [bb_image_filenames[i] for i in random.sample(range(len(bb_image_filenames)), 100)]
bb_images = [cv.imread(black_queen + filename) for filename in bb_image_filenames]
bb_images = [cv.resize(image, (100, 100)) for image in bb_images]
# create a single image containing all the images which is 1000x1000
bb_image = cv.vconcat([cv.hconcat(bb_images[0:10]), cv.hconcat(bb_images[10:20]), cv.hconcat(bb_images[20:30]), cv.hconcat(bb_images[30:40]), cv.hconcat(bb_images[40:50]), cv.hconcat(bb_images[50:60]), cv.hconcat(bb_images[60:70]), cv.hconcat(bb_images[70:80]), cv.hconcat(bb_images[80:90]), cv.hconcat(bb_images[90:100])])

cv.imwrite("output.png", bb_image)


