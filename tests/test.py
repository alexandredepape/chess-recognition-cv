import os

from predictor import detect_2dChessboard
import cv2 as cv

if __name__ == '__main__':
    os.system("rm -rf output/*")
    # create the ouput folder if it doesn't exist:
    if not os.path.exists("output"):
        os.makedirs("output")

    images = "../assets/chessboard_images"
    nb_test_passed = 0
    nb_test_failed = 0
    for index, filename in enumerate(os.listdir(images)):
        valid_extensions = ('.jpg', '.jpeg', '.png')

        # check if the file is an image
        if not filename.lower().endswith(valid_extensions):
            continue
        print("Processing image: ", index, "out of ", len(os.listdir(images)))
        original_image = cv.imread(os.path.join(images, filename))

        cropped = detect_2dChessboard(original_image, filename)

        if cropped is None:
            print("No chessboard found")
            nb_test_failed += 1
        else:
            nb_test_passed += 1
    print("Tests passed: ", nb_test_passed, "out of ", len(os.listdir(images)))
