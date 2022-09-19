valid_extensions = ('.jpg', '.jpeg', '.png')

import os
# remove the files that are not images
for filename in os.listdir("chessboards"):
    print(filename)
    if not filename.endswith(valid_extensions):
        os.remove("chessboards/" + filename)
#renames all the files in input to the format "image_1.jpg", "image_2.jpg", etc.
import os
for index, filename in enumerate(os.listdir("chessboards")):
    os.rename("chessboards/" + filename, "input/image_" + str(index) + ".jpg")
