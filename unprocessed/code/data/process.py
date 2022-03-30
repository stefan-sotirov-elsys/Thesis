import os
import cv2

directories_paths = ["../../datasets/training/clean/", "../../datasets/training/dirty/"]

directories_count = 2

directories_contents = []

directories_sizes = []

total = 0

for dir_path in directories_paths:

    directories_contents.append(os.listdir(dir_path))

    directories_sizes.append(len(directories_contents[-1]))

    total += directories_sizes[-1]

width = 224

height = 224

processed = 1

for i in range(0, directories_count):

    for file_name in directories_contents[i]:
 
        img_path = directories_paths[i] + file_name
 
        print(img_path) # debug

        img = cv2.imread(img_path)

        if img.shape[0] == width:

            # the image has been processed

            print(str(processed) + " / " + str(total)) # debug

            processed += 1

            continue

        # resize and save

        img = cv2.resize(img, (width, height))
 
        cv2.imwrite(img_path, img)

        print(str(processed) + " / " + str(total)) # debug

        processed += 1