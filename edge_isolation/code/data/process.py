import cv2
import os
import numpy
 
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

        # isolate the container

        img = cv2.imread(img_path)

        if img.shape[0] == width:

            # the image has been processed

            print(str(processed) + " / " + str(total)) # debug

            processed += 1

            continue
 
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
        hue = (img_hsv[:, :, 0] < 100) * (img_hsv[:, :, 0] > 3)
    
        saturation = (img_hsv[:, :, 1] > 70)
    
        value = (img_hsv[:, :, 2] > 120)
    
        mask = saturation * value * hue
    
        img_hsv[:, :, 0] *= mask
    
        img_hsv[:, :, 1] *= mask
    
        img_hsv[:, :, 2] *= mask
    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
 
        # isolate the edges

        img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    
        median = numpy.median(img)
    
        lower = int(max(0, 0.66 * median))
    
        upper = int(min(255, 1.33 * median))
    
        img = cv2.Canny(image = img, threshold1 = lower, threshold2 = upper)

        # resize and save

        img = cv2.resize(img, (width, height))
 
        cv2.imwrite(img_path, img)
 
        print(str(processed) + " / " + str(total)) # debug

        processed += 1