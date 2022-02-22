import os
import shutil

directories_count = 2

directories_paths = ["../../dataset/clean/", "../../dataset/dirty/"]

directories_contents = []

directories_sizes = []

largest = 0

for path in directories_paths:

    directories_contents.append(os.listdir(path))

    print(path) # debug

    directories_sizes.append(len(directories_contents[-1]))

    print(directories_sizes[-1]) # debug

    if directories_sizes[-1] > largest:

        largest = directories_sizes[-1]

print(largest) # debug


for i in range(0, directories_count):
        
    size_difference = largest - directories_sizes[i]

    j = 0

    k = 0

    substring = "_"

    while size_difference > 0:

        if j == directories_sizes[i]:

            j = 0

            k += 1

            for l in range(0, k):

                substring += "_"

        shutil.copy(directories_paths[i] + directories_contents[i][j], directories_paths[i] + substring + directories_contents[i][j])

        print(directories_paths[i] + directories_contents[i][j]) # debug

        print(str(size_difference) + " left")
        
        j += 1
            
        size_difference -= 1