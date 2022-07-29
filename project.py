import numpy as np
import cv2 as cv
import math
import random

filter_size = 5 # Considers a nxn neighborhood (e.g., 5x5)
p = 0.2 # Probability of considering a random pixel
m = 1 # Weighting on intesity matching between images
iterations = 3 # Number of times the algorithm runs over the image
oran = 5 # binde -*
border = math.floor(filter_size/2)




input_image = cv.imread('Starry_Night.jpg')
target_image = cv.imread('Johnny_Depp.jpg')

scale_percent = 30 # percent of original size
width = int(target_image.shape[1] * scale_percent / 100)
height = int(target_image.shape[0] * scale_percent / 100)
dim = (width, height)
target_image = cv.resize(target_image, dim, interpolation = cv.INTER_AREA)
 

[i_height, i_width, i_depth] = input_image.shape
[o_height, o_width, o_depth] = target_image.shape

input_image = cv.copyMakeBorder(input_image,border,border,border,border,cv.BORDER_REFLECT)
target_image = cv.copyMakeBorder(target_image,border,border,border,border,cv.BORDER_REFLECT)

#grayscaled versions will be used in intensity distance calculation
input_image_gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
target_image_gray = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)

# Create variables for output image and locations used in input image
# bir öncekilerin dtype ına bakılabilir, buradaki ona göre ayarlanabilir.
output_image = np.zeros((o_height, o_width, o_depth),dtype=np.int8)



def pairwaise(piksel1, piksel2):
    return math.sqrt((piksel1[0] - piksel2[0])^2 + (piksel1[1] - piksel2[1])^2 + (piksel1[2] - piksel2[2])^2)

def grayPair(gray1,gray2):
    return abs(gray1-gray2)

# I assume grayscale version f both input and target are global

def pdist(input_row, input_col, target_row, target_col):
    totaldist = 0
    for i in range(-border,+border):
        for j in range(-border,+border):
         #  print(i,j,target_col,target_row,input_col,input_row) 
           totaldist =totaldist + grayPair(input_image_gray[input_row + i][input_col +j],target_image_gray[target_row + i][target_col +j])
           
    return totaldist

def find_candidate_pixels(current_row, current_col):

    candidate_locations = []
    candidate_pixels = []
    best_distance = 10000
    best_pixel = input_image[0][0]
    (row, col, dim) = [i_height, i_width, i_depth]
    total_num_pixel = row*col
    total_num_cand = math.floor((total_num_pixel*oran) // 2000)
    #toplam candidate sayımızı resmin boyutu ve eşik oranına göre belirliyoruz.

    for rastgele in range(total_num_cand):
        random_row = random.randint(0, row)
        random_col = random.randint(0, col)
        key = (random_row,random_col)
        if key in candidate_locations:
            pass
        else:
            candidate_locations.append(key)
            candidate_pixels.append(input_image[random_row, random_col, :])
            distance = pdist(random_row,random_col,current_row,current_col)
            if distance < best_distance:
                best_pixel = input_image[random_row][random_col]
    return best_pixel

    #find_best_pixel(candidate_pixels,current_row, current_col)


#find candidate içinde çağırılır en kısa mesafedeki pikseli seçer.
def find_best_pixel(candidate_pixels,current_row, current_col):
    pass

def find_distance():
    pass

def add_new_pixel(output,row,col,pixel):
    pass


#for iteration in range(1,iterations): #this loop encapsulates whole output image

for row in range(border,o_height+border): # these loops iterates for each pixel in the output image
    print(row)

    for col in range(border,o_width+border): # sadece valid pixeller için çalışır
        best_pixel = find_candidate_pixels(row,col)
        output_image[row-border, col-border, : ] = best_pixel
        # add_new_pixel(output_image, row, col, best_pixel)

cv.imshow('output',output_image)
cv.waitKey(0)