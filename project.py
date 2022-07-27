import numpy as np
import cv2 as cv
import math
import random


def find_candidate_pixels(row, col, filter, iteration, o_height, o_width):
    #o_height outputun satır sayısı, o_width sütun sayısı
    border = math.floor(filter/2)
    candidate_locations = []
    candidate_pixels = {}
    count = 1

    search_height = range(0,border)
    search_width = range(0,filter-1)

    # In the repeat use the full neighborhood to look at candidates
    # todo bunun üzerine biraz daha düşüneceğim iterasyonun 1 den büyük olması durumuna göre değişmesi
    if iteration > 1:
        search_height = range(-border,border)
    for heights in range(search_height): # c_h  = heights
        for widths in range (search_width): # c_w = weights
            komsuluk = widths-border # c_w_adj = komşuluk
            if ((heights > 0 or (heights == 0 and komsuluk < 0)) or iteration > 1) and row-heights <= o_height+border:

                new_height = used_heights(row-heights,widths+komsuluk)+heights
                new_width = used_widths(row-heights,widths+komsuluk)-komsuluk

            # If we reach the edge of the image, choose a new pixel
            while or(or(new_height < neighborhood, new_height > i_height-neighborhood),
            or(new_width < neighborhood, new_width > i_width-neighborhood))
            new_height = round(random.random()*(i_height-1)+1)
            new_width = round(random.random()*(i_width-1)+1)


        candidate_locations = [candidate_locations; new_height,new_width];
        candidate_pixels{count} = input_image(new_height, new_width, :);
        count = count + 1;


def add_random_pixel_with_p():
    pass

def remove_duplicates():
    pass

def find_best_pixel():
    pass

def add_new_pixel():
    pass 

def find_distance():
    pass

input_image = cv.imread('Starry_Night.jpg')
target_image = cv.imread('Johnny_Depp.jpg')

# Parameters (adjust for your use)
filter_size = 5 # Considers a nxn neighborhood (e.g., 5x5)
p = 0.2 # Probability of considering a random pixel
m = 1 # Weighting on intesity matching between images
iterations = 3 # Number of times the algorithm runs over the image
pad = filter_size // 2 # using instead of n_2          <<<<<<<<<<<<<<<<<<<----------------------

# Step 1: Initialize output image
[i_height, i_width, i_depth] = input_image.shape
[o_height, o_width, o_depth] = target_image.shape
border = math.floor(filter_size/2)


#grayscaled versions will be used in intensity distance calculation
input_image_gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
target_image_gray = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)


# Create variables for output image and locations used in input image
# bir öncekilerin dtype ına bakılabilir, buradaki ona göre ayarlanabilir.
output_image = np.zeros((o_height+border, o_width+border*2, o_depth),dtype=np.int8)
used_heights = np.zeros((o_height+border, o_width+border*2),dtype=np.int8)
used_widths = np.zeros((o_height+border, o_width+border*2),dtype=np.int8)


# Randomly assign "used" pixel locations to borders
# used_heights(1:border,:) = round(rand(border, o_width+border*2)*(i_height-1)+1)
# used_widths(1:border,:) = round(rand(border, o_width+border*2)*(i_width-1)+1)
# used_heights(:,1:border) = round(rand(o_height+border, border)*(i_height-1)+1)
# used_widths(:,1:border) = round(rand(o_height+border, border)*(i_width-1)+1)
# used_heights(:,border+o_width+1:border*2+o_width) = round(rand(o_height+border, border)*(i_height-1)+1)
# used_widths(:,border+o_width+1:border*2+o_width) = round(rand(o_height+border, border)*(i_width-1)+1)


# # Fill output image with appropriate color values
# # first border
# for row in range(1,border):
#     for col in range(1,o_width+border*2):
#         output_image[row,col,:] = input_image[used_heights(row,col),used_widths(row,col),:]
#
# # then rest
# for row in range(border,o_height+border ):
#     for col in range(1,o_width+border*2):
#         output_image[row,col,:] = input_image[used_heights(row,col),used_widths(row,col),:]


# burada matlab çıktısına bakacağım. for döngüleri ile nasıl dolmuş output resmi
# todo matlab dan baktım sadece border ları dolduruyor şimdilik gerek görmedim. Lazım olursa tekrar ekle.



for iteration in range(1,iterations): #this loop encapsulates whole output image

    for row in range(border+1,o_height+border): # these loops iterates for each pixel in the output image
        for col in range(border+1,o_width+border): # sadece valid ğixeller için çalışır

            find_candidate_pixels()
            add_random_pixel_with_p()
            remove_duplicates()
            find_best_pixel()
            add_new_pixel()


def find_distance(h):
    '''
    height = max(-n_2, 1-(h-n_2)):1:min(n_2,o_height-(h-n_2));
    width = max(-n_2, 1-(w-n_2)):1:min(n_2,o_width-(w-n_2));
    input_values = input_image_gray(height+c_h, width+c_w);
    target_values = target_image_gray((h-n_2)+height, (w-n_2)+width);
    input_target_distance = (mean(input_values(:))-mean(target_values(:)))^2;

    distance = m*input_target_distance + (1/n^2)*input_result_distance;
    '''
    if iteration>1:
        height = np.arange(-pad,min(pad,o_height-(h-pad),1)).tolist()
        width = np.arange(-pad,pad,1).tolist()
        n = (max(height)+pad+1)*(max(width)+pad+1)
    else:
        pass

    return            