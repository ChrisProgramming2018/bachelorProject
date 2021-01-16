from PIL import Image
from pymatreader import read_mat                
import cv2
import numpy as np
import sys
import argparse
EPSILON = sys.float_info.epsilon

def convert_to_rgb(minval, maxval, val, colors):
    # "colors" is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.
    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the "colors" pallette.
    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    # Does it fall exactly on one of the color points?
    if f < EPSILON:
        return colors[i]
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))



def main(args):
    image_name = "heat_map{}.png".format(args.name)
    image = Image.open(args.path)
    img_d = np.asarray(image)
    x =  img_d.shape[0]
    y = img_d.shape[1]
    min_v = np.min(img_d)
    max_v = np.max(img_d)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    heat_image = np.empty((x, y, 3))
    for x in range(img_d.shape[0]):
        for y in range(img_d.shape[1]):
            val = img_d[x,y]
            heat_image[x,y] = convert_to_rgb(min_v, max_v, val, colors)
    img = cv2.imwrite(image_name, np.array(heat_image))
    if img:
        print("Image created ", image_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="param.json", type=str)
    parser.add_argument('--name', default="1", type=str)
    arg = parser.parse_args()
    main(arg)
