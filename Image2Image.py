import cv2
import os
import numpy as np
import argparse
from itertools import product
def get_args():
    parser = argparse.ArgumentParser("Jimmy Coder Photomosaic")
    parser.add_argument("--input-image","-i",type = str,default="input.jpg")
    parser.add_argument("--output-image","-o", type=str, default="output.jpg")
    parser.add_argument("--image-file","-f", type=str, default="./messi")
    parser.add_argument("--stride","-s", type=int, default=20)
    args = parser.parse_args()
    return args
def get_avg_color(path,size):
    images = []
    avg_colors = []
    for img_path in os.listdir(path):
        image = cv2.imread(os.path.join(path,img_path),cv2.IMREAD_COLOR)
        image = cv2.resize(image,(size,size))
        images.append(image)
        avg_color = np.sum(np.sum(image,axis=0),axis = 0) / (size**2)
        avg_colors.append(avg_color)
    return images, np.array(avg_colors)
def euclid_distance(color_in_stride,avg_colors):
    distances = []
    for color in avg_colors:
        distance = sum((color_in_stride[i]-color[i])**2 for i in range(len(color_in_stride)))**0.5
        distances.append(distance)
    return distances
def convert(args):
    input_image = cv2.imread(args.input_image,cv2.IMREAD_COLOR)
    height,width,channel = input_image.shape
    output_image = np.zeros((height,width,3),np.uint8)
    images, avg_colors = get_avg_color(args.image_file,args.stride)
    for i,j in product(range(int(width/args.stride)),range(int(height/args.stride))):
        images_in_stride = input_image[j * args.stride: (j+1) * args.stride,i * args.stride: (i+1) * args.stride,:]  #small cropped images in input image
        color_in_stride = np.sum(np.sum(images_in_stride,axis=0),axis = 0) / (args.stride**2)  #calculate the avg color in 1 small part of main image
        distance = euclid_distance(color_in_stride,avg_colors)
        idx = np.argmin(distance)
        output_image[j * args.stride: (j+1) * args.stride,i * args.stride: (i+1) * args.stride,:] = images[idx]
    cv2.imwrite(args.output_image,output_image)

if __name__ == '__main__':
    args = get_args()
    convert(args)