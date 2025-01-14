import cv2
import os
import numpy as np
import argparse
from itertools import product
def get_args():
    parser = argparse.ArgumentParser("Jimmy Coder Photomosaic")
    parser.add_argument("--input","-i",type = str,default="inputvideo.mp4")
    parser.add_argument("--output","-o", type=str, default="output1.mp4")
    parser.add_argument("--image-file","-f", type=str, default="./messi")
    parser.add_argument("--stride","-s", type=int, default=10
                    )
    parser.add_argument("--fps", type=int, default=0, help="frame per second")
    parser.add_argument("--IoU", type=float, default=0.2, help="Intersection over Union")
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
    cap= cv2.VideoCapture(args.input)
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if args.fps == 0 else args.fps
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not os.path.exists(args.image_file) or len(os.listdir(args.image_file)) == 0:
        raise ValueError(f"No images found in the directory: {args.image_file}")
    images, avg_colors = get_avg_color(args.image_file, args.stride)
    while cap.isOpened():
        _,frame = cap.read()
        if not _:
            break
        output_image = np.zeros((height,width,3),np.uint8)
        for i,j in product(range(int(width/args.stride)),range(int(height/args.stride))):
            images_in_stride = frame[j * args.stride: (j+1) * args.stride,i * args.stride: (i+1) * args.stride,:]  #small cropped images in input image
            color_in_stride = np.sum(np.sum(images_in_stride,axis=0),axis = 0) / (args.stride**2)  #calculate the avg color in 1 small part of main image
            distance = euclid_distance(color_in_stride,avg_colors)
            idx = np.argmin(distance)
            output_image[j * args.stride: (j+1) * args.stride,i * args.stride: (i+1) * args.stride,:] = images[idx]
        if args.IoU:
            overlay = cv2.resize(frame,(int(width*args.IoU),int(height*args.IoU)))
            output_image[height-int(height*args.IoU):,width-int(width*args.IoU):,:] = overlay
        out.write(output_image)
    cap.release()
    out.release()


if __name__ == '__main__':
    args = get_args()
    convert(args)