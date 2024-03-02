import os
import sys
import cv2
import time
import multiprocessing
import albumentations as A
from functions.functions import parallel_augment_image
from joblib import Parallel, delayed

if __name__ == '__main__':

    # TODO - Design correct but is required more work for the processes.
    # TODO - Rename the variables: images, image_batches. They are't images but paths.

    flipAndColorJittering = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1.0),
    ])

    rotateAndColorJittering = A.Compose([
        A.Rotate(limit=360, p=1.0),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1.0),
    ])

    flipAndBlur = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.AdvancedBlur(p=1.0),
    ])

    rotateAndBlur = A.Compose([
        A.Rotate(limit=360, p=1.0),
        A.AdvancedBlur(p=1.0),
    ])

    flipAndGray = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.ToGray(p=1.0),
    ])

    rotateAndGray = A.Compose([
        A.Rotate(limit=360, p=1.0),
        A.ToGray(p=1.0),
    ])

    flipAndChannelShuffle = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.ChannelShuffle(p=1.0),
    ])

    rotateAndChannelShuffle = A.Compose([
        A.Rotate(limit=360, p=1.0),
        A.ChannelShuffle(p=1.0),
    ])

    flipAndContrast = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
    ])

    rotateAndContrast = A.Compose([
        A.Rotate(limit=360, p=1.0),
        A.RandomBrightnessContrast(p=1.0),
    ])

    flipAndPixelDropout = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.PixelDropout(dropout_prob=0.1, p=1.0),
    ])

    rotateAndPixelDropout = A.Compose([
        A.Rotate(limit=360, p=1.0),
        A.PixelDropout(dropout_prob=0.1, p=1.0),
    ])

    pixelDropOutAndColorJittering = A.Compose([
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1.0),
        A.PixelDropout(dropout_prob=0.1, p=1.0),
    ])

    transformations= [flipAndColorJittering, rotateAndColorJittering, flipAndBlur, rotateAndBlur, flipAndGray, rotateAndGray, flipAndChannelShuffle, rotateAndChannelShuffle, flipAndContrast, rotateAndContrast, flipAndPixelDropout, rotateAndPixelDropout, pixelDropOutAndColorJittering]

    # Get the images number
    images = next(os.walk('in_images'))[2]
    num_images = len(images)

    # Create a pool of workers 
    if len(sys.argv) > 1:
        num_thread = int(sys.argv[1])
    else:
        num_thread = multiprocessing.cpu_count()
    # print('Using {} processes'.format(num_thread))
    
    
    # Split images into batches
    batch_size = round(num_images // num_thread)
    image_batches = [images[i:i+batch_size] for i in range(0, num_images, batch_size)]

    range(0, num_images, batch_size)
    start_time = time.time()
    Parallel(n_jobs=num_thread)(delayed(parallel_augment_image) (image, transformations) for image in image_batches)
    end_time = time.time()
    print(f'{round(end_time - start_time, 4)}')
    # print('Done!')
