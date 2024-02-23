import os
import sys
import cv2
import time
import albumentations as A
import multiprocessing
from functions.functions import augment_images
from multiprocessing import Pool

if __name__ == '__main__':

    # Transform objects
    brightnessTransform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit = 0.7, p=1.0),
    ])

    cropTransform = A.Compose([
        A.RandomCrop(width=1536, height=1536, p=1.0),
    ])

    rotateTransform = A.Compose([
        A.Rotate(limit=45, p=1.0),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6),
    ])

    colorTransform = A.Compose([
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6),
    ])

    horizontalFlipTransform = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6),
    ])

    verticalFlipTransform = A.Compose([
        A.VerticalFlip(p=1.0),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6),
    ])

    # Load images
    folder_path = 'in_images/'
    image_list = os.listdir(folder_path)
    images = [cv2.imread(folder_path + image) for image in image_list]

    # Create a pool of workers 
    if len(sys.argv) > 1:
        pool_size = int(sys.argv[1])
    else:
        pool_size = multiprocessing.cpu_count()
    pool = Pool(processes=pool_size)
    # print('Using {} processes'.format(pool_size))

    # Split images into batches
    batch_size = round(len(images) // pool_size)
    image_batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
    args = [(batch, [brightnessTransform, cropTransform, rotateTransform, colorTransform, horizontalFlipTransform, verticalFlipTransform]) for batch in image_batches]

    # Augment images
    # print('Augmenting images...')
    start_time = time.time()
    results = pool.starmap(augment_images, args)
    pool.close()
    pool.join()
    end_time = time.time()
    # print(f'Time taken to augment {len(images)} images: {end_time - start_time} seconds')

    # Flatten the list of lists
    transformed_images = [image for sublist in results for image in sublist]


    # print('Saving images...')
    index = 0
    for image in transformed_images:
        out_path = f'out_images/augmented_demo_kitty_{index}.jpg'
        cv2.imwrite(out_path, image)
        index += 1
    print('Done!')
    print(f'{round(end_time - start_time, 4)}')
