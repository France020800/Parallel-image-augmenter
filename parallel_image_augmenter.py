import os
import sys
import cv2
import time
import multiprocessing
import albumentations as A
from functions.functions import augment_images
from joblib import Parallel, delayed

if __name__ == '__main__':

    # Transform objects
    transformation = A.Compose([
        A.Rotate(limit=360, p=1.0),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit = 0.7, p=0.5),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6),
        A.RandomCrop(width=1536, height=1536, p=0.2)
    ])

    # Load images
    folder_path = 'in_images/'
    image_list = os.listdir(folder_path)
    images = [cv2.imread(folder_path + image) for image in image_list]

    # Create a pool of workers 
    if len(sys.argv) > 1:
        num_thread = int(sys.argv[1])
    else:
        num_thread = multiprocessing.cpu_count()
    print('Using {} processes'.format(num_thread))
    
    
    # Split images into batches
    batch_size = round(len(images) // num_thread)
    image_batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]

    # Split images into batches
    start_time = time.time()
    transformed_images = Parallel(n_jobs=num_thread)(delayed(augment_images)(batch, transformation) for batch in image_batches)
    end_time = time.time()
    print(f'{round(end_time - start_time, 4)}')
    # print(f'Time taken to augment {len(images)} images: {end_time - start_time} seconds')

    # Flatten the list of lists
    # transformed_images = [image for sublist in results for image in sublist]

    # print(f'Length of transformed images: {len(transformed_images[0])}')
    # print('Saving images...')
    # index = 0
    # for image in transformed_images:
    #     out_path = f'out_images/augmented_demo_kitty_{index}.jpg'
    #     cv2.imwrite(out_path, image)
    #     index += 1
    print('Done!')
