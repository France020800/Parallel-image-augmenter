import os
import sys
import cv2
import time
import albumentations as A
import multiprocessing
from multiprocessing import Pool

def augment_images(images, transformations):
    thread_name = multiprocessing.current_process().name
    print(f'Start thread {thread_name} with {len(images)} images')
    transformed_images = []
    for image in images:
        transformed_images.append(transformations[0](image=image)['image'])
        transformed_images.append(transformations[1](image=image)['image'])
        transformed_images.append(transformations[2](image=image)['image'])
        transformed_images.append(transformations[3](image=image)['image'])
        transformed_images.append(transformations[4](image=image)['image'])
        transformed_images.append(transformations[5](image=image)['image'])
        transformed_images.append(transformations[6](image=image)['image'])
    index = 0
    for image in transformed_images:
        out_path = f'out_images/augmented_demo_kitty_{thread_name}_{index}.jpg'
        cv2.imwrite(out_path, image)
        index += 1
    print(f'End thread {multiprocessing.current_process().name}')

if __name__ == '__main__':

    # Transform objects
    brightnessTransform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=1, contrast_limit = 1, p=1.0),
    ])

    cropTransform = A.Compose([
        A.RandomCrop(width=128, height=128, p=1.0),
    ])

    rotateTransform = A.Compose([
        A.Rotate(limit=45, p=1.0),
    ])

    colorTransform = A.Compose([
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6),
    ])

    horizontalFlipTransform = A.Compose([
        A.HorizontalFlip(p=1.0),
    ])

    verticalFlipTransform = A.Compose([
        A.VerticalFlip(p=1.0),
    ])

    complexTransform = A.Compose([
        A.Rotate(limit=180, p=1.0),
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
    print('Using {} processes'.format(pool_size))

    # Split images into batches
    batch_size = round(len(images) // pool_size)
    image_batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
    args = [(batch, [brightnessTransform, cropTransform, rotateTransform, colorTransform, horizontalFlipTransform, verticalFlipTransform, complexTransform]) for batch in image_batches]

    # Augment images
    start_time = time.time()
    pool.starmap(augment_images, args)
    pool.close()
    pool.join()
    end_time = time.time()
    print(f'Time taken to augment {len(images)} images: {end_time - start_time} seconds')

    # Flatten the list of lists
    # transformed_images = [image for sublist in results for image in sublist]


    # print('Saving images...')
    # index = 0
    # for image in transformed_images:
    #     out_path = f'out_images/augmented_demo_kitty_{index}.jpg'
    #     cv2.imwrite(out_path, image)
    #     index += 1
    # print('Done!')
