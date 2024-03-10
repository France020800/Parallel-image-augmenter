import albumentations as A
import os
import cv2

def sequential_augment_images(images, transformations):
    cv2.setNumThreads(0)
    transformed_images = []
    for i, image in enumerate(images):
        for j, transformation in enumerate(transformations):
            transformed_image = transformation(image=image)['image']
            transformed_images.append(transformed_image)
            
    for transformed_image in transformed_images:
        out_path = f'out_images/augmented_{j}_demo_kitty_{i}.jpg'
        cv2.imwrite(out_path, transformed_image)

def _augment_images(images, transformations):
    transformed_images = []
    for i, image in enumerate(images):
        for j, transformation in enumerate(transformations):
            transformed_image = transformation(image=image)['image']
            transformed_images.append(transformed_image)

    return transformed_images

def parallel_augment_images(image_to_read, transformations):
    cv2.setNumThreads(0)
    images = _parallel_read(image_to_read)
    transformed_images = _augment_images(images, transformations)
    size = len(transformations)
    for i, name in enumerate(image_to_read):
        for j, image in enumerate(transformed_images[i * size: (i+1) * size]):
            out_path = f'out_images/augmented_{j}_{name}'
            cv2.imwrite(out_path, image)
    return None

def parallel_augment_images_test(args):
    cv2.setNumThreads(0)
    image_to_read = args[0]
    transformations = args[1]
    images = _parallel_read(image_to_read)
    transformed_images = _augment_images(images, transformations)
    size = len(transformations)
    for i, name in enumerate(image_to_read):
        for j, image in enumerate(transformed_images[i * size: (i+1) * size]):
            out_path = f'out_images/augmented_{j}_{name}'
            cv2.imwrite(out_path, image)
    return None

def sequential_read(folder_path):
    cv2.setNumThreads(0)
    image_list = os.listdir(folder_path)
    return [cv2.imread(folder_path + image) for image in image_list]    

def _parallel_read(image_to_read):
    images = [cv2.imread(f'in_images/{image}') for image in image_to_read]
    return images