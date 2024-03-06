import albumentations as A
import os
import cv2

def sequential_augment_images(images, transformations):
    cv2.setNumThreads(0)
    for i, image in enumerate(images):
        for j, transformation in enumerate(transformations):
            transformed_image = transformation(image=image)['image']
            transformed_image = transformation(image=transformed_image)['image']
            transformed_image = transformation(image=transformed_image)['image']
            # out_path = f'out_images/augmented_{j}_demo_kitty_{i}.jpg'
            # cv2.imwrite(out_path, transformed_image)

def _augment_images(images, transformations):
    transformed_images = []
    for i, image in enumerate(images):
        for j, transformation in enumerate(transformations):
            transformed_image = transformation(image=image)['image']
            transformed_image = transformation(image=transformed_image)['image']
            transformed_image = transformation(image=transformed_image)['image']
            # transformed_images.append(transformed_image)

    return transformed_images

def parallel_augment_images(image_to_read, transformations):
    cv2.setNumThreads(0)
    images = _parallel_read(image_to_read)
    transformed_images = _augment_images(images, transformations)
    size = len(transformations)
    # for i, name in enumerate(image_to_read):
    #     for j, image in enumerate(transformed_images[i * size: (i+1) * size]):
    #         out_path = f'out_images/augmented_{j}_{name}'
    #         cv2.imwrite(out_path, image)
    return None

def sequential_read(folder_path):
    cv2.setNumThreads(0)
    image_list = os.listdir(folder_path)
    return [cv2.imread(folder_path + image) for image in image_list]    

def _parallel_read(image_to_read):
    images = [cv2.imread(f'in_images/{image}') for image in image_to_read]
    return images


def _image_resized(image):
    height, width = image.shape[:2]
    if height > 1536 or width > 1536:
        resized_image = A.resize(image, height=int(height*0.6), width=int(width*0.6))
    else:
        resized_image = image
    return resized_image
