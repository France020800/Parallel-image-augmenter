import albumentations as A
import os
import cv2

def augment_images(images, transformations):
    transformed_images = []
    cv2.setNumThreads(0)
    for image in images:
        for transformation in transformations:
            image_to_resize = transformation(image=image)['image']
            resized_image = _image_resized(image_to_resize)
            transformed_images.append(resized_image)

    return transformed_images

def parallel_augment_image(image_to_read, transformations):
    images = _parallel_read(image_to_read)
    transformed_images = augment_images(images, transformations)
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
    cv2.setNumThreads(0)
    images = [cv2.imread(f'in_images/{image}') for image in image_to_read]
    return images


def _image_resized(image):
    height, width = image.shape[:2]
    if height > 1536 or width > 1536:
        resized_image = A.resize(image, height=int(height*0.6), width=int(width*0.6))
    else:
        resized_image = image
    return resized_image
