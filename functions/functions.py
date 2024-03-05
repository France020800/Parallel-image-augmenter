import albumentations as A
import os
import cv2
import time

def augment_images(images, transformations):
    transformed_images = []
    # length = len(images)*len(transformations)
    # print(f'Start thread {threading.current_thread().ident}')
    for image in images:
        for transformation in transformations:
            # time.sleep(0.5)
            image_to_resize = transformation(image=image)['image']
            image_to_resize = transformation(image=image_to_resize)['image']
            image_to_resize = transformation(image=image_to_resize)['image']
            resized_image = _image_resized(image_to_resize)
            transformed_images.append(resized_image)
        # progress = round(len(transformed_images)/length, 2)
        # print(f'Progress: {progress*100}%')

    return transformed_images

def augment_image(image, transformations):
    transformed_images = []
    # print(f'Start thread {threading.current_thread().ident}')
    for transformation in transformations:
        image_to_resize = transformation(image=image)['image']
        resized_image = _image_resized(image_to_resize)
        transformed_images.append(resized_image)

    return transformed_images

def parallel_augment_image(image_to_read, transformations):
    images = _parallel_read(image_to_read)
    transformed_images = augment_images(images, transformations)
    for i, name in enumerate(image_to_read):
        for j, image in enumerate(transformed_images[i * 10: (i+1) * 10]):
            out_path = f'out_images/augmented_{j}_{name}'
            cv2.imwrite(out_path, image)
    return None

def sequential_read(folder_path):
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
