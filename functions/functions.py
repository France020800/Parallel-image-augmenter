import albumentations as A
import threading
import time

def augment_images(images, transformations):
    transformed_images = []
    print(f'Start thread {threading.current_thread().ident}')
    for image in images:
        for transformation in transformations:
            # time.sleep(1)
            image_to_resize = transformation(image=image)['image']
            resized_image = _image_resized(image_to_resize)
            transformed_images.append(resized_image)

    return transformed_images

def augment_image(image, transformations):
    transformed_images = []
    print(f'Start thread {threading.current_thread().ident}')
    for transformation in transformations:
        image_to_resize = transformation(image=image)['image']
        resized_image = _image_resized(image_to_resize)
        transformed_images.append(resized_image)

    return transformed_images


def _image_resized(image):
    height, width = image.shape[:2]
    if height > 1536 or width > 1536:
        resized_image = A.resize(image, height=int(height*0.4), width=int(width*0.4))
    else:
        resized_image = image
    return resized_image