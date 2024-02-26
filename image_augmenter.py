import os
import cv2
import time
import albumentations as A
from functions.functions import augment_images

# Load images
folder_path = 'in_images/'
image_list = os.listdir(folder_path)
images = [cv2.imread(folder_path + image) for image in image_list]

# Transform objects
transformation = A.Compose([
    A.Rotate(limit=360, p=1.0),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit = 0.7, p=0.5),
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6),
    A.RandomCrop(width=1536, height=1536, p=0.2)
])


print('Augmenting images...')
start_time = time.time()
transformed_images = augment_images(images, transformation)
end_time = time.time() 
print(f'Time taken to augment {len(images)} images: {round(end_time - start_time, 4)} seconds')

# print('Saving images...')
# index = 0
# for image in transformed_images:
#     out_path = f'out_images/augmented_demo_kitty_{index}.jpg'
#     cv2.imwrite(out_path, image)
#     index += 1
print('Done!')

