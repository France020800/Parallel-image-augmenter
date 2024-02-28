import os
import cv2
import time
import albumentations as A
from functions.functions import augment_images, sequential_read

# Transform objects
transformation = A.Compose([
    A.Rotate(limit=360, p=1.0),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit = 0.7, p=0.5),
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.5),
    A.RandomCrop(width=1536, height=1536, p=0.2)
])

GB_shift = A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)
saturation_transformation = A.HueSaturationValue(p=1.0)
channel_shuffle = A.ChannelShuffle(p=1.0)
random_gamma = A.RandomGamma(p=1.0)
random_brightness = A.RandomBrightnessContrast(p=1.0)
blur = A.Blur(p=1.0)
gray_transformation = A.ToGray(p=1.0)
random_rotate = A.Rotate(limit=360, p=1.0)
vertical_flip = A.VerticalFlip(p=1.0)
horizontal_flip = A.HorizontalFlip(p=1.0)
transformations = [GB_shift, saturation_transformation, channel_shuffle, random_gamma, random_brightness, blur, gray_transformation, random_rotate, vertical_flip, horizontal_flip]

start_time = time.time()
print('Loading images...')
folder_path = 'in_images/'
images = sequential_read(folder_path)

print('Augmenting images...')
transformed_images = augment_images(images, transformations)

# print('Saving images...')
index = 0
for image in transformed_images:
    out_path = f'out_images/augmented_demo_kitty_{index}.jpg'
    cv2.imwrite(out_path, image)
    index += 1
    
end_time = time.time() 
print(f'Time taken to augment {len(images)} images: {round(end_time - start_time, 4)} seconds')
print('Done!')

