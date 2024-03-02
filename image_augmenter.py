import os
import cv2
import time
import albumentations as A
from functions.functions import augment_images, sequential_read

# Transform objects
flipAndColorJittering = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1.0),
])

rotateAndColorJittering = A.Compose([
    A.Rotate(limit=360, p=1.0),
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1.0),
])

flipAndBlur = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.AdvancedBlur(p=1.0),
])

rotateAndBlur = A.Compose([
    A.Rotate(limit=360, p=1.0),
    A.AdvancedBlur(p=1.0),
])

flipAndGray = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.ToGray(p=1.0),
])

rotateAndGray = A.Compose([
    A.Rotate(limit=360, p=1.0),
    A.ToGray(p=1.0),
])

flipAndChannelShuffle = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.ChannelShuffle(p=1.0),
])

rotateAndChannelShuffle = A.Compose([
    A.Rotate(limit=360, p=1.0),
    A.ChannelShuffle(p=1.0),
])

flipAndContrast = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.RandomBrightnessContrast(p=1.0),
])

rotateAndContrast = A.Compose([
    A.Rotate(limit=360, p=1.0),
    A.RandomBrightnessContrast(p=1.0),
])

flipAndPixelDropout = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.PixelDropout(dropout_prob=0.1, p=1.0),
])

rotateAndPixelDropout = A.Compose([
    A.Rotate(limit=360, p=1.0),
    A.PixelDropout(dropout_prob=0.1, p=1.0),
])

pixelDropOutAndColorJittering = A.Compose([
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1.0),
    A.PixelDropout(dropout_prob=0.1, p=1.0),
])
transformations = [flipAndColorJittering, rotateAndColorJittering, flipAndBlur, rotateAndBlur, flipAndGray, rotateAndGray, flipAndChannelShuffle, rotateAndChannelShuffle, flipAndContrast, rotateAndContrast, flipAndPixelDropout, rotateAndPixelDropout, pixelDropOutAndColorJittering]

start_time = time.time()
# print('Loading images...')
folder_path = 'in_images/'
images = sequential_read(folder_path)

# print('Augmenting images...')
transformed_images = augment_images(images, transformations)

# print('Saving images...')
    
image_names = next(os.walk('in_images'))[2]
for i, name in enumerate(image_names):
        for j, image in enumerate(transformed_images[i * 10: (i+1) * 10]):
            out_path = f'out_images/augmented_{j}_{name}'
            cv2.imwrite(out_path, image)
end_time = time.time() 
print(f'{round(end_time - start_time, 4)}')
# print(f'Time taken to augment {len(images)} images: {round(end_time - start_time, 4)} seconds')
# print('Done!')

