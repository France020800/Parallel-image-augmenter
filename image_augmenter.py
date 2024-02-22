import os
import cv2
import time
import albumentations as A

# Load images
folder_path = 'in_images/'
image_list = os.listdir(folder_path)
images = [cv2.imread(folder_path + image) for image in image_list]

brightnessTransform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=1, contrast_limit = 1, p=1.0),
])

cropTransform = A.Compose([
    A.RandomCrop(width=1536, height=1536, p=1.0),
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

transformed_images = []
start_time = time.time()
for image in images:
    transformed_images.append(brightnessTransform(image=image)['image'])
    transformed_images.append(cropTransform(image=image)['image'])
    transformed_images.append(rotateTransform(image=image)['image'])
    transformed_images.append(colorTransform(image=image)['image'])
    transformed_images.append(horizontalFlipTransform(image=image)['image'])
    transformed_images.append(verticalFlipTransform(image=image)['image'])
    transformed_images.append(complexTransform(image=image)['image'])

print('Saving images...')
index = 0
for image in transformed_images:
    out_path = f'out_images/augmented_demo_kitty_{index}.jpg'
    cv2.imwrite(out_path, image)
    index += 1
print('Done!')

end_time = time.time() 
print(f'Time taken to augment {len(images)} images: {end_time - start_time} seconds')
