import os
import cv2
import time
import albumentations as A

def image_resized(image):
    height, width = image.shape[:2]
    if height <= 1536 or width <= 1536:
        resized_image = A.resize(image, height=1536, width=1536)
    resized_image = A.resize(image, height=int(height*0.4), width=int(width*0.4))
    return resized_image

# Load images
folder_path = 'in_images/'
image_list = os.listdir(folder_path)
images = [cv2.imread(folder_path + image) for image in image_list]

brightnessTransform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit = 0.7, p=1.0),
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

transformations = [brightnessTransform, cropTransform, rotateTransform, colorTransform, horizontalFlipTransform, verticalFlipTransform, complexTransform]
transformed_images = []
print('Augmenting images...')
start_time = time.time()
for image in images:
    for transformation in transformations:
        image_to_resize = transformation(image=image)['image']
        image_resize = image_resized(image_to_resize)
        transformed_images.append(image_resize)
        
end_time = time.time() 
print(f'Time taken to augment {len(images)} images: {end_time - start_time} seconds')

print('Saving images...')
index = 0
for image in transformed_images:
    out_path = f'out_images/augmented_demo_kitty_{index}.jpg'
    cv2.imwrite(out_path, image)
    index += 1
print('Done!')

