import random as rd
import numpy as np
import cv2

def preprocess(file, degree = 2):
    '''
    This function turns a image with any size and any number of channels to a grayscale image with size 224*224.
    '''
    # read the image, turn it to grayscale
    image = cv2.imread(file)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # apply CLAHE to the grayscale image
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    image = clahe.apply(image)
    # turn the dtype to float
    image = image.astype(np.float32)

    # adjust size to 224 using Gaussian Pyramid, Laplacian Pyramid and Lanczos resampling
    if np.mean(image.shape) < 64:
        image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_LANCZOS4)
        image = cv2.pyrUp(image)
        image = cv2.pyrUp(image)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    elif np.mean(image.shape) == 64:
        image = cv2.pyrUp(image)
        image = cv2.pyrUp(image)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    elif np.mean(image.shape) <= 112:
        image = cv2.resize(image, (112, 112), interpolation = cv2.INTER_LANCZOS4)
        image = cv2.pyrUp(image)
    elif np.mean(image.shape) < 224:
        image = cv2.pyrUp(image)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    elif np.mean(image.shape) == 224:
        image = image
    elif np.mean(image.shape) < 448:
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    elif np.mean(image.shape) < 896:
        image = cv2.pyrDown(image)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    else:
        image = cv2.resize(image, (896, 896), interpolation = cv2.INTER_LANCZOS4)
        image = cv2.pyrDown(image)
        image = cv2.pyrDown(image)
    
    # apply Laplacian sharpening
    if degree >= 2:
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        image = cv2.add(image, laplacian)
    
    # normalize the array and change dtype back to uint8
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return image

def fileprocess(image, degree = 2):
    '''
    This function is similar to the former one.
    This function is used not in the pipeline but in the app.
    '''
    # read the image, turn it to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # apply CLAHE to the grayscale image
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    image = clahe.apply(image)
    # turn the dtype to float
    image = image.astype(np.float32)

    # adjust size to 224 using Gaussian Pyramid, Laplacian Pyramid and Lanczos resampling
    if np.mean(image.shape) < 64:
        image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_LANCZOS4)
        image = cv2.pyrUp(image)
        image = cv2.pyrUp(image)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    elif np.mean(image.shape) == 64:
        image = cv2.pyrUp(image)
        image = cv2.pyrUp(image)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    elif np.mean(image.shape) <= 112:
        image = cv2.resize(image, (112, 112), interpolation = cv2.INTER_LANCZOS4)
        image = cv2.pyrUp(image)
    elif np.mean(image.shape) < 224:
        image = cv2.pyrUp(image)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    elif np.mean(image.shape) == 224:
        image = image
    elif np.mean(image.shape) < 448:
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    elif np.mean(image.shape) < 896:
        image = cv2.pyrDown(image)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    else:
        image = cv2.resize(image, (896, 896), interpolation = cv2.INTER_LANCZOS4)
        image = cv2.pyrDown(image)
        image = cv2.pyrDown(image)
    
    # apply Laplacian sharpening
    if degree >= 2:
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        image = cv2.add(image, laplacian)
    
    # normalize the array and change dtype back to uint8
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return image

def augment(image, train = 0):
    '''
    This function will apply random augmentation on the image array.
    '''
    if rd.random() < 0.5: # rotate
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    if rd.random() < 0.5: # horizontal flip
        image = cv2.flip(image, 1)

    if rd.random() < 0.5: # vertical flip
        image = cv2.flip(image, 0)
    
    if train > 0:
        if rd.random() < 0.5: # brightness
            brightness_val = rd.uniform(0.8, 1.5)
            image = cv2.multiply(image, np.array([brightness_val]))
        
        if rd.random() < 0.5: # contrast
            contrast_val = rd.uniform(0.8, 1.5)
            image = cv2.multiply(image, np.array([contrast_val]))

        if (train % 2) == 1: # random noise
            magnitude = 20
            noise = np.random.randint(- magnitude, magnitude + 1, size = image.shape, dtype = np.int16)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image