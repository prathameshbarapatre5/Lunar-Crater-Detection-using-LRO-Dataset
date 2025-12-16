import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, unsharp_mask
from google.colab import drive
import time

drive.mount('/content/drive')

def load_tiff_image(file_path):
    image = io.imread(file_path)
    #Print image confirmation for debugging
    print(f"Image shape: {image.shape}")

    #Process multi-channel images (e.g. RGB or hyperspectral)
    if len(image.shape) == 3 and image.shape[0] > 1:
        if image.shape[0] == 6:
            image = np.mean(image[:3], axis=0)
        elif image.shape[0] > 3:
            image = np.mean(image, axis=0)
        else:
            image = rgb2gray(np.moveaxis(image, 0, -1))
    elif len(image.shape) > 2:
        image = rgb2gray(image)
    return image

def process_image(image, sigma=2.0, min_size=10, closing_disk_size=5):

    #Normalize values for consistency (0-1 range)
    if np.max(image) > 1:
        image = image / np.max(image)


    #Smooth the image to reduce noise
    smoothed_image = filters.gaussian(image, sigma=sigma)


    plt.figure(figsize=(6, 6))
    plt.imshow(smoothed_image, cmap='gray')
    plt.title("Smoothed Image")
    plt.axis('off')
    plt.show()


    #Sharpen edges to define crater rims
    sharp_image = unsharp_mask(smoothed_image, radius=1.0, amount=1.5)


    plt.figure(figsize=(6, 6))
    plt.imshow(sharp_image, cmap='gray')
    plt.title("Sharpened Image")
    plt.axis('off')
    plt.show()

    #Detect edges using Sobel filter
    edges = filters.sobel(sharp_image)


    plt.figure(figsize=(6, 6))
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")
    plt.axis('off')
    plt.show()


    #Binarize the image using Otsu's thresholding
    thresh = threshold_otsu(edges)
    binary_image = edges > thresh


    plt.figure(figsize=(6, 6))
    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    plt.show()


    #Clean up noise using morphological operations
    #Remove small white spots (noise)
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=min_size)
    #Close small holes inside objects
    cleaned_image = morphology.binary_closing(cleaned_image, morphology.disk(closing_disk_size))

    return cleaned_image

def display_intermediate_steps(smoothed_image, sharp_image, edges, binary_image):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(smoothed_image, cmap='gray')
    plt.title("Smoothed Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(sharp_image, cmap='gray')
    plt.title("Sharpened Image")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')

    plt.show()

def detect_craters(cleaned_image):
    #Identify connected components (craters) in the binary image
    labeled_image, num_labels = measure.label(cleaned_image, background=0, return_num=True)


    #Extract properties of detected regions
    regions = measure.regionprops(labeled_image)
    print(f"Number of craters detected: {num_labels}")

    return labeled_image, regions

def display_craters(original_image, labeled_image, regions, title):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(original_image, cmap='gray')


    for region in regions:
        #Get bounding box coordinates
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)


        #Plot center of the crater
        y0, x0 = region.centroid
        plt.plot(x0, y0, '.g', markersize=10)

    plt.title(title)
    plt.axis('off')

    plt.show()

def zoom_and_process(image, bbox, sigma=1.0, min_size=64, closing_disk_size=3):
    x_min, x_max, y_min, y_max = bbox
    #Crop to the region of interest
    cropped_image = image[y_min:y_max, x_min:x_max]


    #Process (Smooth -> Edge -> Binary -> Clean) and Detect
    cleaned_image = process_image(cropped_image, sigma, min_size, closing_disk_size)
    labeled_image, regions = detect_craters(cleaned_image)

    return cropped_image, labeled_image, regions

def main():
    file_path = '/content/drive/MyDrive/Colab Notebooks/Lunar_Clementine_NIR_cal_empcor_500m.tif'

    start_time = time.time()
    #Load full satellite image
    image = load_tiff_image(file_path)
    bbox = (1000, 1500, 2000, 2500)
    #Focus on a specific area to detect craters
    cropped_image, labeled_image, regions = zoom_and_process(image, bbox)
    display_craters(cropped_image, labeled_image, regions, f"Detected Craters: {len(regions)} found in Zoomed Area")
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")


main()
