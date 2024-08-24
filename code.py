import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, unsharp_mask
from google.colab import drive
import time

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Load the TIFF image from Google Drive
def load_tiff_image(file_path):
    image = io.imread(file_path)
    print(f"Image shape: {image.shape}")  # Print the shape of the image for debugging

    # Handle cases where the image has multiple channels
    if len(image.shape) == 3 and image.shape[0] > 1:
        if image.shape[0] == 6:
            image = np.mean(image[:3], axis=0)  # Average first three channels
        elif image.shape[0] > 3:
            image = np.mean(image, axis=0)
        else:
            image = rgb2gray(np.moveaxis(image, 0, -1))
    elif len(image.shape) > 2:
        image = rgb2gray(image)
    return image

# Step 3: Process the Image to Highlight Craters
def process_image(image, sigma=2.0, min_size=10, closing_disk_size=5):
    # Normalize image values if needed
    if np.max(image) > 1:
        image = image / np.max(image)

    # Apply a Gaussian filter to smooth the image
    smoothed_image = filters.gaussian(image, sigma=sigma)

    # Display the smoothed image
    plt.figure(figsize=(6, 6))
    plt.imshow(smoothed_image, cmap='gray')
    plt.title("Smoothed Image")
    plt.axis('off')
    plt.show()

    # Enhance sharpness using an unsharp mask filter
    sharp_image = unsharp_mask(smoothed_image, radius=1.0, amount=1.5)

    # Display the sharpened image
    plt.figure(figsize=(6, 6))
    plt.imshow(sharp_image, cmap='gray')
    plt.title("Sharpened Image")
    plt.axis('off')
    plt.show()

    # Use the Sobel filter to detect edges
    edges = filters.sobel(sharp_image)

    # Display the edges
    plt.figure(figsize=(6, 6))
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")
    plt.axis('off')
    plt.show()

    # Threshold the image to separate craters using Otsu's method
    thresh = threshold_otsu(edges)
    binary_image = edges > thresh

    # Display the binary image
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    plt.show()

    # Perform morphological operations to enhance the features
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=min_size)
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

# Step 4: Detect and Label Craters
def detect_craters(cleaned_image):
    # Label the detected features
    labeled_image, num_labels = measure.label(cleaned_image, background=0, return_num=True)

    # Get the properties of the labeled regions (craters)
    regions = measure.regionprops(labeled_image)
    print(f"Number of craters detected: {num_labels}")

    return labeled_image, regions

# Step 5: Display the Original Image and Detected Craters
def display_craters(original_image, labeled_image, regions, title):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(original_image, cmap='gray')

    # Draw rectangles around detected regions
    for region in regions:
        # Draw a rectangle around each region
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

        # Draw the centroid
        y0, x0 = region.centroid
        plt.plot(x0, y0, '.g', markersize=10)

    plt.title(title)
    plt.axis('off')

    plt.show()

# Step 6: Zoom into a Specific Region and Detect Craters
def zoom_and_process(image, bbox, sigma=1.0, min_size=64, closing_disk_size=3):
    x_min, x_max, y_min, y_max = bbox
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Process the cropped image
    cleaned_image = process_image(cropped_image, sigma, min_size, closing_disk_size)
    labeled_image, regions = detect_craters(cleaned_image)

    return cropped_image, labeled_image, regions

# Main function to execute the workflow
def main():
    # Provide the path to the TIFF file on Google Drive
    file_path = '/content/drive/MyDrive/Colab Notebooks/Lunar_Clementine_NIR_cal_empcor_500m.tif'  # Your file path

    start_time = time.time()

    # Load the image
    image = load_tiff_image(file_path)

    # Define the bounding box for the region of interest (ROI)
    bbox = (1000, 1500, 2000, 2500)  # Example bounding box (x_min, x_max, y_min, y_max)

    # Zoom into the region, process and display results
    cropped_image, labeled_image, regions = zoom_and_process(image, bbox)

    display_craters(cropped_image, labeled_image, regions, f"Detected Craters: {len(regions)} found in Zoomed Area")

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

# Run the main function
main()
