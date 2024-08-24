## **README**

### **Lunar Crater Detection**

**Authors:** Mr. Prathmesh Barapatre, Ms. Sejal Jain

This repository contains Python code for detecting and analyzing lunar craters using the Lunar Reconnaissance Orbiter (LRO) dataset. The code leverages image processing techniques and computer vision algorithms to identify and characterize craters on the lunar surface.

Dataset used in the project can be found here, this was taken from LRO Website(https://pds-imaging.jpl.nasa.gov/volumes/lro.html)
link for dataset: https://drive.google.com/file/d/1L-14-V4QG1VUtHarI1agPCGRAtBlrfvf/view?usp=sharing

**Key Features:**
* **Image Loading and Preprocessing:** Loads TIFF images from Google Drive and prepares them for analysis by converting to grayscale, smoothing, sharpening, edge detection, and thresholding.
* **Crater Detection:** Identifies crater regions based on connected components and morphological operations.
* **Visualization:** Displays intermediate processing steps and overlays detected craters on the original image.
* **Regional Analysis:** Allows for zooming into specific regions of interest for detailed crater analysis.

**Requirements:**
* Python 3.x
* NumPy
* Matplotlib
* Scikit-image
* Google Colab (for accessing Google Drive)

**Usage:**
1. **Mount Google Drive:** If using Google Colab, mount your Google Drive to access the LRO image.
2. **Modify File Path:** Update the `file_path` variable in the `main()` function to point to your LRO TIFF image.
3. **Define Bounding Box (Optional):** If desired, adjust the `bbox` variable to specify a region of interest for zoomed-in analysis.
4. **Run the Code:** Execute the `main()` function to perform crater detection and visualization.

**Future Enhancements:**
* Integrate machine learning for crater classification.
* Develop analysis modules for extracting crater properties (diameter, depth, etc.).
* Improve user interface for interactive region selection and analysis.

**Contributions:**
Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests.
