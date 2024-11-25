# Unsupervised Cell Image Segmentation

This project implements an unsupervised approach to segment cell images into distinct regions:

+ **Background**
+ **Cytoplasm**
+ **Nuclei**

The primary goal is to identify these structures in microscopy images without the need for labeled data.

The segmentation process utilizes a range of clustering techniques, each with its unique strengths. The **Clusterer** algorithm aggregates the results from all the individual clustering methods using a majority voting mechanism. This ensemble approach combines the strengths of each algorithm, potentially improving overall segmentation performance, especially when individual methods yield conflicting results.

By relying on the collective output of multiple algorithms, the **Clusterer** provides more robust and stable segmentation results. In some cases, it may even outperform individual algorithms, depending on the specific characteristics of the data and images.

## Features

- Unsupervised segmentation using clustering techniques.
- Preprocessing pipeline for image preparation.
- Conversion of clustering results into segmentation maps.
- Utility functions for image loading, resizing, and visualization.

## Installation

**Prerequisite:** Ensure [Poetry](https://python-poetry.org/docs/#installation) is installed on your system.

1. Clone the repository:
   ```bash
   git clone https://github.com/PetrFridrich/UnsupervisedCellImageSegmentation.git
   ```
2.  Navigate to the project directory:
    ```bash
    cd UnsupervisedCellImageSegmentation
    ```
3. Install the dependencies with Poetry:
    ```bash 
    poetry install
    ```
This will install all required dependencies as defined in pyproject.toml.

## Project Structure

The project is organized as follows:

<pre>
UnsupervisedCellImageSegmentation/
├── data/                                   # Folder for datasets
│   ├── images/                                 # Raw cell images
│   ├── labels/                                 # Ground truth labels
│   └── other_images/                           # Raw cell images without labels
├── src/                                    # Folder with source code
│   ├── analysis/                               # Folder for notebooks related to analysis
│   │   ├── image_analysis.ipynb                    # Notebook for analyzing input images
│   │   └── evaluation_analysis.ipynb               # Notebook for evaluating the performance of segmentation
│   ├── common/                                 # Folder for common utilities shared across the project
│   │   └── image_utils.py                          # Image utility functions (e.g., loading, resizing, preprocessing)
│   ├── model/                                  # Folder for models
│   │   └── clusterer.py                            # Clusterer class implementation
│   └── __main__.py                             # Main script to run Unsupervised Cell Image Segmentation
├── .gitignore                              # Git ignore file
├── pyproject.toml                          # Poetry project configuration and dependencies
├── poetry.lock                             # Poetry lock file to ensure consistent dependencies
├── LICENSE                                 # License for the project
└── README.md                               # This README file
</pre>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
