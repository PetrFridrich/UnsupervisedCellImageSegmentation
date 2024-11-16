import os
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from model.clusterer import Clusterer
from common.image_utils import load_image, load_label, save_image, resize_img, img_to_X, y_to_img, reshape_label


def process_images(data_path, output_path):
    """
    Segments images in the specified directory and saves the segmented outputs.

    Parameters:
        data_path (Path): Path to the directory containing the input images.
        output_path (Path): Path to the directory where segmented images will be saved.

    Returns:
        None
    """

    print('Starting image segmentation process...')

    # Define pipelines for image preprocessing
    into_model_pipeline = Pipeline([
        ('load_image', FunctionTransformer(load_image)),
        ('resize_image', FunctionTransformer(resize_img)),
        ('transform_image', FunctionTransformer(img_to_X)),
    ])
    
    out_model_pipeline = Pipeline([
        ('transform_vector', FunctionTransformer(y_to_img)),
    ])

    # Initialize model
    model = Clusterer(n_clusters=3)
    image_names = os.listdir(data_path)

    for _, name in tqdm(enumerate(image_names), total=len(image_names), desc='Processing images'):

        # Preprocess and segment the image
        X = into_model_pipeline.transform(data_path / Path(name))

        y = model.fit_predict(X)

        # Transform segmentation result back to image and save
        img_seg = out_model_pipeline.transform(y)
        save_image(img_seg, f'{output_path / Path(name).stem}.png')

    return None


def evaluate_images(data_path, label_path, output_path):
    """
    Evaluates image segmentation results against ground truth labels and computes metrics.

    Parameters:
        data_path (Path): Path to the directory containing the input images.
        label_path (Path): Path to the directory containing the ground truth labels.
        output_path (Path): Path to the directory where evaluation results will be saved.

    Returns:
        None
    """

    print('Starting image evaluation process...')

    # Define pipelines for image and label preprocessing
    into_model_pipeline_img = Pipeline([
        ('load_image', FunctionTransformer(load_image)),
        ('resize_image', FunctionTransformer(resize_img)),
        ('transform_image', FunctionTransformer(img_to_X)),
    ])

    into_model_pipeline_lbl = Pipeline([
        ('load_label', FunctionTransformer(load_label)),
        ('resize_label', FunctionTransformer(resize_img)),
        ('reshape_label', FunctionTransformer(reshape_label)),
    ])

    # Initialize model
    model = Clusterer(n_clusters=3)
    image_names = os.listdir(data_path)
    label_names = os.listdir(label_path)

    result_df = pd.DataFrame()

    for _, (img_name, lbl_name) in tqdm(enumerate(zip(image_names, label_names)), total=np.min([len(image_names), len(label_names)]), desc='Processing images'):

        # Preprocess image and label
        X = into_model_pipeline_img.transform(data_path / Path(img_name))
        y_target = into_model_pipeline_lbl.transform(label_path / Path(lbl_name))

        # Evaluate model and append results
        df = model.eval(X, y_target)
        result_df = pd.concat([result_df, df], axis=0)

    # Save evaluation results
    result_df.insert(0, 'Image Name', image_names)
    result_df.to_csv(output_path / 'Jaccard_score.csv', index=False)

    return None


def main():
    """
    Main function to handle the image segmentation and evaluation processes.

    It performs the following:
        - Segments images from a given directory and saves results.
        - Evaluates segmentation results against labels and generates a CSV report.

    Parameters:
        None

    Returns:
        None
    """

    # Paths for image segmentation
    DATA_PATH = Path('./data/images/')
    OUTPUT_PATH_SEGMENTATION = Path('./results/segmentation/')
    OUTPUT_PATH_SEGMENTATION.mkdir(parents=True, exist_ok=True)

    # Run segmentation process
    process_images(DATA_PATH, OUTPUT_PATH_SEGMENTATION)

    # Paths for evaluation
    LABEL_PATH = Path('./data/labels/')
    OUTPUT_PATH_EVALUATION = Path('./results/evaluation/')
    OUTPUT_PATH_EVALUATION.mkdir(parents=True, exist_ok=True)

    # Run evaluation process
    evaluate_images(DATA_PATH, LABEL_PATH, OUTPUT_PATH_EVALUATION)

    return None


if __name__ == '__main__':

    main()