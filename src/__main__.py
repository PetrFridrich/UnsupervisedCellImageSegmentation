import pandas as pd
import os

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from pathlib import Path

from model.clusterer import Clusterer
from common.image_utils import load_image, load_label, save_image, resize_img, img_to_X, y_to_img, reshape_label


def process(data_path, output_path):

    into_model_pipeline = Pipeline([
                                    ('load_image', FunctionTransformer(load_image)),
                                    ('resize_image', FunctionTransformer(resize_img)),
                                    ('transform_image', FunctionTransformer(img_to_X))
    ])

    out_model_pipeline = Pipeline([
                                    ('transform_vector', FunctionTransformer(y_to_img))
    ])

    model = Clusterer(n_clusters=3)
    names = os.listdir(data_path)

    for _, name in enumerate(names):

        X = into_model_pipeline.transform(data_path / Path(name))

        y = model.fit_predict(X)

        img_seg = out_model_pipeline.transform(y)

        save_image(img_seg, f'{output_path / Path(name).stem}.png')

    return None


def process(data_path, label_path, output_path):

    into_model_pipeline_img = Pipeline([
                                    ('load_image', FunctionTransformer(load_image)),
                                    ('resize_image', FunctionTransformer(resize_img)),
                                    ('transform_image', FunctionTransformer(img_to_X))
    ])

    into_model_pipeline_lbl = Pipeline([
                                    ('load_label', FunctionTransformer(load_label)),
                                    ('resize_label', FunctionTransformer(resize_img)),
                                    ('reshape_label', FunctionTransformer(reshape_label))
    ])

    model = Clusterer(n_clusters=3)

    img_names = os.listdir(data_path)
    lbl_names = os.listdir(label_path)

    result_df = pd.DataFrame()

    for _, (img_name, lbl_name) in enumerate(zip(img_names, lbl_names)):

        X = into_model_pipeline_img.transform(data_path / Path(img_name))

        y_target = into_model_pipeline_lbl.transform(label_path / Path(lbl_name))

        df = model.eval(X,y_target)

        result_df = pd.concat([result_df, df], axis=0)

    result_df.insert(0, 'Names', img_names)
    result_df.to_csv(output_path / Path('Jaccard_score.csv'), index=False)

    return None


def main():

    DATA_PATH = Path('./data/images/')
    LABEL_PATH = Path('./data/labels/')
    OUTPUT_PATH = Path('./results/evaluation/')

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    process(DATA_PATH, LABEL_PATH, OUTPUT_PATH)


    # DATA_PATH = Path('./data/images/')
    # OUTPUT_PATH = Path('./results/segmentation/')

    # OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # process(DATA_PATH, OUTPUT_PATH)


if __name__ == '__main__':
    print('Hello, home!')

    main()
