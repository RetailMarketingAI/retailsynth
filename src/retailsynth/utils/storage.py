import os
import shutil
from pathlib import Path

import pandas as pd


def load_result(base_path):
    # load stored feature data frame from local path
    store_visit_df = pd.read_parquet(
        Path(base_path, "store_visit_df.parquet").resolve()
    )
    category_choice_df = pd.read_parquet(
        Path(base_path, "category_choice_df.parquet").resolve()
    )
    product_choice_df = pd.read_parquet(
        Path(base_path, "product_choice_df.parquet").resolve()
    )
    product_demand_df = pd.read_parquet(
        Path(base_path, "product_demand_df.parquet").resolve()
    )
    products_df = pd.read_parquet(
        Path(base_path, "annotated_products.parquet").resolve()
    )
    return {
        "store_visit": store_visit_df,
        "category_choice": category_choice_df,
        "product_choice": product_choice_df,
        "product_demand": product_demand_df,
        "products": products_df,
    }


def clear_feature_directory(feature_path):
    shutil.rmtree(feature_path)


def clear_cwd(subdir: str = "", exclude: str = None):
    """empty the current directory

        Parameters
        ----------
        subdir (str, optional): clear only the specified subdirectory is given. Defaults to "".
        exclude (str, optional): specify type of file to keep in the directory. Defaults to None.
    """
    path = Path(os.getcwd(), subdir).resolve()
    if not path.exists():
        return
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if exclude is not None and file_path.endswith(exclude):
                continue
            elif os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
