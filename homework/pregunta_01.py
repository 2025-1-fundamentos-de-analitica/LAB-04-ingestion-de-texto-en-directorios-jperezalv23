import os
import zipfile
import pandas as pd

def unzip_file(src_zip: str, dest_folder: str) -> None:
    with zipfile.ZipFile(src_zip, "r") as archive:
        archive.extractall(dest_folder)

def create_dataframe(root_dir: str, dataset_type: str) -> pd.DataFrame:
    records = []
    labels = ["negative", "positive", "neutral"]
    for label in labels:
        sentiment_folder = os.path.join(root_dir, dataset_type, label)
        for doc in os.listdir(sentiment_folder):
            doc_path = os.path.join(sentiment_folder, doc)
            with open(doc_path, encoding="utf-8") as file:
                text = file.read()
                records.append({"phrase": text, "target": label})
    return pd.DataFrame(records)

def pregunta_01():
    archive_path = "files/input.zip"
    extract_folder = "files"
    input_folder = os.path.join(extract_folder, "input")
    output_folder = os.path.join(extract_folder, "output")
    os.makedirs(output_folder, exist_ok=True)

    unzip_file(archive_path, extract_folder)

    for partition in ["test", "train"]:
        dataset = create_dataframe(input_folder, partition)
        output_file = os.path.join(output_folder, f"{partition}_dataset.csv")
        dataset.to_csv(output_file, index=False)
