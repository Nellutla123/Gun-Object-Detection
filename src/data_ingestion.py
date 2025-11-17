import os
import sys
import kagglehub
import shutil
import zipfile
from src.logger import get_logger
from src.custom_exception import CustomException
from config.data_ingestion_config import *


logger = get_logger(__name__)

class DataIngestion():

    def __init__(self, dataset_name:str, target_dir:str):
        self.dataset_name = dataset_name
        self.target_dir = target_dir

    def create_raw_dir(self):
        """
        Create the raw directory if it does not exist.
        """
        raw_dir = os.path.join(self.target_dir, 'raw')
        
        if not os.path.exists(raw_dir):
            try:
                os.makedirs(raw_dir)
                logger.info(f"Created raw directory at {raw_dir}")
        
            except Exception as e:
                logger.error(f"Error creating raw directory...")
                raise CustomException("Failed to create raw directory", str(e))
        
        return raw_dir
            
    def extract_images_and_labels(self, path:str, raw_dir:str):
        """
        Extract images and labels from the dataset.
        """
        try:
            if path.endswith('.zip'):
                logger.info(f"Extracting zipfile")
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(path)
            
            images_folder = os.path.join(path, 'Images')
            labels_folder = os.path.join(path, 'Labels')

            if os.path.exists(images_folder):
                shutil.move(images_folder, os.path.join(raw_dir, 'Images'))
                logger.info(f"Moved images to {os.path.join(raw_dir, 'Images')}")
            else:
                logger.info(f"Images folder not found at {images_folder}")


            if os.path.exists(labels_folder):
                shutil.move(labels_folder, os.path.join(raw_dir, 'Labels'))
                logger.info(f"Moved labels to {os.path.join(raw_dir, 'Labels')}")
            else:
                logger.info(f"labels folder not found at {labels_folder}")

        except Exception as e:
            logger.error(f"Error extracting images and labels: {str(e)}")
            raise CustomException("Failed to extract images and labels", str(e))
        
    def download_dataset(self, raw_dir:str):
        """
        Download the dataset from Kaggle.
        """
        try:
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Downloaded dataset from {path}")

            self.extract_images_and_labels(path, raw_dir)

        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise CustomException("Failed to download dataset", str(e))
        
    def run(self):
        """
        Run the data ingestion process.
        """
        try:
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
            logger.info("Data ingestion completed successfully.")
        
        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            raise CustomException("Data ingestion failed", str(e))
        
if __name__ == "__main__":
    data_ingestion = DataIngestion(dataset_name=DATASET_NAME, target_dir=TARGET_DIR)
    data_ingestion.run()