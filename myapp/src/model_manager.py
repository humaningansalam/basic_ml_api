import os
import shutil
import logging
from zipfile import ZipFile, BadZipFile
from collections import OrderedDict
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Any, Optional, Tuple

import myapp.common.tool_util as tool_util

class ModelManager:
    def __init__(self, store_path: str, max_cache_size: int = 10):
        self.store_path = store_path
        self.max_cache_size = max_cache_size
        self.metadata_store: Dict[str, Dict[str, str]] = {}
        self.model_cache = OrderedDict()
        self.load_metadata_store()

    def load_metadata_store(self) -> None:
        """메타데이터 저장소 초기화"""
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
            return

        for model_hash in os.listdir(self.store_path):
            model_folder_path = os.path.join(self.store_path, model_hash)
            if os.path.isdir(model_folder_path):
                self.metadata_store[model_hash] = {
                    'file_path': model_folder_path,
                    'used': tool_util.get_kr_time()
                }

    def clean_old_models(self) -> None:
        """오래된 모델 삭제"""
        to_remove = []
        for model_hash, metadata in self.metadata_store.items():
            if metadata['used'] < tool_util.one_week_ago():
                to_remove.append(model_hash)

        for model_hash in to_remove:
            remove_path = os.path.join(self.store_path, model_hash)
            if os.path.exists(remove_path):
                shutil.rmtree(remove_path)
                logging.info(f"Removed old model: {model_hash}")
            del self.metadata_store[model_hash]

            if model_hash in self.model_cache:
                del self.model_cache[model_hash]

    def load_model_to_cache(self, model_hash: str) -> Optional[Any]:
        """모델을 캐시에 로드하고 반환하는 함수"""
        if model_hash in self.model_cache:
            self.model_cache.move_to_end(model_hash)
            return self.model_cache[model_hash]

        if len(self.model_cache) >= self.max_cache_size:
            oldest_key, _ = self.model_cache.popitem(last=False)
            logging.info(f"Removing oldest model from cache: {oldest_key}")

        if model_hash not in self.metadata_store:
            raise KeyError(f"Model hash {model_hash} not found in metadata store")

        model_folder_path = self.metadata_store[model_hash]['file_path']
        if not os.path.exists(model_folder_path):
            raise OSError(f"Model path does not exist: {model_folder_path}")

        keras_file_path = None
        for root, _, files in os.walk(model_folder_path):
            for file in files:
                if file.endswith(".keras"):
                    keras_file_path = os.path.join(root, file)
                    break
            if keras_file_path:
                break

        if keras_file_path is None:
            raise OSError(f"No .keras file found in model folder: {model_folder_path}")

        model = tf.keras.models.load_model(keras_file_path)
        self.model_cache[model_hash] = model
        logging.info(f"Model {model_hash} loaded into cache successfully from .keras file")
        return model

    def predict(self, model_hash: str, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """예측 수행 함수"""
        try:
            model = self.load_model_to_cache(model_hash)
            self.metadata_store[model_hash]['used'] = tool_util.get_kr_time()
            prediction = model.predict(data)
            return prediction, 200

        except (KeyError, OSError) as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise

        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def upload_model(self, model_file, model_hash: str) -> Tuple[str, int]:
        """모델 업로드 및 저장"""
        model_folder_path = os.path.join(self.store_path, model_hash)
        os.makedirs(model_folder_path, exist_ok=True)

        temp_zip_path = os.path.join(model_folder_path, 'temp.zip')
        model_file.save(temp_zip_path)

        try:
            with ZipFile(temp_zip_path, 'r') as zip_ref:
                keras_file_found = False
                for name in zip_ref.namelist():
                    if name.endswith('.keras'):
                        keras_file_found = True
                        break
                if not keras_file_found:
                    raise ValueError("No .keras file found in the zip archive.")
                zip_ref.extractall(model_folder_path)

        except BadZipFile:
            logging.error("Bad zip file uploaded.")
            raise ValueError("Invalid zip file uploaded.")

        except Exception as e:
            logging.error(f"Error extracting .keras file: {e}")
            raise

        finally:
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)

        self.metadata_store[model_hash] = {
            'file_path': model_folder_path,
            'used': tool_util.get_kr_time()
        }

        return 'File uploaded and processed successfully', 200

    def get_model_info(self, model_hash: str) -> Dict[str, str]:
        """모델 정보 반환"""
        if model_hash not in self.metadata_store:
            raise KeyError(f"Model {model_hash} not found")
        return self.metadata_store[model_hash]