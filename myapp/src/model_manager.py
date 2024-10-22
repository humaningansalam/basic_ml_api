import os
import shutil
import logging
from zipfile import ZipFile
from collections import OrderedDict
import numpy as np
import keras
from typing import Dict, Any, Optional, Tuple

from myapp.common.prometheus_metric import get_metrics
import myapp.common.tool_util as tool_util

class ModelManager:
    def __init__(self, store_path: str, max_cache_size: int = 10):
        self.store_path = store_path
        self.max_cache_size = max_cache_size
        self.metadata_store: Dict[str, Dict[str, str]] = {}
        self.model_cache = OrderedDict()
        self.metrics = get_metrics()
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
            remove_path = self.metadata_store[model_hash]['file_path']
            if os.path.exists(remove_path):
                shutil.rmtree(remove_path)
                logging.info(f"Removed old model: {model_hash}")
            del self.metadata_store[model_hash]
            
            if model_hash in self.model_cache:
                del self.model_cache[model_hash]
                
        self.metrics.set_model_cache_usage(len(self.model_cache))

    def load_model_to_cache(self, model_hash: str) -> Optional[Any]:
        """모델을 캐시에 로드하고 반환하는 함수"""
        try:
            if model_hash in self.model_cache:
                self.model_cache.move_to_end(model_hash)
                return self.model_cache[model_hash]
                
            if len(self.model_cache) >= self.max_cache_size:
                oldest_key, _ = self.model_cache.popitem(last=False)
                logging.info(f"Removing oldest model from cache: {oldest_key}")
            
            if model_hash not in self.metadata_store:
                raise KeyError(f"Model hash {model_hash} not found in metadata store")
                
            model_file_path = self.metadata_store[model_hash]['file_path']
            if not os.path.exists(model_file_path):
                raise OSError(f"Model path does not exist: {model_file_path}")
                
            model = keras.models.load_model(model_file_path)
            self.model_cache[model_hash] = model
            self.metrics.set_model_cache_usage(len(self.model_cache))
            logging.info(f"Model {model_hash} loaded into cache successfully")
            return model
            
        except (OSError, KeyError) as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
            
        except Exception as e:
            logging.error(f"Unexpected error loading model: {str(e)}")
            raise

    def predict(self, model_hash: str, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """예측 수행 함수"""
        try:
            if model_hash not in self.metadata_store:
                raise KeyError("Model not found in metadata store")

            if model_hash in self.model_cache:
                self.metrics.increment_cache_hit()
                model = self.model_cache[model_hash]
            else:
                self.metrics.increment_cache_miss()
                model = self.load_model_to_cache(model_hash)

            self.metadata_store[model_hash]['used'] = tool_util.get_kr_time()
            prediction = model.predict(data)
            self.metrics.increment_predictions_completed()
            
            return prediction, 200

        except (KeyError, OSError) as e:
            self.metrics.increment_error_count('predict_model_load_failed')
            logging.error(f"Model loading failed: {str(e)}")
            raise

        except Exception as e:
            self.metrics.increment_error_count('predict_error')
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def upload_model(self, model_file, model_hash: str) -> Tuple[str, int]:
        """모델 업로드 및 저장"""
        try:
            model_folder_path = os.path.join(self.store_path, model_hash)
            os.makedirs(model_folder_path, exist_ok=True)
            
            model_file_path = os.path.join(model_folder_path, model_file.filename)
            model_file.save(model_file_path)
            
            with ZipFile(model_file_path, 'r') as zipObj:
                zipObj.extractall(model_folder_path)
            
            os.remove(model_file_path)
            
            self.metadata_store[model_hash] = {
                'file_path': model_folder_path,
                'used': tool_util.get_kr_time()
            }
            
            # 새로 업로드된 모델을 캐시에 로드
            self.load_model_to_cache(model_hash)
            
            return 'File uploaded and processed successfully', 200
            
        except Exception as e:
            self.metrics.increment_error_count('upload_model_error')
            if os.path.exists(model_folder_path):
                shutil.rmtree(model_folder_path)
            raise