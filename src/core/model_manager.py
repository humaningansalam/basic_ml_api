#core/model_manager
import os
import shutil
import logging
import threading
import time
from zipfile import ZipFile, BadZipFile
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple

import tensorflow as tf
import numpy as np

from src.common import utils
from src.common.metrics import get_metrics

class ModelManager:
    def __init__(self, store_path: str, max_cache_size: int = 10):
        """모델 관리자 초기화 (경로 설정, 캐시 설정)"""
        self.store_path = store_path
        self.max_cache_size = max_cache_size
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.model_cache = OrderedDict()
        self.metrics = get_metrics()
        self.logger = logging.getLogger(__name__)
        
        # 초기화 작업
        self._load_metadata_store()
        self._start_cleanup_scheduler()

    def _load_metadata_store(self) -> None:
        """기존 저장된 모델들의 메타데이터 로드"""
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
            return

        for model_hash in os.listdir(self.store_path):
            model_folder_path = os.path.join(self.store_path, model_hash)
            if os.path.isdir(model_folder_path):
                self.metadata_store[model_hash] = {
                    'file_path': model_folder_path,
                    'used': utils.get_kr_time()
                }

    def _start_cleanup_scheduler(self):
        """주기적인 모델 정리 스케줄러 시작"""
        def scheduled_cleanup():
            while True:
                self.clean_old_models()
                time.sleep(5 * 3600)  # 5시간 대기

        thread = threading.Thread(target=scheduled_cleanup, daemon=True)
        thread.start()

    def clean_old_models(self) -> None:
        """오래된 모델 삭제 (1주일 이상 미사용)"""
        try:
            to_remove = []
            for model_hash, metadata in self.metadata_store.items():
                if metadata['used'] < utils.one_week_ago():
                    to_remove.append(model_hash)

            for model_hash in to_remove:
                remove_path = os.path.join(self.store_path, model_hash)
                if os.path.exists(remove_path):
                    shutil.rmtree(remove_path)
                    self.logger.info(f"Removed old model: {model_hash}")
                del self.metadata_store[model_hash]
                
                # 캐시에서도 제거
                if model_hash in self.model_cache:
                    del self.model_cache[model_hash]
                    self.metrics.set_model_cache_usage(len(self.model_cache))
                    
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def load_model_to_cache(self, model_hash: str) -> Optional[Any]:
        """모델을 캐시에 로드하고 반환하는 함수 (LRU 방식)"""
        if model_hash in self.model_cache:
            self.model_cache.move_to_end(model_hash)
            self.metrics.increment_cache_hit()
            return self.model_cache[model_hash]

        # 캐시 크기 초과 시 가장 오래된 모델 제거
        if len(self.model_cache) >= self.max_cache_size:
            self.model_cache.popitem(last=False)

        if model_hash not in self.metadata_store:
            raise KeyError(f"Model hash {model_hash} not found")

        model_folder_path = self.metadata_store[model_hash]['file_path']
        keras_file_path = None
        
        # .keras 파일 탐색
        for root, _, files in os.walk(model_folder_path):
            for file in files:
                if file.endswith(".keras"):
                    keras_file_path = os.path.join(root, file)
                    break
            if keras_file_path: break

        if not keras_file_path:
            raise OSError("No .keras file found")

        model = tf.keras.models.load_model(keras_file_path)
        self.model_cache[model_hash] = model
        self.metrics.increment_cache_miss()
        self.metrics.set_model_cache_usage(len(self.model_cache))
        return model

    def predict(self, model_hash: str, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """예측 수행 함수"""
        try:
            model = self.load_model_to_cache(model_hash)
            self.metadata_store[model_hash]['used'] = utils.get_kr_time()
            prediction = model.predict(data)
            return prediction, 200
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def upload_model(self, model_file, model_hash: str) -> Tuple[str, int]:
        """모델 업로드 및 저장 (Zip 해제)"""
        if not model_hash or len(model_hash) < 8:
             raise ValueError("Invalid model hash")
             
        model_folder_path = os.path.join(self.store_path, model_hash)
        os.makedirs(model_folder_path, exist_ok=True)
        temp_zip_path = os.path.join(model_folder_path, 'temp.zip')
        model_file.save(temp_zip_path)

        try:
            with ZipFile(temp_zip_path, 'r') as zip_ref:
                if not any(name.endswith('.keras') for name in zip_ref.namelist()):
                    raise ValueError("No .keras file in zip")
                zip_ref.extractall(model_folder_path)
        except BadZipFile:
            raise ValueError("Invalid zip file")
        finally:
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)

        self.metadata_store[model_hash] = {
            'file_path': model_folder_path,
            'used': utils.get_kr_time()
        }
        return 'Model uploaded successfully', 200

    def get_model_info(self, model_hash: str) -> Dict[str, str]:
        """모델 정보 반환"""
        if model_hash not in self.metadata_store:
            raise KeyError(f"Model {model_hash} not found")
        return self.metadata_store[model_hash]