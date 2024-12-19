from tensorflow.keras.models import load_model

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
            model_file_path = os.path.join(self.store_path, model_hash + '.keras')
            if os.path.isfile(model_file_path):
                self.metadata_store[model_hash] = {
                    'file_path': model_file_path,
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
                os.remove(remove_path)
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

        model_file_path = self.metadata_store[model_hash]['file_path']
        if not os.path.exists(model_file_path):
            raise OSError(f"Model path does not exist: {model_file_path}")

        model = load_model(model_file_path)
        self.model_cache[model_hash] = model
        logging.info(f"Model {model_hash} loaded into cache successfully")
        return model

    def upload_model(self, model_file, model_hash: str) -> Tuple[str, int]:
        """모델 업로드 및 저장"""
        try:
            model_file_path = os.path.join(self.store_path, model_hash + '.keras')
            model_file.save(model_file_path)

            self.metadata_store[model_hash] = {
                'file_path': model_file_path,
                'used': tool_util.get_kr_time()
            }

            self.load_model_to_cache(model_hash)
            return 'File uploaded and processed successfully', 200

        except Exception as e:
            logging.error(f"Error uploading model: {e}")
            if os.path.exists(model_file_path):
                os.remove(model_file_path)
            raise

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

    def get_model_info(self, model_hash: str) -> Dict[str, str]:
        """모델 정보 반환"""
        if model_hash not in self.metadata_store:
            raise KeyError(f"Model {model_hash} not found")
        return self.metadata_store[model_hash]
