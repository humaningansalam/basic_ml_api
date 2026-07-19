import zipfile

import numpy as np
import tensorflow as tf

from examples.create_demo_model import create_demo_archive


def test_create_demo_archive_contains_loadable_deterministic_model(tmp_path):
    archive_path = create_demo_archive(tmp_path / 'demo-model.zip')

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ['model.keras']
        archive.extractall(tmp_path / 'extracted')

    model = tf.keras.models.load_model(tmp_path / 'extracted' / 'model.keras')
    prediction = model(np.array([[3.0]], dtype=np.float32), training=False).numpy()

    assert prediction.tolist() == [[6.0]]
