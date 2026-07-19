import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf


def create_demo_archive(output_path: Path) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(1, use_bias=False),
    ])
    model.layers[0].set_weights([np.array([[2.0]], dtype=np.float32)])

    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / 'model.keras'
        model.save(model_path)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(model_path, arcname='model.keras')

    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Create a deterministic demo model archive.')
    parser.add_argument('--output', type=Path, default=Path('demo-model.zip'))
    args = parser.parse_args(argv)
    output_path = create_demo_archive(args.output)
    print(f'Created {output_path}')


if __name__ == '__main__':
    main()
