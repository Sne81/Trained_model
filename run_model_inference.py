import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = 'oral_cancer_model.h5'
TEST_DIR = 'test'
IMG_SIZE = (224, 224)


def find_sample_image(test_dir):
    # Find first image file under subdirectories of test/
    for cls in sorted(os.listdir(test_dir)):
        cls_path = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                return os.path.join(cls_path, fname), cls
    return None, None


def load_and_preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    if not os.path.exists(MODEL_PATH):
        print('Model not found at', MODEL_PATH)
        return 2

    sample_path, sample_cls = find_sample_image(TEST_DIR)
    if sample_path is None:
        print('No test images found under', TEST_DIR)
        return 3

    print('Loading model from', MODEL_PATH)
    start = time.time()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    load_time = time.time() - start
    print(f'Model loaded in {load_time:.2f}s')
    try:
        model.summary()
    except Exception:
        pass

    print('Using sample image:', sample_path, ' (true label:', sample_cls, ')')
    x = load_and_preprocess(sample_path)
    start = time.time()
    preds = model.predict(x)
    pred_time = time.time() - start
    preds = np.asarray(preds).reshape(-1)
    top_idx = int(np.argmax(preds))

    # Build class list from test subfolders (sorted)
    classes = [d for d in sorted(os.listdir(TEST_DIR)) if os.path.isdir(os.path.join(TEST_DIR, d))]
    label = classes[top_idx] if 0 <= top_idx < len(classes) else str(top_idx)

    print(f'Prediction time: {pred_time:.3f}s')
    print('Predicted class index:', top_idx)
    print('Predicted label:', label)
    print('Probabilities:')
    for i, p in enumerate(preds):
        name = classes[i] if i < len(classes) else str(i)
        print(f'  {i:02d} {name:20s} {p:.5f}')


if __name__ == '__main__':
    raise SystemExit(main())
