"""
Full pipeline: preprocessing, generators, ResNet50 training, and save final model
This is the same content that was previously in `code.py` but placed here to avoid
shadowing the standard library module named `code`.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_ROOT = SCRIPT_DIR
CLEAN_ROOT = os.path.join(SCRIPT_DIR, 'cleaned')
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

def ensure_extensions(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.commonpath([os.path.abspath(dirpath), os.path.abspath(CLEAN_ROOT)]) == os.path.abspath(CLEAN_ROOT):
            continue
        for f in filenames:
            if f.startswith('.'):
                continue
            if '.' not in f:
                try:
                    os.rename(os.path.join(dirpath, f), os.path.join(dirpath, f + '.jpg'))
                except Exception:
                    pass

def preprocess_and_copy(src_root, dst_root, img_size=IMG_SIZE):
    if os.path.exists(dst_root):
        # keep existing cleaned content if present
        pass
    else:
        os.makedirs(dst_root, exist_ok=True)
    total = 0
    for root, dirs, files in os.walk(src_root):
        if os.path.commonpath([os.path.abspath(root), os.path.abspath(dst_root)]) == os.path.abspath(dst_root):
            continue
        rel = os.path.relpath(root, src_root)
        parts = rel.split(os.sep)
        if rel == '.':
            class_name = 'unknown'
        else:
            class_name = parts[0]
        out_dir = os.path.join(dst_root, class_name)
        os.makedirs(out_dir, exist_ok=True)
        for fname in files:
            if fname.startswith('.'):
                continue
            src_path = os.path.join(root, fname)
            img = cv2.imread(src_path)
            if img is None:
                print(f"Warning: can't read {src_path}")
                continue
            try:
                img = cv2.GaussianBlur(img, (3,3), 0)
                img = cv2.medianBlur(img, 3)
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l,a,b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l,a,b))
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                img = cv2.resize(img, img_size)
                out_name = fname if '.' in fname else fname + '.jpg'
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, (img).astype('uint8'))
                total += 1
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
    print(f"Preprocessing complete. Saved {total} images to {dst_root}")

def build_generators(clean_root, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=25, horizontal_flip=True, brightness_range=[0.7,1.3], validation_split=0.2)
    train_gen = datagen.flow_from_directory(clean_root, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training', shuffle=True)
    val_gen = datagen.flow_from_directory(clean_root, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation', shuffle=False)
    return train_gen, val_gen

def build_model(num_classes, input_shape=(224,224,3), lr=1e-4):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base.layers[:-10]:
        layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print('Dataset root:', RAW_ROOT)
    ensure_extensions(RAW_ROOT)
    preprocess_and_copy(RAW_ROOT, CLEAN_ROOT)
    train_gen, val_gen = build_generators(CLEAN_ROOT)
    print('Found classes:', train_gen.class_indices)
    model = build_model(num_classes=len(train_gen.class_indices))
    model.summary()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint = ModelCheckpoint(f'best_model_{timestamp}.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint, early, reduce_lr])
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy', []), label='Train Acc')
    plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
    plt.legend(); plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.legend(); plt.title('Loss')
    plt.tight_layout()
    plt.show()
    # Save final model as requested
    model.save('oral_cancer_model.h5')
    print('Saved final model to oral_cancer_model.h5')

if __name__ == '__main__':
    main()
