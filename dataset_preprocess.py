# ===============================================================
# ===  IMPORT LIBRARIES  ========================================
# ===============================================================
import os, cv2, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===============================================================
# ===  PATHS & PARAMETERS  ======================================
# ===============================================================
# Dataset root detection: prefer an explicit folder, otherwise use the script folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# If you have a raw folder configured, set RAW_ROOT manually here. Otherwise the script
# will look for common dataset folders (train/val/test) under the script directory.
PREFERRED_RAW = None  # e.g. r"C:\path\to\raw_dataset" or None to auto-detect

if PREFERRED_RAW and os.path.exists(PREFERRED_RAW):
    RAW_ROOT = PREFERRED_RAW
else:
    # Look for common subfolders in the workspace
    candidates = [
        os.path.join(SCRIPT_DIR, 'train'),
        os.path.join(SCRIPT_DIR, 'dataset'),
        os.path.join(SCRIPT_DIR, 'data'),
        SCRIPT_DIR
    ]
    RAW_ROOT = None
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            # prefer the directory that contains subfolders for classes
            RAW_ROOT = c
            break
    if RAW_ROOT is None:
        # fallback to script dir
        RAW_ROOT = SCRIPT_DIR

# Cleaned output directory (will contain one subfolder per class)
CLEAN_ROOT = os.path.join(SCRIPT_DIR, 'cleaned')
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

os.makedirs(CLEAN_ROOT, exist_ok=True)

# ===============================================================
# ===  STEP 1: RENAME IMAGES TO ADD EXTENSIONS  =================
# ===============================================================
print("🔹 Renaming images to add .jpg extensions if missing (recursive)...")
for root, dirs, files in os.walk(RAW_ROOT):
    # skip the cleaned output folder if it's inside RAW_ROOT
    if os.path.commonpath([os.path.abspath(root), os.path.abspath(CLEAN_ROOT)]) == os.path.abspath(CLEAN_ROOT):
        continue
    for filename in files:
        src_path = os.path.join(root, filename)
        # skip hidden/system files
        if filename.startswith('.'):
            continue
        # if there's no extension, add .jpg
        if '.' not in filename:
            new_path = src_path + '.jpg'
            try:
                os.rename(src_path, new_path)
            except Exception as e:
                print(f"Could not rename {src_path}: {e}")
print("✅ Renaming complete.")

# ===============================================================
# ===  STEP 2: IMAGE PREPROCESSING (Noise, CLAHE, Resize) ======
# ===============================================================
def preprocess_image(src_path, dst_path, img_size=IMG_SIZE):
    img = cv2.imread(src_path)
    if img is None:
        print(f"⚠️ Warning: Could not read {src_path}")
        return
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.medianBlur(img, 3)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    cv2.imwrite(dst_path, (img * 255).astype(np.uint8))

print("🔹 Preprocessing images (recursive) into cleaned directory...")
count = 0
for root, dirs, files in os.walk(RAW_ROOT):
    # skip the cleaned output folder if it's inside RAW_ROOT
    if os.path.commonpath([os.path.abspath(root), os.path.abspath(CLEAN_ROOT)]) == os.path.abspath(CLEAN_ROOT):
        continue
    # infer class name from the immediate folder name (relative to RAW_ROOT)
    rel = os.path.relpath(root, RAW_ROOT)
    parts = rel.split(os.sep)
    # If files are directly in RAW_ROOT, place them in an "unknown" class folder
    class_name = parts[0] if parts and parts[0] != '.' and parts[0] != os.curdir and parts[0] != '..' else 'unknown'
    for filename in files:
        if filename.startswith('.'):
            continue
        src_path = os.path.join(root, filename)
        # ensure class folder exists in CLEAN_ROOT
        dst_dir = os.path.join(CLEAN_ROOT, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, filename if '.' in filename else filename + '.jpg')
        preprocess_image(src_path, dst_path)
        count += 1
print(f"✅ Preprocessing complete. Total files processed: {count}")

# ===============================================================
# ===  STEP 3: DATA GENERATORS USING DATAFRAME  =================
# ===============================================================
print("🔹 Setting up data generators from cleaned directory (flow_from_directory)...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    CLEAN_ROOT,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    CLEAN_ROOT,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"Train samples: {getattr(train_gen, 'n', 0)}, Validation samples: {getattr(val_gen, 'n', 0)}")

# ===============================================================
# ===  STEP 4: BUILD RESNET50 MODEL  ============================
# ===============================================================
print("🔹 Building ResNet50 model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-10]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
preds = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================================================
# ===  STEP 5: CALLBACKS & TRAINING  ============================
# ===============================================================
callbacks = [
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print("🔹 Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ===============================================================
# ===  STEP 6: VISUALIZE TRAINING  ==============================
# ===============================================================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')
plt.show()

# ===============================================================
# ===  STEP 7: SAVE FINAL MODEL  ================================
# ===============================================================
model.save("oral_cancer_model.h5")
print("✅ Model saved successfully!")
