"""
train.py

Lightweight training runner that:
- uses preprocessed images in `cleaned/` (if present), otherwise triggers preprocessing
- builds a ResNet50-based classifier
- trains with checkpoints + early stopping + LR reduction
- saves final model in native .keras format

Usage:
	python train.py [--root PATH] [--dry]
"""
import os
import argparse
import shutil
from datetime import datetime

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_ROOT = os.path.join(SCRIPT_DIR, 'cleaned')


def ensure_cleaned(root_hint=None):
	# If cleaned exists, use it. Otherwise, try running dataset_preprocess.py if present.
	if os.path.isdir(CLEAN_ROOT) and os.path.exists(os.path.join(CLEAN_ROOT, 'manifest.csv')):
		return CLEAN_ROOT
	preprocess_script = os.path.join(SCRIPT_DIR, 'dataset_preprocess.py')
	if os.path.exists(preprocess_script):
		print('Running preprocessing script to create cleaned/ ...')
		os.system(f'"{preprocess_script}"')
		if os.path.isdir(CLEAN_ROOT):
			return CLEAN_ROOT
	raise RuntimeError('cleaned/ not found and preprocessing failed. Run dataset_preprocess.py first.')


def main(root_hint=None, dry=False):
	# Defer heavy imports until we know we will train
	cleaned = ensure_cleaned(root_hint)
	if dry:
		print('Dry run: cleaned/ is present at', cleaned)
		return

	import tensorflow as tf
	from tensorflow.keras.preprocessing.image import ImageDataGenerator
	from tensorflow.keras.applications import ResNet50
	from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
	from tensorflow.keras.models import Model
	from tensorflow.keras.optimizers import Adam
	from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

	datagen = ImageDataGenerator(
		rescale=1./255,
		rotation_range=25,
		horizontal_flip=True,
		brightness_range=[0.7, 1.3],
		validation_split=0.2
	)
	train_gen = datagen.flow_from_directory(
		cleaned,
		target_size=IMG_SIZE,
		batch_size=BATCH_SIZE,
		class_mode='categorical',
		subset='training',
		shuffle=True
	)
	val_gen = datagen.flow_from_directory(
		cleaned,
		target_size=IMG_SIZE,
		batch_size=BATCH_SIZE,
		class_mode='categorical',
		subset='validation',
		shuffle=False
	)
	print('Found classes:', train_gen.class_indices)

	base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
	for layer in base.layers[:-10]:
		layer.trainable = False
	x = base.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.4)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.3)(x)
	out = Dense(len(train_gen.class_indices), activation='softmax')(x)
	model = Model(inputs=base.input, outputs=out)
	model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	checkpoint = ModelCheckpoint(f'best_model_{timestamp}.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
	early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

	history = model.fit(
		train_gen,
		validation_data=val_gen,
		epochs=EPOCHS,
		callbacks=[checkpoint, early, reduce_lr]
	)

	# Save final model in native Keras format
	final_model_path = os.path.join(SCRIPT_DIR, f'oral_cancer_resnet50_model_{timestamp}.keras')
	model.save(final_model_path)
	print('Saved final model to', final_model_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default=None, help='Dataset root (optional)')
	parser.add_argument('--dry', action='store_true', help='Dry run (do not train)')
	args = parser.parse_args()
	main(root_hint=args.root, dry=args.dry)

