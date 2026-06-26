import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# Paths
DATASET_PATH = 'lung_colon_image_set/lung_image_sets/'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
# Augmentation for training data
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=15,
width_shift_range=0.1,
height_shift_range=0.1,
horizontal_flip=True,
zoom_range=0.1,
validation_split=0.15
)
train_gen = train_datagen.flow_from_directory(
DATASET_PATH, target_size=IMG_SIZE,
batch_size=BATCH_SIZE, class_mode='categorical',
subset='training', seed=42
)
val_gen = train_datagen.flow_from_directory(
DATASET_PATH, target_size=IMG_SIZE,
batch_size=BATCH_SIZE, class_mode='categorical',
subset='validation', seed=42
)
def build_cnn(input_shape=(224,224,3), num_classes=3):
model = models.Sequential([
# Block 1
layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
layers.MaxPooling2D((2,2)),
layers.BatchNormalization(),
# Block 2
layers.Conv2D(64, (3,3), activation='relu'),
layers.MaxPooling2D((2,2)),
  layers.BatchNormalization(),
# Block 3
layers.Conv2D(128, (3,3), activation='relu'),
layers.MaxPooling2D((2,2)),
# Dense layers
layers.Flatten(),
layers.Dense(256, activation='relu'),
layers.Dropout(0.5),
layers.Dense(num_classes, activation='softmax')
])
model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)
return model
model = build_cnn()
model.summary()
# Callbacks
callbacks = [
keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]
history = model.fit(
train_gen, epochs=EPOCHS,
validation_data=val_gen,
callbacks=callbacks
)
# Accuracy / Loss curves
fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy'); axes[0].legend()
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss'); axes[1].legend()
plt.show()
# Confusion Matrix
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_gen.classes
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=val_gen.class_indices.keys(),
yticklabels=val_gen.class_indices.keys())
plt.title('Confusion Matrix'); plt.show()
# Classification Report
print(classification_report(y_true, y_pred_classes,
target_names=['lung_aca', 'lung_n', 'lung_scc']))
