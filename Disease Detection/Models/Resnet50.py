import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Train the ResNet50 disease classifier.")
parser.add_argument("--smoke-test", action="store_true", help="Run one small batch through training, validation, and testing.")
parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "Dataset" / "Disease Detection Dataset", help="Directory containing train, val, and test splits")
parser.add_argument("--epochs", type=int, help="Number of training epochs (default: 50)")
parser.add_argument("--full-epoch", action="store_true", help="Use every training and validation batch each epoch")
args = parser.parse_args()

dataset_dir = args.dataset_dir
dir_train = dataset_dir / "train"
dir_val = dataset_dir / "val"
dir_test = dataset_dir / "test"
batch_size = 2 if args.smoke_test else 16
epochs = 1 if args.smoke_test else (args.epochs or 50)

train_gen = ImageDataGenerator(rescale = 1./255)
train_data = train_gen.flow_from_directory(dir_train, batch_size=batch_size, class_mode='categorical', target_size=(224, 224))

val_gen = ImageDataGenerator(rescale = 1./255)
val_data = val_gen.flow_from_directory(dir_val, batch_size=batch_size, class_mode='categorical', target_size=(224, 224))

test_gen = ImageDataGenerator(rescale = 1./255)
test_data = test_gen.flow_from_directory(dir_test, batch_size=batch_size, class_mode='categorical', target_size=(224, 224), shuffle=False)

resnet = ResNet50(input_shape=[224, 224] + [3], weights=None if args.smoke_test else 'imagenet', include_top=False)

for layer in resnet.layers:
    layer.trainable = False
    
x = Flatten()(resnet.output)
out = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=out)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

history = model.fit(
  train_data,
  validation_data=val_data,
  epochs=epochs,
  steps_per_epoch=1 if args.smoke_test else (None if args.full_epoch else 200),
  validation_steps=1 if args.smoke_test else None)

test_loss, test_accuracy = model.evaluate(test_data, steps=1 if args.smoke_test else None)
print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}")
if not args.smoke_test:
    model.save(Path(__file__).with_name("resnet50_model.keras"))

plt.plot(history.history["accuracy"], marker="o", label="Training accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plot_name = "resnet50_smoke_test.png" if args.smoke_test else "resnet50_training.png"
plt.savefig(Path(__file__).with_name(plot_name), dpi=120, bbox_inches="tight")