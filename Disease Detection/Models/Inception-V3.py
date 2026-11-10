import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

parser = argparse.ArgumentParser(description="Train the InceptionV3 disease classifier.")
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
train_data = train_gen.flow_from_directory(dir_train, batch_size=batch_size, class_mode='categorical', target_size=(299, 299))

val_gen = ImageDataGenerator(rescale = 1./255)
val_data = val_gen.flow_from_directory(dir_val, batch_size=batch_size, class_mode='categorical', target_size=(299, 299))

test_gen = ImageDataGenerator(rescale = 1./255)
test_data = test_gen.flow_from_directory(dir_test, batch_size=batch_size, class_mode='categorical', target_size=(299, 299), shuffle=False)

base_model = InceptionV3(input_shape = (299, 299, 3),
                                include_top = False,
                                weights=None if args.smoke_test else 'imagenet')
for layer in base_model.layers:
  layer.trainable = False
  
x = layers.Flatten()(base_model.output)
x= layers.Dense(64, activation='relu')(x)
x = layers.Dense(train_data.num_classes, activation='softmax')(x)

 
model = Model( base_model.input, x)

model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    steps_per_epoch=1 if args.smoke_test else (None if args.full_epoch else 100),
    validation_steps=1 if args.smoke_test else None,
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'r-o', label='Training accuracy')
plt.plot(epochs, val_acc, 'b-o', label='Validation accuracy')
plt.title('Training and validation accuracy')
 
plt.figure()
 
plt.plot(epochs, loss, 'r-o', label='Training Loss')
plt.plot(epochs, val_loss, 'b-o', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plot_name = "inception_v3_smoke_test.png" if args.smoke_test else "inception_v3_training.png"
plt.savefig(Path(__file__).with_name(plot_name), dpi=120, bbox_inches="tight")
plt.close("all")
test_loss, test_accuracy = model.evaluate(test_data, steps=1 if args.smoke_test else None)
print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}")
if not args.smoke_test:
    model.save(Path(__file__).with_name("inception_v3_model.keras"))