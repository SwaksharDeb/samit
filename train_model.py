#!/usr/bin/env python3
"""
TensorFlow code to load the Stagnant Water and Wet Surface Dataset
and create a MobileNetV1 model that matches the model.tflite architecture.

The model is trained for binary classification (stagnant water detection):
- Class 0: no_water (no stagnant water detected)
- Class 1: stagnant_water (stagnant water detected)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import glob

# Configuration
DATASET_DIR = "Stagnant water and Wet surface Dataset"
IMG_SIZE = 96  # Input size
IMG_CHANNELS = 1  # Grayscale
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2
ALPHA = 0.25  # Width multiplier for MobileNetV1 (smaller = faster/smaller model)


def load_dataset(dataset_dir, img_size=96, grayscale=True):
    """
    Load images and labels from the dataset.

    The annotation files are in YOLO format: class_id x_center y_center width height
    We extract only the class_id for classification (ignoring bounding boxes).
    If an image has multiple annotations, we use the first class or majority class.

    Args:
        dataset_dir: Path to the dataset directory
        img_size: Target image size (square)
        grayscale: Whether to load as grayscale (1 channel)

    Returns:
        images: numpy array of shape (N, img_size, img_size, 1) if grayscale
        labels: numpy array of shape (N,)
    """
    images = []
    labels = []

    # Get all jpeg files
    image_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))
    print(f"Found {len(image_files)} images")

    for img_path in image_files:
        # Get corresponding annotation file
        txt_path = img_path.replace('.jpeg', '.txt')

        if not os.path.exists(txt_path):
            print(f"Warning: No annotation for {img_path}")
            continue

        # Read annotation file
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if not lines or all(line.strip() == '' for line in lines):
            print(f"Warning: Empty annotation for {img_path}")
            continue

        # Parse annotations - get class labels
        # For classification, we use majority class or first class
        class_ids = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_ids.append(int(parts[0]))

        if not class_ids:
            continue

        # Use majority class (most common class in the image)
        label = max(set(class_ids), key=class_ids.count)

        # Load and preprocess image
        try:
            color_mode = 'grayscale' if grayscale else 'rgb'
            img = load_img(img_path, target_size=(img_size, img_size), color_mode=color_mode)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images.shape}")
    print(f"Class distribution: no_water={np.sum(labels==0)}, stagnant_water={np.sum(labels==1)}")

    return images, labels


def preprocess_for_mobilenet(images):
    """
    Preprocess images for MobileNetV1.
    MobileNet expects input in range [-1, 1].
    """
    # Normalize from [0, 255] to [-1, 1]
    return (images / 127.5) - 1.0


def _depthwise_conv_block(x, pointwise_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    """
    Depthwise separable convolution block (MobileNet style).
    """
    pointwise_filters = int(pointwise_filters * alpha)

    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        (3, 3),
        padding='same',
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name=f'conv_dw_{block_id}'
    )(x)
    x = layers.BatchNormalization(name=f'conv_dw_{block_id}_bn')(x)
    x = layers.ReLU(6., name=f'conv_dw_{block_id}_relu')(x)

    # Pointwise convolution
    x = layers.Conv2D(
        pointwise_filters,
        (1, 1),
        padding='same',
        use_bias=False,
        strides=(1, 1),
        name=f'conv_pw_{block_id}'
    )(x)
    x = layers.BatchNormalization(name=f'conv_pw_{block_id}_bn')(x)
    x = layers.ReLU(6., name=f'conv_pw_{block_id}_relu')(x)

    return x


def create_mobilenet_model(input_shape=(96, 96, 1), num_classes=2, alpha=0.25):
    """
    Create a MobileNetV1-style model for binary classification with grayscale input.

    This creates a lightweight CNN similar to MobileNetV1 but supports
    single-channel (grayscale) input.

    Args:
        input_shape: Input image shape (H, W, C) - supports (96, 96, 1)
        num_classes: Number of output classes (2 for binary)
        alpha: Width multiplier (0.25, 0.5, 0.75, or 1.0)

    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # Initial convolution
    first_filters = int(32 * alpha)
    x = layers.Conv2D(
        first_filters,
        (3, 3),
        padding='same',
        use_bias=False,
        strides=(2, 2),
        name='conv1'
    )(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)

    # Depthwise separable convolution blocks (MobileNetV1 architecture)
    x = _depthwise_conv_block(x, 64, alpha, strides=(1, 1), block_id=1)
    x = _depthwise_conv_block(x, 128, alpha, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, strides=(1, 1), block_id=3)
    x = _depthwise_conv_block(x, 256, alpha, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, strides=(1, 1), block_id=5)
    x = _depthwise_conv_block(x, 512, alpha, strides=(2, 2), block_id=6)

    # 5 blocks with 512 filters
    for i in range(7, 12):
        x = _depthwise_conv_block(x, 512, alpha, strides=(1, 1), block_id=i)

    x = _depthwise_conv_block(x, 1024, alpha, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, strides=(1, 1), block_id=13)

    # Global average pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Classification head - single output with sigmoid for binary classification
    outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='mobilenet_v1_classifier')

    return model


def create_simple_cnn(input_shape=(96, 96, 1), num_classes=2):
    """
    Alternative: Create a simpler CNN for binary classification.
    Use this for faster training or if the MobileNet-style model is too large.
    """
    inputs = layers.Input(shape=input_shape, name='input')

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='simple_cnn_binary')

    return model


def convert_to_tflite(model, representative_dataset=None, quantize=True, output_path='model_new.tflite'):
    """
    Convert Keras model to TensorFlow Lite format.

    Args:
        model: Keras model
        representative_dataset: Generator function for calibration data (for quantization)
        quantize: Whether to apply int8 quantization
        output_path: Path to save the TFLite model

    Returns:
        Path to saved TFLite model
    """
    # Create a concrete function with fixed batch size of 1 for embedded deployment
    input_shape = model.input_shape[1:]  # Get (96, 96, 1) without batch dimension
    fixed_input = tf.keras.Input(shape=input_shape, batch_size=1, name='input')
    fixed_output = model(fixed_input)
    fixed_model = tf.keras.Model(fixed_input, fixed_output)

    converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)

    if quantize:
        # Full integer quantization (matches model.tflite)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
            # Full integer quantization
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024:.1f} KB")

    return output_path


def create_representative_dataset(images, num_samples=100):
    """
    Create a representative dataset generator for TFLite quantization.
    """
    def representative_data_gen():
        for i in range(min(num_samples, len(images))):
            sample = images[i:i+1]
            # Ensure proper type
            yield [sample.astype(np.float32)]

    return representative_data_gen


def main():
    print("=" * 60)
    print("Stagnant Water Detection (Binary Classification)")
    print("=" * 60)
    print(f"Input shape: ({IMG_SIZE}, {IMG_SIZE}, {IMG_CHANNELS})")

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    images, labels = load_dataset(DATASET_DIR, IMG_SIZE, grayscale=(IMG_CHANNELS == 1))

    # 2. Preprocess images
    print("\n[2/5] Preprocessing images...")
    images = preprocess_for_mobilenet(images)

    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Labels are already binary (0 or 1), no one-hot encoding needed for sigmoid output

    # 3. Create model
    print("\n[3/5] Creating MobileNetV1-style model...")
    model = create_mobilenet_model(
        input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS),
        num_classes=NUM_CLASSES,
        alpha=ALPHA
    )
    model.summary()

    # Compile model with binary crossentropy for binary classification
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train model
    print("\n[4/5] Training model...")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nValidation accuracy: {val_acc:.4f}")

    # Save Keras model
    model.save('model_keras.h5')
    print("Keras model saved to model_keras.h5")

    # 5. Convert to TFLite
    print("\n[5/5] Converting to TFLite...")

    # Create representative dataset for quantization
    rep_dataset = create_representative_dataset(X_train, num_samples=100)

    # Convert with quantization
    convert_to_tflite(
        model,
        representative_dataset=rep_dataset,
        quantize=True,
        output_path='model_new.tflite'
    )

    # Also save non-quantized version for comparison
    convert_to_tflite(
        model,
        quantize=False,
        output_path='model_float.tflite'
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - model_keras.h5      : Keras model")
    print("  - model_new.tflite    : Quantized TFLite model")
    print("  - model_float.tflite  : Float TFLite model")


if __name__ == '__main__':
    main()
