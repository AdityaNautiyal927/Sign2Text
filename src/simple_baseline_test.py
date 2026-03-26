import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "combined"
MODEL_DIR = ROOT_DIR / "data" / "models"

print("🧪 Testing with SUPER SIMPLE baseline model...")
print("This will help identify if the issue is with data or model complexity")

IMG_SIZE = (32, 32)  
BATCH_SIZE = 32
EPOCHS = 10

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.3,  
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.3,
    subset="validation", 
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"📊 Found {num_classes} classes: {class_names[:5]}..." if len(class_names) > 5 else class_names)


def quick_data_check():
    print("\n🔍 Quick Data Analysis:")
    

    for images, labels in train_ds.take(1):
        print(f"  Batch shape: {images.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Image range: [{images.numpy().min():.1f}, {images.numpy().max():.1f}]")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label range: [{labels.numpy().min()}, {labels.numpy().max()}]")
        print(f"  Unique labels in batch: {np.unique(labels.numpy())}")
        break

quick_data_check()

def create_ultra_simple_model():
    """The simplest possible model that should still work."""
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_ultra_simple_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n🏗️  Ultra-simple model created:")
model.summary()


print(f"\n🚀 Testing training for {EPOCHS} epochs...")
print("If this fails, the issue is with your data, not model complexity!")

try:
    history = model.fit(
        train_ds,
        validation_data=val_ds, 
        epochs=EPOCHS,
        verbose=1
    )
    
    final_acc = history.history['val_accuracy'][-1]
    print(f"\n✅ Simple model achieved {final_acc:.1%} validation accuracy")
    
    if final_acc > 0.2:
        print("🎉 Your data is working! The issue was model complexity.")
        print("💡 You can now try the improved complex model.")
    else:
        print("❌ Even simple model failed. Issue is likely with your data.")
        print("💡 Check your dataset preparation and image quality.")
        
    # Save for testing
    model.save(MODEL_DIR / "simple_baseline.keras")
    joblib.dump(class_names, MODEL_DIR / "simple_class_names.joblib")
    print(f"💾 Simple model saved for testing")
    
except Exception as e:
    print(f"❌ Simple training failed: {e}")
    print("💡 This confirms there's a fundamental issue with your data setup.")


def visualize_samples():
    """Show some sample images to verify they look correct."""
    try:
        plt.figure(figsize=(10, 6))
        
        for images, labels in train_ds.take(1):
            for i in range(min(8, len(images))):
                plt.subplot(2, 4, i + 1)
                img = images[i].numpy().astype("uint8")
                label = class_names[labels[i].numpy()]
                plt.imshow(img)
                plt.title(f"{label}")
                plt.axis('off')
            break
        
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "sample_images.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("📸 Sample images saved and displayed")
        
    except Exception as e:
        print(f"⚠️  Could not visualize samples: {e}")

print("\n🖼️  Attempting to show sample images...")
visualize_samples()

print("\n🔬 Baseline test complete!")
print("If the simple model works, your data is fine and you just need a better architecture.")
print("If the simple model fails, focus on fixing your data preparation.")