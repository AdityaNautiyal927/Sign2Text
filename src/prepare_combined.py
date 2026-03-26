import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np
from collections import Counter
import json
import random

alphabet_path = r"C:\Users\SUMAN\Desktop\sign2text\data\raw\asl_alphabet\asl_alphabet_train\asl_alphabet_train"
words_path = r"C:\Users\SUMAN\Desktop\sign2text\data\raw\asl_words\images\train"
combined_path = r"C:\Users\SUMAN\Desktop\sign2text\data\combined"

TARGET_SIZE = (64, 64)  
MIN_IMAGES_PER_CLASS = 50
MAX_IMAGES_PER_CLASS = 1500  
PROBLEMATIC_CLASSES = {'del', 'nothing', 'space', 'blank'} 

DESIRED_WORDS = {
    'hello', 'thanks', 'yes', 'no', 'please', 'sorry', 'you', 'me', 'love', 
    'help', 'stop', 'go', 'good', 'bad', 'more', 'finish', 'want', 'need',
    'water', 'food', 'work', 'home', 'family', 'friend', 'happy', 'sad'
}

def is_valid_image(file_path):
    """Minimal validation - just check if image can be read."""
    try:
        img = cv2.imread(str(file_path))
        if img is None:
            return False

        h, w = img.shape[:2]
        if h < 20 or w < 20:  
            return False
            
        return True
    except Exception:
        return False

def minimal_preprocess_and_save(src_path, dst_path, target_size=TARGET_SIZE):
    """MINIMAL preprocessing to match real-time conditions."""
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return False

        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        success = cv2.imwrite(str(dst_path), img_resized, 
                             [cv2.IMWRITE_PNG_COMPRESSION, 1])  
        return success
    except Exception as e:
        print(f"Error preprocessing {src_path}: {e}")
        return False

def copy_and_validate_file(src, dst):
    """Copy with minimal preprocessing only if valid."""
    try:
        if is_valid_image(src):
            return minimal_preprocess_and_save(src, dst)
        else:
            return False
    except Exception as e:
        print(f"Error copying {src}: {e}")
        return False

def process_alphabet_dataset(src_root, dst_root, max_workers=4):
    """Process ASL Alphabet dataset (A-Z letters)."""
    print(f"Processing ASL Alphabet from: {src_root}")
    
    class_stats = {}
    successful_copies = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for class_name in os.listdir(src_root):
            src_class = os.path.join(src_root, class_name)
            if not os.path.isdir(src_class):
                continue

            if len(class_name) == 1 and class_name.isupper():
                print(f"  Processing letter: {class_name}")
            else:
                print(f"  Skipping non-letter class: {class_name}")
                continue
            
            dst_class = os.path.join(dst_root, class_name)
            os.makedirs(dst_class, exist_ok=True)

            image_files = []
            for file in os.listdir(src_class):
                src_file = os.path.join(src_class, file)
                if (os.path.isfile(src_file) and 
                    file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))):

                    base_name = f"{class_name}_{len(image_files):04d}"
                    dst_file = os.path.join(dst_class, f"{base_name}.png")
                    image_files.append((src_file, dst_file))

            if len(image_files) > MAX_IMAGES_PER_CLASS:
                print(f"    Limiting {class_name}: {len(image_files)} → {MAX_IMAGES_PER_CLASS}")
                random.shuffle(image_files)
                image_files = image_files[:MAX_IMAGES_PER_CLASS]
            
            class_stats[class_name] = len(image_files)

            for src_file, dst_file in image_files:
                futures.append(executor.submit(copy_and_validate_file, src_file, dst_file))

        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                successful_copies += 1
            
            if i % 200 == 0:
                print(f"    Progress: {i}/{len(futures)} ({successful_copies} successful)")
    
    print(f"  Alphabet processing complete: {successful_copies} images")
    return class_stats, successful_copies

def process_words_dataset(src_root, dst_root, max_workers=4):
    """Process ASL Words dataset - only include common/useful words."""
    print(f"Processing ASL Words from: {src_root}")
    
    class_stats = {}
    successful_copies = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for class_name in os.listdir(src_root):
            src_class = os.path.join(src_root, class_name)
            if not os.path.isdir(src_class):
                continue
            
            clean_class_name = class_name.lower().strip()
            
            if clean_class_name in PROBLEMATIC_CLASSES:
                print(f"  Skipping problematic class: {class_name}")
                continue
            
            if DESIRED_WORDS and clean_class_name not in DESIRED_WORDS:
                continue
            
            print(f"  Processing word: {clean_class_name}")
            dst_class = os.path.join(dst_root, clean_class_name)
            os.makedirs(dst_class, exist_ok=True)
            
            image_files = []
            for file in os.listdir(src_class):
                src_file = os.path.join(src_class, file)
                if (os.path.isfile(src_file) and 
                    file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))):
                    
                    base_name = f"{clean_class_name}_{len(image_files):04d}"
                    dst_file = os.path.join(dst_class, f"{base_name}.png")
                    image_files.append((src_file, dst_file))
            
            if len(image_files) < MIN_IMAGES_PER_CLASS:
                print(f"    WARNING: {clean_class_name} has only {len(image_files)} images (min: {MIN_IMAGES_PER_CLASS})")
            
            if len(image_files) > MAX_IMAGES_PER_CLASS:
                print(f"    Limiting {clean_class_name}: {len(image_files)} → {MAX_IMAGES_PER_CLASS}")
                random.shuffle(image_files)
                image_files = image_files[:MAX_IMAGES_PER_CLASS]
            
            class_stats[clean_class_name] = len(image_files)
            
            for src_file, dst_file in image_files:
                futures.append(executor.submit(copy_and_validate_file, src_file, dst_file))
        
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                successful_copies += 1
            
            if i % 200 == 0:
                print(f"    Progress: {i}/{len(futures)} ({successful_copies} successful)")
    
    print(f"  Words processing complete: {successful_copies} images")
    return class_stats, successful_copies

def analyze_combined_dataset(data_path):
    """Analyze the final combined dataset."""
    print(f"\nAnalyzing combined dataset: {data_path}")
    
    class_counts = {}
    letters = {}
    words = {}
    total_images = 0
    
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            class_counts[class_name] = count
            total_images += count
            
            if len(class_name) == 1 and class_name.isupper():
                letters[class_name] = count
            else:
                words[class_name] = count
    
    print(f"\n{'='*60}")
    print("COMBINED DATASET ANALYSIS")
    print('='*60)
    print(f"Total Classes: {len(class_counts)}")
    print(f"  - Letters (A-Z): {len(letters)}")
    print(f"  - Words: {len(words)}")
    print(f"Total Images: {total_images}")
    
    if letters:
        print(f"\nLETTERS ({len(letters)} classes):")
        missing_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - set(letters.keys())
        if missing_letters:
            print(f"  Missing letters: {sorted(missing_letters)}")
        
        letter_counts = list(letters.values())
        print(f"  Average per letter: {np.mean(letter_counts):.1f}")
        print(f"  Range: {min(letter_counts)} - {max(letter_counts)}")
     
    if words:
        print(f"\nWORDS ({len(words)} classes):")
        word_counts = list(words.values())
        print(f"  Average per word: {np.mean(word_counts):.1f}")
        print(f"  Range: {min(word_counts)} - {max(word_counts)}")
        print(f"  Words included: {', '.join(sorted(words.keys())[:10])}...")
    
    all_counts = list(class_counts.values())
    if all_counts:
        imbalance_ratio = max(all_counts) / min(all_counts)
        print(f"\nBALANCE ANALYSIS:")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
        if imbalance_ratio > 3:
            print(f"  Status: IMBALANCED - use class weights in training")
        else:
            print(f"  Status: REASONABLY BALANCED")

    print(f"\nTRAINING READINESS:")
    if len(class_counts) >= 2 and total_images >= 100:
        print(f"  Status: READY ✓")
    else:
        print(f"  Status: INSUFFICIENT DATA ✗")
    
    analysis_data = {
        'summary': {
            'total_classes': len(class_counts),
            'letters': len(letters),
            'words': len(words),
            'total_images': total_images,
            'imbalance_ratio': max(all_counts) / min(all_counts) if all_counts else 0
        },
        'letters': letters,
        'words': words,
        'all_classes': class_counts
    }
    
    analysis_path = os.path.join(data_path, "dataset_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"  Analysis saved: {analysis_path}")
    return class_counts

def main():
    print("=" * 80)
    print("ASL DATASET PREPARATION - FIXED FOR REAL-TIME COMPATIBILITY")
    print("=" * 80)
    print("Key fixes:")
    print("- Minimal preprocessing (no denoising/filtering)")
    print("- Correct target size (64x64)")
    print("- Include ASL words for vocabulary expansion")
    print("- Better class balancing")
    print("=" * 80)
    
    os.makedirs(combined_path, exist_ok=True)
    random.seed(42)  
    
    alphabet_stats = {}
    words_stats = {}
    
    if os.path.exists(alphabet_path):
        print("\n1. PROCESSING ASL ALPHABET DATASET")
        print("-" * 40)
        print(f"Path: {alphabet_path}")

        try:
            alphabet_contents = [d for d in os.listdir(alphabet_path) if os.path.isdir(os.path.join(alphabet_path, d))]
            print(f"Found {len(alphabet_contents)} directories: {alphabet_contents[:10]}{'...' if len(alphabet_contents) > 10 else ''}")
        except Exception as e:
            print(f"Error reading alphabet directory: {e}")
        
        alphabet_stats, alphabet_success = process_alphabet_dataset(alphabet_path, combined_path)
    else:
        print(f"\nWARNING: Alphabet path not found: {alphabet_path}")
        alphabet_success = 0
    
    if os.path.exists(words_path):
        print(f"\n2. PROCESSING ASL WORDS DATASET")
        print("-" * 40)
        print(f"Path: {words_path}")

        try:
            words_contents = [d for d in os.listdir(words_path) if os.path.isdir(os.path.join(words_path, d))]
            print(f"Found {len(words_contents)} directories: {words_contents[:10]}{'...' if len(words_contents) > 10 else ''}")
        except Exception as e:
            print(f"Error reading words directory: {e}")
        
        words_stats, words_success = process_words_dataset(words_path, combined_path)
    else:
        print(f"\nWARNING: Words path not found: {words_path}")
        print(f"Expected path: {words_path}")
        print("Check if:")
        print("1. The path exists")
        print("2. You have the ASL words dataset downloaded")
        print("3. The folder structure matches the expected path")
        words_success = 0

    print(f"\n3. FINAL ANALYSIS")
    print("-" * 40)
    final_stats = analyze_combined_dataset(combined_path)
    
    total_classes = len(final_stats)
    total_images = sum(final_stats.values())
    
    print(f"\n{'='*60}")
    print("PREPARATION COMPLETE!")
    print('='*60)
    print(f"Output directory: {combined_path}")
    print(f"Total classes: {total_classes}")
    print(f"Total images: {total_images}")
    print(f"Letters processed: {alphabet_success}")
    print(f"Words processed: {words_success}")
    
    if total_classes >= 2 and total_images >= 100:
        print(f"\nSTATUS: READY FOR TRAINING ✓")
        print("You can now run train_combined.py")
    else:
        print(f"\nSTATUS: INSUFFICIENT DATA ✗")
        print(f"Need at least 2 classes and 100 images")
    
    print(f"\nREAL-TIME COMPATIBILITY:")
    print(f"✓ Minimal preprocessing applied")
    print(f"✓ Target size matches training (64x64)")
    print(f"✓ No artificial denoising/filtering")
    print(f"✓ Should work better with webcam data")

if __name__ == "__main__":
    main()

