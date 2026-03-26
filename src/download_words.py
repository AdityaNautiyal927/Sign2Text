import os
import subprocess
import zipfile
import requests
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
WORDS_DIR = DATA_DIR / "asl_words"

DATA_DIR.mkdir(parents=True, exist_ok=True)

def check_kaggle_cli():
    """Check if Kaggle CLI is available."""
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def install_kaggle_cli():
    """Install Kaggle CLI."""
    print("Installing Kaggle CLI...")
    try:
        subprocess.run(['pip', 'install', 'kaggle'], check=True)
        print("✓ Kaggle CLI installed")
        return True
    except:
        print("✗ Failed to install Kaggle CLI")
        return False

def setup_kaggle_auth():
    """Guide user through Kaggle authentication setup."""
    print("\n" + "="*50)
    print("KAGGLE AUTHENTICATION SETUP")
    print("="*50)
    print("You need a Kaggle API token to download datasets.")
    print()
    print("Steps:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Save the downloaded kaggle.json file")
    print()
    
    possible_locations = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(f"C:/Users/{os.getenv('USERNAME')}/.kaggle/kaggle.json") if os.name == 'nt' else None
    ]
    
    for location in possible_locations:
        if location and location.exists():
            print(f"✓ Found kaggle.json at: {location}")
            return True
    
    print("5. Place kaggle.json in one of these locations:")
    for location in possible_locations:
        if location:
            print(f"   {location}")
    
    input("\nPress Enter after you've set up kaggle.json...")

    for location in possible_locations:
        if location and location.exists():
            try:
                os.chmod(location, 0o600)
                print("✓ Kaggle authentication configured")
                return True
            except:
                pass
    
    print("✗ kaggle.json not found. Please set it up first.")
    return False

def download_asl_words_datasets():
    """Download popular ASL words datasets."""
    datasets = [
        {
            'name': 'ayuraj/asl-dataset',
            'description': 'ASL Dataset with words and phrases',
            'folder': 'asl-dataset'
        },
        {
            'name': 'danrasband/asl-alphabet-test', 
            'description': 'ASL Alphabet Test (includes some words)',
            'folder': 'asl-alphabet-test'
        }
    ]
    
    downloaded = 0
    
    for dataset in datasets:
        print(f"\nDownloading: {dataset['description']}")
        print(f"Dataset: {dataset['name']}")
        
        try:
           
            temp_dir = WORDS_DIR / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            
            result = subprocess.run([
                'kaggle', 'datasets', 'download', 
                dataset['name'],
                '-p', str(temp_dir),
                '--unzip'
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print(f"✓ Downloaded {dataset['name']}")
                
                
                dataset_dir = WORDS_DIR / dataset['folder']
                dataset_dir.mkdir(exist_ok=True)
                
                
                for item in temp_dir.iterdir():
                    dest = dataset_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                
                downloaded += 1
            else:
                print(f"✗ Failed to download {dataset['name']}")
                print(f"Error: {result.stderr}")
                
        except Exception as e:
            print(f"✗ Error downloading {dataset['name']}: {e}")
        
        finally:
            
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    return downloaded

def find_word_folders():
    """Find and organize word folders from downloaded datasets."""
    word_folders = []

    if WORDS_DIR.exists():
        for root, dirs, files in os.walk(WORDS_DIR):
            for dir_name in dirs:
                if dir_name.lower() not in ['images', 'train', 'test', 'val', 'temp', '__pycache__']:
                    
                    if len(dir_name) > 1 and any(c.isalpha() for c in dir_name):
                        dir_path = Path(root) / dir_name
                        
                        image_files = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpeg'))
                        if image_files:
                            word_folders.append((dir_name.lower().strip(), dir_path, len(image_files)))
    
    return word_folders

def organize_word_data():
    """Organize found word data into clean structure."""
    word_folders = find_word_folders()
    
    if not word_folders:
        print("No word folders found in downloaded data")
        return 0
    
    print(f"\nFound {len(word_folders)} potential word classes:")

    priority_words = {
        'hello', 'thanks', 'yes', 'no', 'please', 'sorry', 'you', 'me', 'love',
        'help', 'stop', 'go', 'good', 'bad', 'more', 'finish', 'want', 'need'
    }
    
    organized_dir = WORDS_DIR / "organized"
    organized_dir.mkdir(exist_ok=True)
    
    organized_count = 0
    
    for word, source_path, image_count in word_folders:
        clean_word = word.lower().strip()
        
        
        if clean_word in {'del', 'nothing', 'space', 'blank'}:
            continue
        
        if not priority_words or clean_word in priority_words:
            print(f"  Organizing: {clean_word} ({image_count} images)")
            
            dest_dir = organized_dir / clean_word
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            
            shutil.copytree(source_path, dest_dir)
            organized_count += 1
        else:
            print(f"  Skipping: {clean_word} (not in priority list)")
    
    print(f"\n✓ Organized {organized_count} word classes in: {organized_dir}")
    return organized_count

def create_updated_prepare_script():
    """Update prepare_combined.py with correct words path."""
    prepare_script = BASE_DIR / "prepare_combined.py"
    
    if prepare_script.exists():
        with open(prepare_script, 'r') as f:
            content = f.read()
        
        organized_words_path = str(WORDS_DIR / "organized").replace('\\', '\\\\')
        
        new_content = content.replace(
            'words_path = r"C:\\Users\\SUMAN\\Desktop\\sign2text\\data\\raw\\asl_words\\images\\train"',
            f'words_path = r"{organized_words_path}"'
        )
        
        with open(prepare_script, 'w') as f:
            f.write(new_content)
        
        print(f"✓ Updated prepare_combined.py with new words path")
    else:
        print("prepare_combined.py not found - you'll need to update the words_path manually")

def main():
    print("="*60)
    print("ASL WORDS DATASET DOWNLOADER")
    print("="*60)
    print("This will supplement your existing alphabet dataset with words")
    print()
    
    # Step 1: Check Kaggle CLI
    if not check_kaggle_cli():
        print("Kaggle CLI not found. Installing...")
        if not install_kaggle_cli():
            print("Failed to install Kaggle CLI")
            print("Please install manually: pip install kaggle")
            return
    
    # Step 2: Setup authentication
    if not setup_kaggle_auth():
        print("Kaggle authentication not set up")
        return
    
    # Step 3: Download datasets
    print(f"\nDownloading ASL words datasets to: {WORDS_DIR}")
    downloaded = download_asl_words_datasets()
    
    if downloaded == 0:
        print("No datasets downloaded successfully")
        return
    
    print(f"\n✓ Downloaded {downloaded} datasets")
    
    # Step 4: Organize word data
    print("\nOrganizing word data...")
    organized = organize_word_data()
    
    if organized == 0:
        print("No word data organized")
        return
    
    # Step 5: Update prepare script
    create_updated_prepare_script()
    
    print(f"\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Words downloaded: {organized} classes")
    print(f"Location: {WORDS_DIR / 'organized'}")
    print(f"Project root: {BASE_DIR}")
    print()
    print("NEXT STEPS:")
    print("1. Run: python prepare_combined.py")
    print("   (It will now process both your letters AND the downloaded words)")
    print("2. Run: python src/train_combined.py")
    print("3. Run: python src/realtime_combined.py")
    print()
    print("Your model will now recognize both letters A-Z AND common words!")
    
    print(f"\nEXPECTED DATA STRUCTURE:")
    print(f"  {BASE_DIR}")
    print(f"  ├── data/")
    print(f"  │   ├── raw/")
    print(f"  │   │   ├── asl_alphabet/")
    print(f"  │   │   └── asl_words/organized/")
    print(f"  │   └── combined/ (created by prepare_combined.py)")
    print(f"  └── src/")
    print(f"      ├── train_combined.py")
    print(f"      └── realtime_combined.py")

if __name__ == "__main__":
    main()