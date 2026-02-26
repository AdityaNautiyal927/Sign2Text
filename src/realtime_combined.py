import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import joblib
from collections import deque, Counter
import time

# Simple, reliable configuration
IMG_SIZE = (64, 64)  # Match training exactly
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "data" / "models"

# Much more lenient settings
PREDICTION_WINDOW = 8
MIN_CONFIDENCE = 0.15    # Very low - your model gives low confidences
STABLE_FRAMES_NEEDED = 5  # Frames to consider prediction stable
BOOST_FACTOR = 3.0       # Aggressive confidence boost for display

class SimpleASLRecognizer:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.prediction_history = deque(maxlen=PREDICTION_WINDOW)
        self.sentence = ""
        self.fps = 0
        self.show_debug = True
        
        # Load model and classes
        self.load_model_and_classes()
        
        print("Simple ASL Recognizer - Focus on Basic Functionality")
        print(f"Loaded: {len(self.class_names)} classes")
        print("Strategy: Minimal preprocessing, maximum reliability")

    def load_model_and_classes(self):
        """Load model and classes with simple fallback."""
        # Find model file
        model_paths = [
            MODEL_DIR / "best_model.keras",
            MODEL_DIR / "sign_model.keras",
            "best_model.keras",
            "sign_model.keras"
        ]
        
        self.model_path = None
        for path in model_paths:
            if Path(path).exists():
                self.model_path = str(path)
                break
        
        if not self.model_path:
            raise FileNotFoundError("No model found. Run train_combined.py first.")
        
        # Find classes
        class_paths = [
            MODEL_DIR / "class_names.joblib",
            "class_names.joblib"
        ]
        
        self.classes_path = None
        for path in class_paths:
            if Path(path).exists():
                self.classes_path = str(path)
                break
        
        if not self.classes_path:
            raise FileNotFoundError("No classes found. Run train_combined.py first.")
        
        # Load
        self.model = load_model(self.model_path)
        self.class_names = joblib.load(self.classes_path)
        
        print(f"Model: {self.model_path}")
        print(f"Classes: {self.class_names}")

    def simple_preprocess(self, roi):
        """Minimal preprocessing - just match training format exactly."""
        try:
            # Convert BGR to RGB
            if len(roi.shape) == 3:
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            
            # Resize exactly like training
            resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
            
            # Normalize exactly like training: [0,1] range
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            batch = np.expand_dims(normalized, axis=0)
            
            return batch
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def predict_sign(self, roi):
        """Simple, direct prediction."""
        processed = self.simple_preprocess(roi)
        if processed is None:
            return "ERROR", 0.0, []
        
        try:
            # Get raw predictions
            predictions = self.model.predict(processed, verbose=0)[0]
            
            # Get top 5 predictions
            top_indices = np.argsort(predictions)[-5:][::-1]
            top_predictions = []
            
            for idx in top_indices:
                if idx < len(self.class_names):
                    class_name = self.class_names[idx]
                    raw_confidence = float(predictions[idx])
                    # Boost confidence for display only
                    display_confidence = min(1.0, raw_confidence * BOOST_FACTOR)
                    top_predictions.append((class_name, display_confidence))
            
            if not top_predictions:
                return "UNKNOWN", 0.0, []
            
            best_class = top_predictions[0][0]
            best_confidence = top_predictions[0][1]
            
            return best_class, best_confidence, top_predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "ERROR", 0.0, []

    def get_stable_prediction(self):
        """Simple majority vote over recent predictions."""
        if len(self.prediction_history) < 3:
            return None, 0.0
        
        # Get recent predictions (exclude error states)
        valid_preds = [p for p in self.prediction_history if p not in ["ERROR", "UNKNOWN"]]
        
        if len(valid_preds) < 3:
            return None, 0.0
        
        # Simple majority vote
        pred_counts = Counter(valid_preds)
        most_common = pred_counts.most_common(1)[0]
        prediction = most_common[0]
        count = most_common[1]
        
        # Calculate stability (how consistent recent predictions are)
        stability = count / len(valid_preds)
        confidence = min(1.0, stability * 0.8 + 0.2)  # Base confidence from stability
        
        return prediction, confidence

    def draw_simple_ui(self, frame, roi_coords, prediction, confidence, top_predictions):
        """Simple, clear UI."""
        x1, y1, x2, y2 = roi_coords
        
        # Always draw green box (system active)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Simple status panel
        panel_height = 150
        cv2.rectangle(frame, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)
        
        y_pos = 25
        
        # Current prediction
        pred_text = f"CURRENT: {prediction} ({confidence:.3f})"
        cv2.putText(frame, pred_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        
        # Stable prediction
        stable_pred, stable_conf = self.get_stable_prediction()
        if stable_pred:
            stable_text = f"STABLE: {stable_pred} ({stable_conf:.3f})"
            color = (0, 255, 255) if stable_conf > 0.6 else (128, 128, 128)
            cv2.putText(frame, stable_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "STABLE: Building...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        y_pos += 30
        
        # Sentence
        sentence_display = self.sentence if len(self.sentence) < 60 else "..." + self.sentence[-57:]
        sentence_text = f"SENTENCE: {sentence_display}"
        cv2.putText(frame, sentence_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 25
        
        # FPS and history
        history_text = f"FPS: {self.fps:.1f} | History: {len(self.prediction_history)}/{PREDICTION_WINDOW}"
        cv2.putText(frame, history_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Debug: Show top predictions
        if self.show_debug and top_predictions:
            debug_x = frame.shape[1] - 250
            debug_y = 25
            
            cv2.putText(frame, "TOP PREDICTIONS:", (debug_x, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            debug_y += 20
            
            for i, (cls, conf) in enumerate(top_predictions[:4]):
                pred_text = f"{i+1}. {cls}: {conf:.3f}"
                cv2.putText(frame, pred_text, (debug_x, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                debug_y += 18
        
        # Show ROI in corner for debugging
        if self.show_debug:
            try:
                roi_display = frame[y1:y2, x1:x2]
                if roi_display.size > 0:
                    roi_small = cv2.resize(roi_display, (100, 100))
                    frame[frame.shape[0]-110:frame.shape[0]-10, 10:110] = roi_small
                    cv2.putText(frame, "ROI", (15, frame.shape[0]-115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            except:
                pass
        
        # Help text
        help_y = frame.shape[0] - 60
        cv2.rectangle(frame, (0, help_y-5), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        
        help_texts = [
            "CONTROLS: Q=Quit | C=Clear | SPACE=Add space | A=Add letter | R=Reset | T=Diagnostic | L=Letter test | P=Patterns | S=Stats",
            "OPTIMIZATION: Focus on signs that work well | Clean background | Good lighting | Hold steady 5+ seconds",
        ]
        
        for i, text in enumerate(help_texts):
            cv2.putText(frame, text, (10, help_y + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    def run(self):
        """Simple main loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("\nSimple ASL Recognition Started")
        print("Focus: Get basic predictions working reliably")
        print("Put your hand in the green box and make clear signs")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror view
                
                # Simple center ROI extraction
                height, width = frame.shape[:2]
                roi_size = min(height, width) // 2
                center_x, center_y = width // 2, height // 2
                
                x1 = center_x - roi_size // 2
                y1 = center_y - roi_size // 2
                x2 = center_x + roi_size // 2
                y2 = center_y + roi_size // 2
                
                roi = frame[y1:y2, x1:x2]
                roi_coords = (x1, y1, x2, y2)
                
                # Always predict - no hand detection filtering
                prediction, confidence, top_predictions = self.predict_sign(roi)
                
                # Add to history
                self.prediction_history.append(prediction)
                
                # Update FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                
                # Draw UI
                self.draw_simple_ui(frame, roi_coords, prediction, confidence, top_predictions)
                
                cv2.imshow('Simple ASL Recognition - Basic Function Test', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c'):
                    self.sentence = ""
                    print("Sentence cleared")
                elif key == ord(' '):
                    self.sentence += " "
                    print(f"Added space: '{self.sentence}'")
                elif key == ord('a'):
                    stable_pred, stable_conf = self.get_stable_prediction()
                    if stable_pred and stable_conf > 0.5:
                        self.sentence += stable_pred
                        print(f"Added '{stable_pred}': '{self.sentence}'")
                        self.prediction_history.clear()  # Reset for next sign
                    else:
                        print(f"Not stable enough: {stable_pred} ({stable_conf:.3f})")
                elif key == ord('r'):
                    self.prediction_history.clear()
                    print("History cleared")
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('t'):
                    # Test mode - show raw model outputs
                    if top_predictions:
                        print("\nRAW MODEL OUTPUT:")
                        for i, (cls, conf) in enumerate(top_predictions):
                            print(f"  {i+1}. {cls}: {conf/BOOST_FACTOR:.4f} (raw) -> {conf:.4f} (boosted)")
                        print("If predictions seem random, there's likely a domain shift issue.")
        
        except KeyboardInterrupt:
            print("\nStopped")
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    try:
        recognizer = SimpleASLRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()





