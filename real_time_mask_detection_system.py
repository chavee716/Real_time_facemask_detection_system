import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import datetime
import threading
import json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, accuracy_score
import seaborn as sns
import pandas as pd

# Configuration Settings
MODEL_PATH = "face_mask_model.h5"  # Path to save/load the trained model
EMAIL_COOLDOWN = 60  # Seconds between email alerts to prevent spam
CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence for mask detection
SCREENSHOT_INTERVAL = 5  # Time between automatic screenshots (seconds)

# Performance Monitoring Settings
PERFORMANCE_LOG_PATH = "performance_logs"
PERFORMANCE_TRACKING = True  # Enable/disable performance tracking
INFERENCE_SPEED_TRACKING = True  # Track model inference speed
VISUALIZATION_PATH = "visualizations"  # Path to save visualizations

# Email Configuration
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_SENDER = "sender@gmail.com"  # Update with your email
EMAIL_PASSWORD = "xxxxxxxxxxxxxx"    # Use app password for Gmail
EMAIL_RECEIVER = "receiver@gmail.com" # Update with receiver email
EMAIL_SUBJECT = "ALERT: Person Without Mask Detected"

# Initialize global variables
last_email_time = 0
email_lock = threading.Lock()
running = True  # Global flag to control the detection loop
performance_data = {
    "inference_times": [],
    "detection_counts": {"with_mask": 0, "without_mask": 0},
    "false_positives": 0,
    "false_negatives": 0
}

def build_model():
    """Build and compile the CNN model for mask detection"""
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        keras.layers.MaxPooling2D(2, 2),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        # Third convolutional block
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2, activation='softmax')  # 2 classes: with_mask and without_mask
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def process_dataset():
    """Process the dataset and convert to numpy arrays"""
    # Get file lists
    with_mask_path = 'data/with_mask/'
    without_mask_path = 'data/without_mask/'
    
    # Check if directories exist
    if not os.path.exists(with_mask_path) or not os.path.exists(without_mask_path):
        raise FileNotFoundError(f"Dataset directories not found. Please ensure they exist at {with_mask_path} and {without_mask_path}")
    
    with_mask_files = [f for f in os.listdir(with_mask_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    without_mask_files = [f for f in os.listdir(without_mask_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(with_mask_files) == 0 or len(without_mask_files) == 0:
        raise ValueError("No image files found in the dataset directories")
    
    print(f"Processing dataset: {len(with_mask_files)} with mask images, {len(without_mask_files)} without mask images")
    
    # Create labels
    with_mask_labels = [1] * len(with_mask_files)
    without_mask_labels = [0] * len(without_mask_files)
    
    # Combine labels
    labels = with_mask_labels + without_mask_labels
    
    # Process images
    data = []
    processed_count = 0
    total_images = len(with_mask_files) + len(without_mask_files)
    
    # Process with_mask images
    for img_file in with_mask_files:
        try:
            image = Image.open(os.path.join(with_mask_path, img_file))
            image = image.resize((128, 128))
            image = image.convert('RGB')
            image = np.array(image)
            data.append(image)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{total_images} images...")
                
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
    
    # Process without_mask images
    for img_file in without_mask_files:
        try:
            image = Image.open(os.path.join(without_mask_path, img_file))
            image = image.resize((128, 128))
            image = image.convert('RGB')
            image = np.array(image)
            data.append(image)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{total_images} images...")
                
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
    
    # Convert to numpy arrays
    X = np.array(data)
    Y = np.array(labels)
    
    print(f"Dataset processing complete. Shape: X={X.shape}, Y={Y.shape}")
    
    return X, Y

def plot_training_history(history):
    """Plot the training history"""
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_PATH}/training_history.png')
    print(f"Training history plot saved to {VISUALIZATION_PATH}/training_history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_PATH}/confusion_matrix.png')
    print(f"Confusion matrix saved to {VISUALIZATION_PATH}/confusion_matrix.png")
    plt.close()

def evaluate_model_performance(model, X_test, y_test):
    """Evaluate model performance with multiple metrics"""
    # Scale data
    X_test_scaled = X_test / 255.0
    
    # Get predictions
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Without Mask', 'With Mask']))
    
    # Create and log metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save metrics to log
    os.makedirs(PERFORMANCE_LOG_PATH, exist_ok=True)
    log_file = os.path.join(PERFORMANCE_LOG_PATH, "model_performance.json")
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            log_data.append(metrics)
        except:
            log_data = [metrics]
    else:
        log_data = [metrics]
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, ['Without Mask', 'With Mask'])
    
    # Return metrics
    return metrics

def load_face_detector():
    """Load face detector model"""
    prototxt_path = "deploy.prototxt"
    weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    # Check if files exist, if not download them
    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        download_face_detector()
    
    # Load face detection model
    face_detector = cv2.dnn.readNet(prototxt_path, weights_path)
    
    return face_detector

def download_face_detector():
    """Download face detector model files"""
    import urllib.request
    
    print("Downloading face detector model files...")
    
    # Create directory if needed
    os.makedirs("models", exist_ok=True)
    
    # Download prototxt
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    try:
        urllib.request.urlretrieve(prototxt_url, "deploy.prototxt")
        print("Downloaded deploy.prototxt")
    except Exception as e:
        print(f"Error downloading prototxt: {e}")
        raise
    
    # Download weights
    weights_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    try:
        urllib.request.urlretrieve(weights_url, "res10_300x300_ssd_iter_140000.caffemodel")
        print("Downloaded res10_300x300_ssd_iter_140000.caffemodel")
    except Exception as e:
        print(f"Error downloading weights: {e}")
        raise
    
    print("Face detector model files downloaded successfully")

def send_email_alert(frame, no_mask_count):
    """Send email alert with the captured image"""
    global last_email_time
    
    # Check if email configuration is set
    if EMAIL_SENDER == "your_email@gmail.com" or EMAIL_PASSWORD == "your_app_password":
        print("Email configuration not set. Skipping email alert.")
        return
    
    # Check if enough time has passed since the last email
    current_time = time.time()
    with email_lock:
        if current_time - last_email_time < EMAIL_COOLDOWN:
            return
        last_email_time = current_time
    
    # Create directory for violation images if it doesn't exist
    os.makedirs("violations", exist_ok=True)
    
    # Save the frame as an image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"violations/violation_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)
    
    # Create email message
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = EMAIL_SUBJECT
    
    # Email body
    body = f"""
    SECURITY ALERT: Face Mask Violation Detected
    
    Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Number of people without masks: {no_mask_count}
    
    Please check the attached image and take appropriate action.
    
    This is an automated message from the Face Mask Detection System.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach image
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            image = MIMEImage(img_data, name=os.path.basename(image_path))
            msg.attach(image)
    except Exception as e:
        print(f"Error attaching image to email: {e}")
        return
    
    # Send email
    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Email alert sent to {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def log_performance_data():
    """Log performance data to file"""
    if not PERFORMANCE_TRACKING:
        return
    
    os.makedirs(PERFORMANCE_LOG_PATH, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare data
    global performance_data
    log_entry = {
        "timestamp": timestamp,
        "detection_counts": performance_data["detection_counts"],
        "avg_inference_time": sum(performance_data["inference_times"]) / max(1, len(performance_data["inference_times"])),
        "false_positives": performance_data["false_positives"],
        "false_negatives": performance_data["false_negatives"]
    }
    
    # Save to file
    log_file = os.path.join(PERFORMANCE_LOG_PATH, "runtime_performance.json")
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            log_data.append(log_entry)
        except:
            log_data = [log_entry]
    else:
        log_data = [log_entry]
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    # Reset performance data
    performance_data["inference_times"] = []
    
    # Generate performance visualizations
    visualize_performance(log_data)

def visualize_performance(log_data):
    """Create visualizations of performance data"""
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    
    # Extract data for plotting
    timestamps = [entry["timestamp"] for entry in log_data]
    with_mask_counts = [entry["detection_counts"]["with_mask"] for entry in log_data]
    without_mask_counts = [entry["detection_counts"]["without_mask"] for entry in log_data]
    inference_times = [entry["avg_inference_time"] for entry in log_data]
    
    # Create date for x-axis
    x = range(len(timestamps))
    
    # Plot detection counts
    plt.figure(figsize=(12, 6))
    plt.plot(x, with_mask_counts, label='With Mask')
    plt.plot(x, without_mask_counts, label='Without Mask')
    plt.xlabel('Session')
    plt.ylabel('Count')
    plt.title('Mask Detection Counts')
    plt.legend()
    plt.xticks(x, [t.split()[1] for t in timestamps], rotation=45)
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_PATH}/detection_counts.png')
    plt.close()
    
    # Plot inference times
    plt.figure(figsize=(12, 6))
    plt.plot(x, inference_times)
    plt.xlabel('Session')
    plt.ylabel('Average Inference Time (s)')
    plt.title('Model Inference Time')
    plt.xticks(x, [t.split()[1] for t in timestamps], rotation=45)
    plt.tight_layout()
    plt.savefig(f'{VISUALIZATION_PATH}/inference_times.png')
    plt.close()

def train_model(train_from_scratch=False):
    """Train the model or load pre-trained weights"""
    # Check if model already exists
    if os.path.exists(MODEL_PATH) and not train_from_scratch:
        print(f"Loading existing model from {MODEL_PATH}")
        try:
            model = keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will train a new model instead.")
            train_from_scratch = True
    
    if train_from_scratch:
        print("Training new model...")
        
        # Process dataset
        try:
            X, Y = process_dataset()
        except Exception as e:
            print(f"Error processing dataset: {e}")
            raise
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Scale data
        X_train_scaled = X_train / 255.0
        X_test_scaled = X_test / 255.0
        
        # Build model
        model = build_model()
        
        # Set up early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Add TensorBoard logging
        os.makedirs(os.path.join(PERFORMANCE_LOG_PATH, "tensorboard"), exist_ok=True)
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=os.path.join(PERFORMANCE_LOG_PATH, "tensorboard"),
            histogram_freq=1
        )
        
        # Train model
        # Using 15 epochs as per comment in original code
        history = model.fit(
            X_train_scaled, Y_train,
            epochs=5,
            validation_split=0.1,
            batch_size=32,
            callbacks=[early_stopping, tensorboard_callback]
        )
        
        # Evaluate model with detailed metrics
        print("\nEvaluating model performance...")
        metrics = evaluate_model_performance(model, X_test, Y_test)
        print(f"\nModel Performance:\n- Accuracy: {metrics['accuracy']:.4f}\n- Precision: {metrics['precision']:.4f}\n- Recall: {metrics['recall']:.4f}\n- F1 Score: {metrics['f1']:.4f}")
        
        # Save model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
        # Plot training history
        plot_training_history(history)
        
        return model
    else:
        # Should never reach here due to the try/except above
        print(f"Creating new model as {MODEL_PATH} doesn't exist")
        return train_model(train_from_scratch=True)

def detect_face_mask():
    """Main function to detect face masks using webcam"""
    global running, performance_data
    
    # Load mask detection model
    try:
        mask_model = train_model()
    except Exception as e:
        print(f"Error loading/training model: {e}")
        return
    
    # Load face detector
    try:
        face_detector = load_face_detector()
    except Exception as e:
        print(f"Error loading face detector: {e}")
        return
    
    # Create directories if they don't exist
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("violations", exist_ok=True)
    
    # Initialize GUI window before accessing webcam
    window_name = 'Face Mask Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # Set a reasonable window size
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Please make sure your webcam is connected and not in use by another application.")
        return
    
    # Set webcam properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Face mask detection started. Press 'q' to quit.")
    print("Press 's' to take a manual screenshot.")
    print("Press 'p' to show performance metrics.")
    
    # Variable to track when the last screenshot was taken
    last_screenshot_time = 0
    last_performance_log_time = time.time()
    
    # FPS calculation variables
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    
    running = True
    
    try:
        while running:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Create a copy for display and saving
            display_frame = frame.copy()
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # FPS calculation
            frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_start_time = time.time()
            
            # Create a blob from the frame
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            # Pass the blob through the face detection network
            face_detector.setInput(blob)
            detections = face_detector.forward()
            
            # Counter for people without masks
            no_mask_count = 0
            with_mask_count = 0
            
            # Process each detection
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter weak detections
                if confidence > 0.5:
                    # Get face coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure the face region is within the frame
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    # Extract face
                    face = frame[startY:endY, startX:endX]
                    
                    if face.size == 0:
                        continue
                    
                    try:
                        # Preprocess face for mask detection
                        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        face_resized = face_pil.resize((128, 128))
                        face_array = np.array(face_resized) / 255.0
                        face_input = np.expand_dims(face_array, axis=0)
                        
                        # Record inference start time
                        if INFERENCE_SPEED_TRACKING:
                            inference_start = time.time()
                        
                        # Predict mask
                        prediction = mask_model.predict(face_input, verbose=0)
                        
                        # Record inference time
                        if INFERENCE_SPEED_TRACKING:
                            inference_time = time.time() - inference_start
                            performance_data["inference_times"].append(inference_time)
                        
                        mask_prob = prediction[0][1]  # probability of mask
                        
                        # Set color and label based on prediction
                        if mask_prob > CONFIDENCE_THRESHOLD:
                            label = "Mask"
                            color = (0, 255, 0)  # Green
                            with_mask_count += 1
                            performance_data["detection_counts"]["with_mask"] += 1
                        else:
                            label = "No Mask"
                            color = (0, 0, 255)  # Red
                            no_mask_count += 1
                            performance_data["detection_counts"]["without_mask"] += 1
                        
                        # Display prediction probability
                        text = f"{label}: {mask_prob:.2f}"
                        
                        # Draw bounding box and text
                        cv2.rectangle(display_frame, (startX, startY), (endX, endY), color, 2)
                        
                        # Ensure text is within frame
                        y_position = max(startY - 10, 15)
                        cv2.putText(display_frame, text, (startX, y_position),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
            
            # Show statistics on frame
            stats_text = f"No Mask: {no_mask_count} | With Mask: {with_mask_count} | FPS: {fps}"
            cv2.putText(display_frame, stats_text, (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, (10, h - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add instructions
            cv2.putText(display_frame, "Press 'q' to quit, 's' for screenshot, 'p' for performance", (w - 580, h - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Take screenshot automatically if people without masks are detected
            current_time = time.time()
            if no_mask_count > 0 and (current_time - last_screenshot_time > SCREENSHOT_INTERVAL):
                screenshot_path = f"screenshots/violation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"\nScreenshot saved: {screenshot_path}")
                last_screenshot_time = current_time
                
                # Send email alert in a separate thread
                threading.Thread(target=send_email_alert, args=(display_frame, no_mask_count)).start()
            
            # Log performance data periodically (every 60 seconds)
            if current_time - last_performance_log_time > 60 and PERFORMANCE_TRACKING:
                log_performance_data()
                last_performance_log_time = current_time
            
            # Always display the frame
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting detection...")
                running = False
                break
            elif key == ord('s'):
                # Manual screenshot
                screenshot_path = f"screenshots/manual_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"\nManual screenshot saved: {screenshot_path}")
            elif key == ord('p'):
                # Show performance metrics
                if PERFORMANCE_TRACKING:
                    print("\nPerformance Metrics:")
                    print(f"- With mask detections: {performance_data['detection_counts']['with_mask']}")
                    print(f"- Without mask detections: {performance_data['detection_counts']['without_mask']}")
                    if len(performance_data["inference_times"]) > 0:
                        avg_inference = sum(performance_data["inference_times"]) / len(performance_data["inference_times"])
                        print(f"- Average inference time: {avg_inference:.4f} seconds")
                    print(f"- Current FPS: {fps}")
                else:
                    print("\nPerformance tracking is disabled")
                
    except Exception as e:
        print(f"\nError during detection: {e}")
    finally:
        # Release resources
        running = False
        
        # Log final performance data
        if PERFORMANCE_TRACKING:
            log_performance_data()
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nFace mask detection stopped.")

def visualize_model_results(model, num_samples=5):
    """Visualize model predictions on sample images to assess performance"""
    try:
        # Process dataset to get test samples
        print("Loading sample images for visualization...")
        X, Y = process_dataset()
        
        # Split data
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Choose random samples
        import random
        indices = random.sample(range(len(X_test)), min(num_samples, len(X_test)))
        samples = X_test[indices]
        true_labels = y_test[indices]
        
        # Create directory for result visualization
        os.makedirs(VISUALIZATION_PATH, exist_ok=True)
        
        # Scale samples
        samples_scaled = samples / 255.0
        
        # Get predictions
        predictions = model.predict(samples_scaled)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        for i in range(len(samples)):
            plt.subplot(2, 3, i + 1)
            plt.imshow(samples[i])
            
            true_label = "With Mask" if true_labels[i] == 1 else "Without Mask"
            pred_label = "With Mask" if pred_labels[i] == 1 else "Without Mask"
            confidence = predictions[i][pred_labels[i]]
            
            # Set title color based on correctness
            title_color = "green" if true_labels[i] == pred_labels[i] else "red"
            
            # Update counters for false positives/negatives
            if true_labels[i] != pred_labels[i]:
                if pred_labels[i] == 1:  # Predicted mask when there was none
                    performance_data["false_positives"] += 1
                else:  # Predicted no mask when there was one
                    performance_data["false_negatives"] += 1
            
            plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})", color=title_color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{VISUALIZATION_PATH}/sample_predictions.png')
        print(f"Sample predictions visualization saved to {VISUALIZATION_PATH}/sample_predictions.png")
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing model results: {e}")

def generate_performance_report():
    """Generate a comprehensive performance report"""
    os.makedirs(PERFORMANCE_LOG_PATH, exist_ok=True)
    report_path = os.path.join(PERFORMANCE_LOG_PATH, "performance_report.txt")
    
    try:
        # Gather performance data
        model_performance_file = os.path.join(PERFORMANCE_LOG_PATH, "model_performance.json")
        runtime_performance_file = os.path.join(PERFORMANCE_LOG_PATH, "runtime_performance.json")
        
        model_metrics = None
        runtime_metrics = None
        
        if os.path.exists(model_performance_file):
            with open(model_performance_file, 'r') as f:
                model_metrics = json.load(f)
        
        if os.path.exists(runtime_performance_file):
            with open(runtime_performance_file, 'r') as f:
                runtime_metrics = json.load(f)
        
        # Create report
        with open(report_path, 'w') as f:
            f.write("===========================================================\n")
            f.write("             FACE MASK DETECTION PERFORMANCE REPORT        \n")
            f.write("===========================================================\n\n")
            f.write(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. MODEL PERFORMANCE METRICS\n")
            f.write("----------------------------\n")
            
            if model_metrics and len(model_metrics) > 0:
                latest = model_metrics[-1]
                f.write(f"Latest evaluation (from {latest['timestamp']}):\n")
                f.write(f"- Accuracy: {latest['accuracy']:.4f}\n")
                f.write(f"- Precision: {latest['precision']:.4f}\n")
                f.write(f"- Recall: {latest['recall']:.4f}\n")
                f.write(f"- F1 Score: {latest['f1']:.4f}\n\n")
                
                if len(model_metrics) > 1:
                    f.write("Performance trend:\n")
                    for i, metric in enumerate(model_metrics):
                        f.write(f"  Evaluation {i+1} ({metric['timestamp']}): Accuracy={metric['accuracy']:.4f}, F1={metric['f1']:.4f}\n")
            else:
                f.write("No model evaluation data available\n\n")
            
            f.write("\n2. RUNTIME PERFORMANCE\n")
            f.write("----------------------\n")
            
            if runtime_metrics and len(runtime_metrics) > 0:
                latest = runtime_metrics[-1]
                f.write(f"Latest runtime metrics (from {latest['timestamp']}):\n")
                f.write(f"- Average inference time: {latest['avg_inference_time']:.4f} seconds\n")
                f.write(f"- With mask detections: {latest['detection_counts']['with_mask']}\n")
                f.write(f"- Without mask detections: {latest['detection_counts']['without_mask']}\n")
                
                # Calculate overall statistics
                total_mask = sum(m['detection_counts']['with_mask'] for m in runtime_metrics)
                total_no_mask = sum(m['detection_counts']['without_mask'] for m in runtime_metrics)
                avg_inf_time = sum(m['avg_inference_time'] for m in runtime_metrics) / len(runtime_metrics)
                
                f.write("\nOverall runtime statistics:\n")
                f.write(f"- Total sessions: {len(runtime_metrics)}\n")
                f.write(f"- Total with mask detections: {total_mask}\n")
                f.write(f"- Total without mask detections: {total_no_mask}\n")
                f.write(f"- Overall compliance rate: {total_mask/(total_mask+total_no_mask)*100:.2f}%\n")
                f.write(f"- Average inference time across all sessions: {avg_inf_time:.4f} seconds\n")
            else:
                f.write("No runtime performance data available\n")
            
            f.write("\n3. RECOMMENDATIONS\n")
            f.write("------------------\n")
            
            # Add recommendations based on performance data
            if model_metrics and runtime_metrics:
                latest_model = model_metrics[-1]
                
                if latest_model['accuracy'] < 0.85:
                    f.write("- Model accuracy is below 85%. Consider retraining with more diverse data.\n")
                
                if latest_model['recall'] < 0.85:
                    f.write("- Low recall indicates the model is missing mask violations. Consider lowering the confidence threshold.\n")
                
                if latest_model['precision'] < 0.85:
                    f.write("- Low precision indicates many false alarms. Consider raising the confidence threshold.\n")
                
                avg_inf_time = sum(m['avg_inference_time'] for m in runtime_metrics) / len(runtime_metrics)
                if avg_inf_time > 0.1:
                    f.write("- Inference time is high. Consider optimizing the model or using hardware acceleration.\n")
            else:
                f.write("Not enough data to provide recommendations\n")
        
        print(f"Performance report generated at {report_path}")
        return report_path
        
    except Exception as e:
        print(f"Error generating performance report: {e}")
        return None

def main():
    """Main function"""
    print("\n=================================================")
    print("   Advanced Face Mask Detection System v3.0")
    print("=================================================")
    print("This system detects if people are wearing face masks, displays live feed,")
    print("and sends alerts when violations are detected.")
    print("\nBefore starting, please ensure the following:")
    print("1. Your webcam is connected and working")
    print("2. You've updated the email configuration in the script")
    print("3. Your dataset is organized in the following structure:")
    print("   - data/with_mask/")
    print("   - data/without_mask/")
    
    # Check email configuration
    if EMAIL_SENDER == "your_email@gmail.com" or EMAIL_PASSWORD == "your_app_password":
        print("\nWARNING: Email configuration is not set.")
        print("Email alerts will be disabled.")
    
    # Check OpenCV installation
    try:
        cv2_version = cv2._version_
        print(f"\nOpenCV version: {cv2_version}")
    except Exception:
        print("\nWARNING: Could not detect OpenCV version.")
    
    # Create directories for performance monitoring
    if PERFORMANCE_TRACKING:
        os.makedirs(PERFORMANCE_LOG_PATH, exist_ok=True)
        os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    
    print("\nOptions:")
    print("1. Start detection with existing model (if available)")
    print("2. Train a new model from scratch")
    print("3. Generate performance report")
    print("4. Visualize model predictions")
    print("5. Exit")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            detect_face_mask()
        elif choice == '2':
            try:
                model = train_model(train_from_scratch=True)
                print("\nModel training complete. Starting detection...")
                detect_face_mask()
            except KeyboardInterrupt:
                print("\nProcess interrupted by user.")
            except Exception as e:
                print(f"\nError during model training: {e}")
                print("Please check your dataset and try again.")
        elif choice == '3':
            report_path = generate_performance_report()
            if report_path:
                print(f"Performance report generated at {report_path}")
            else:
                print("Failed to generate performance report")
        elif choice == '4':
            try:
                model = train_model(train_from_scratch=False)
                visualize_model_results(model)
            except Exception as e:
                print(f"Error visualizing model results: {e}")
        elif choice == '5':
            print("Exiting program.")
        else:
            print("Invalid choice. Exiting program.")
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")

if _name_ == "_main_":
    main()
