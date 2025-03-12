import numpy as np
import os
import mne
from scipy.signal import butter, lfilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Band-pass filter function
def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=160, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=1)

# EEG Preprocessing Function
def preprocess_edf_fixed(edf_path, segment_length=500):
    """Loads, filters, and segments EEG data from an EDF file."""
    print(f"🔍 Loading EDF file: {edf_path}")
   
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"❌ Error loading EDF file: {e}")
        return None, None

    # Select Correct EEG Channels
    eeg_channels = [ch for ch in raw.ch_names if ch.endswith('.') or ch.endswith('..')]
   
    if not eeg_channels:
        print("❌ No EEG channels found! Skipping file.")
        return None, None

    raw.pick_channels(eeg_channels)

    # Apply band-pass filter (0.5 - 40Hz)
    raw.filter(0.5, 40.0, fir_design='firwin')

    # Normalize EEG data
    data = raw.get_data().astype(np.float32)  # Explicitly convert to float32 for PyTorch
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    # Read Event Markers from Annotations
    annotations = mne.read_annotations(edf_path)
    raw.set_annotations(annotations)

    events = [(annot['onset'], annot['description']) for annot in annotations]
    print(f"✅ Extracted {len(events)} event markers.")

    # Segment EEG Data
    segments = []
    for start_idx in range(0, data.shape[1] - segment_length + 1, segment_length // 2):
        segment = data[:, start_idx:start_idx + segment_length]
        if segment.shape[1] == segment_length:
            segments.append(segment)

    if len(segments) == 0:
        print("⚠ No valid EEG segments found.")
        return None, None

    print(f"✅ Processed {len(segments)} EEG segments from {edf_path}")
    return np.array(segments), events

# PyTorch Dataset Class
class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
       
    def __len__(self):
        return len(self.labels)
       
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# PyTorch LSTM Model
class LSTMEEG(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(LSTMEEG, self).__init__()
       
        # LSTM parameters
        self.lstm_input_size = input_shape[0]  # Number of EEG channels
        self.lstm_hidden_size = 64
        self.num_layers = 2
       
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
       
        self.dropout = nn.Dropout(0.3)
       
        # Fully connected layers
        self.fc1 = nn.Linear(self.lstm_hidden_size * 2, 128)  # *2 because bidirectional
        self.fc2 = nn.Linear(128, num_classes)
       
    def forward(self, x):
        # Input shape: [batch, channels, time]
       
        # Reshape for LSTM: [batch, time, features]
        x = x.permute(0, 2, 1)
       
        # Apply LSTM layers
        x, _ = self.lstm(x)
        x = self.dropout(x)
       
        # Take the output of the last time step
        x = x[:, -1, :]
       
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
       
        return x

# Train function
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=30):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
   
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
       
        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
           
            # Zero the parameter gradients
            optimizer.zero_grad()
           
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
           
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
           
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
       
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
       
        # Validation loop
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
       
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
               
                outputs = model(inputs)
                loss = criterion(outputs, labels)
               
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
       
        val_epoch_loss = val_running_loss / len(test_loader)
        val_epoch_acc = 100 * val_correct / val_total
        test_losses.append(val_epoch_loss)
        test_accs.append(val_epoch_acc)
       
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
        print("-" * 50)
   
    return model, train_losses, test_losses, train_accs, test_accs

# Main Execution
def main():
    dataset_dir = r"C:\Users\Admin\Documents\ML SARVESH Project BCI\EEG-Signal Dataset\eeg-motor-movementimagery-dataset-1.0.0\files"
   
    all_X, all_y = [], []
    label_encoder = LabelEncoder()
    event_labels_list = []
   
    print("🛠 Scanning directory for EDF files...")
   
    # Find all EDF files in all subject folders
    edf_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.edf'):
                edf_files.append(os.path.join(root, file))
   
    if not edf_files:
        print("❌ No EDF files found in the dataset directory. Exiting.")
        return
   
    print(f"📂 Found {len(edf_files)} EDF files. Processing...")
   
    # Tracking variables for successful processing
    files_processed = 0
    files_skipped = 0
   
    # First pass: collect all unique event labels for encoding
    for edf_file in edf_files:
        print(f"\n🔍 First pass - collecting event labels from {edf_file}...")
        _, events = preprocess_edf_fixed(edf_file, segment_length=500)
       
        if events:
            event_labels = [event[1] for event in events]
            event_labels_list.extend(event_labels)
   
    # Encode labels
    unique_labels = list(set(event_labels_list))
    if not unique_labels:
        print("❌ No event labels found. Exiting.")
        return
       
    label_encoder.fit(unique_labels)
    print(f"✅ Found {len(unique_labels)} unique event types")
    print(f"Label Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
   
    # Second pass: process data and associate segments with events
    for edf_file in edf_files:
        print(f"\n🔍 Second pass - processing {edf_file}...")
        segments, events = preprocess_edf_fixed(edf_file, segment_length=500)
       
        if segments is None or not events:
            print(f"⚠ Skipping {edf_file} due to missing data.")
            files_skipped += 1
            continue
       
        # Get sampling frequency from the file
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            fs = raw.info['sfreq']
        except Exception as e:
            print(f"❌ Error reading sampling frequency: {e}")
            files_skipped += 1
            continue
           
        # Associate each segment with the nearest event
        # Calculate the time for each segment
        segment_length = 500
        segment_times = []
       
        for i in range(0, segments.shape[0]):
            # Calculate the time of this segment (midpoint)
            segment_start = i * (segment_length // 2)
            segment_mid = segment_start + (segment_length // 2)
            segment_time = segment_mid / fs  # Convert to seconds
            segment_times.append(segment_time)
       
        # For each segment, find the closest event
        segment_labels = []
        for segment_time in segment_times:
            closest_event = None
            min_distance = float('inf')
           
            for event_time, event_label in events:
                distance = abs(segment_time - event_time)
                if distance < min_distance:
                    min_distance = distance
                    closest_event = event_label
           
            # If the closest event is too far, assign a default label
            if min_distance > 5.0:  # 5 seconds threshold
                closest_event = "unknown"
               
            segment_labels.append(closest_event)
       
        # Encode the segment labels
        encoded_labels = []
        for label in segment_labels:
            if label in label_encoder.classes_:
                encoded_labels.append(label_encoder.transform([label])[0])
            else:
                # Handle unknown labels (if any)
                if "unknown" not in label_encoder.classes_:
                    # Skip segments with unknown labels
                    continue
                encoded_labels.append(label_encoder.transform(["unknown"])[0])
       
        # Add valid segments and their labels to the dataset
        valid_segments = []
        valid_labels = []
       
        for i, label in zip(range(len(encoded_labels)), encoded_labels):
            if label is not None:
                valid_segments.append(segments[i])
                valid_labels.append(label)
       
        all_X.append(np.array(valid_segments))
        all_y.append(valid_labels)
       
        files_processed += 1

    print(f"\n✅ Processed {files_processed} files successfully.")
    print(f"⚠ Skipped {files_skipped} files due to issues.")
   
    # Combine all segments and labels
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
   
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    # Create PyTorch datasets and loaders
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)
   
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
   
    # Initialize model, loss function, and optimizer
    model = LSTMEEG(input_shape=X_train.shape[1:], num_classes=len(unique_labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    model, train_losses, test_losses, train_accs, test_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, epochs=30
    )

    # Evaluate the model - testing
    model.eval()

    # Make predictions
    all_preds = []
    all_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)
           
            # Forward pass
            outputs = model(inputs)
           
            # Get the predicted class
            _, predicted = torch.max(outputs, 1)
           
            # Store predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=label_encoder.classes_))

    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Execute the main function
if __name__ == "__main__":
    main()
