import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_loader import TinySTGNN_Optimized

# Configuration
DATA_DIR = ".." # Parent directory containing .npy files
OUTPUT_DIR = "." # Current directory to save models

def main():
    print("🚀 Starting Model Setup Pipeline...")
    
    # 1. Load Data
    x_path = os.path.join(DATA_DIR, "X_aug.npy")
    y_path = os.path.join(DATA_DIR, "Y_aug.npy")
    stgnn_path = os.path.join(DATA_DIR, "best_stgnn_hybrid.pth")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"❌ Error: Data files not found in {os.path.abspath(DATA_DIR)}")
        print("Please ensure X_aug.npy and Y_aug.npy exist in the parent directory.")
        return

    print("✅ Loading augmented data...")
    X_all = np.load(x_path) # [N, 5, 10, 14]
    Y_raw = np.load(y_path) # [N, 1]
    
    # 2. Process Labels (Top 30% Threshold)
    THRESHOLD_PERCENTILE = 70 
    threshold = np.percentile(Y_raw, THRESHOLD_PERCENTILE)
    print(f"ℹ️  Overload Threshold (Top {100-THRESHOLD_PERCENTILE}%): Score >= {threshold:.4f}")
    
    Y_binary = (Y_raw >= threshold).astype(int).flatten()
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, Y_binary, test_size=0.2, random_state=42, stratify=Y_binary
    )
    
    # 4. Load Pre-trained ST-GNN
    print("✅ Loading ST-GNN weights...")
    device = torch.device("cpu")
    stgnn = TinySTGNN_Optimized()
    
    if os.path.exists(stgnn_path):
        stgnn.load_state_dict(torch.load(stgnn_path, map_location=device))
        # Save a copy to local directory for self-containment
        torch.save(stgnn.state_dict(), os.path.join(OUTPUT_DIR, "best_stgnn_hybrid.pth"))
        print(f"💾 Saved local copy of ST-GNN to {os.path.join(OUTPUT_DIR, 'best_stgnn_hybrid.pth')}")
    else:
        print("⚠️  Warning: Pre-trained ST-GNN weights not found. Using random initialization (Not recommended for inference).")
    
    stgnn.to(device)
    stgnn.eval()
    
    # 5. Extract Features for GB Training
    print("🔄 Extracting hybrid features using ST-GNN...")
    
    def extract(X_data):
        # Batch processing to avoid memory issues
        batch_size = 32
        n_samples = len(X_data)
        features_list = []
        raw_list = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = X_data[i:i+batch_size]
                tensor = torch.FloatTensor(batch).to(device)
                _, feats = stgnn(tensor, return_features=True)
                features_list.append(feats.cpu().numpy())
                
                # Flatten raw
                raw_flat = batch.reshape(len(batch), -1)
                raw_list.append(raw_flat)
                
        return np.vstack(features_list), np.vstack(raw_list)

    train_stgnn_feats, train_raw = extract(X_train)
    test_stgnn_feats, test_raw = extract(X_test)
    
    # 6. Normalize Raw Features
    print("⚖️  Fitting StandardScaler...")
    scaler = StandardScaler()
    train_raw_norm = scaler.fit_transform(train_raw)
    test_raw_norm = scaler.transform(test_raw)
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print("💾 Saved scaler.pkl")
    
    # 7. Concatenate
    X_train_hybrid = np.hstack([train_raw_norm, train_stgnn_feats])
    X_test_hybrid = np.hstack([test_raw_norm, test_stgnn_feats])
    
    # 8. Train Gradient Boosting
    print("🏋️  Training Gradient Boosting Classifier...")
    gb_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    gb_clf.fit(X_train_hybrid, y_train)
    
    # 9. Evaluate & Save
    acc = gb_clf.score(X_test_hybrid, y_test)
    print(f"✅ GB Training Complete. Test Accuracy: {acc:.4f}")
    
    joblib.dump(gb_clf, os.path.join(OUTPUT_DIR, "gb_model.pkl"))
    print("💾 Saved gb_model.pkl")
    
    print("\n🎉 Setup Complete! You can now run the Streamlit app.")

if __name__ == "__main__":
    main()
