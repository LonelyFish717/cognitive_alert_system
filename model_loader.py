import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
import numpy as np

# ==============================================================================
# ST-GNN Architecture
# ==============================================================================
class DynamicGraphLayer(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=16):
        super(DynamicGraphLayer, self).__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        
    def forward(self, x):
        B, N, F_dim = x.size()
        mean = x.mean(dim=2, keepdim=True)
        xm = x - mean
        var = torch.sum(xm ** 2, dim=2, keepdim=True)
        std = torch.sqrt(var) + 1e-8
        xm_norm = xm / std
        corr = torch.bmm(xm_norm, xm_norm.transpose(1, 2))
        A = torch.abs(corr)
        x_agg = torch.bmm(A, x)
        x_hidden = self.fc(x_agg)
        x_hidden = F.relu(x_hidden)
        x_out = torch.mean(x_hidden, dim=1)
        return x_out

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim=32):
        super(TemporalAttention, self).__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_out):
        scores = self.linear(lstm_out)
        scores = torch.tanh(scores)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        return context

class TinySTGNN_Optimized(nn.Module):
    def __init__(self, input_dim=14, gat_hidden=16, lstm_hidden=32, num_classes=2):
        super(TinySTGNN_Optimized, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.spatial_layer = DynamicGraphLayer(in_dim=input_dim, hidden_dim=gat_hidden)
        self.lstm = nn.LSTM(input_size=gat_hidden, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.temporal_att = TemporalAttention(hidden_dim=lstm_hidden)
        self.classifier = nn.Sequential(nn.Linear(lstm_hidden, num_classes))
        
    def forward(self, x, return_features=False):
        # x shape: [B, 5, 10, 14] -> Permute to [B, 10, 5, 14]
        x = x.permute(0, 2, 1, 3) 
        x = self.layer_norm(x)
        batch_size, seq_len, num_nodes, feat_dim = x.size()
        
        spatial_outputs = []
        for t in range(seq_len):
            xt = x[:, t, :, :]
            st = self.spatial_layer(xt)
            spatial_outputs.append(st)
            
        spatial_seq = torch.stack(spatial_outputs, dim=1)
        lstm_out, _ = self.lstm(spatial_seq)
        context_vector = self.temporal_att(lstm_out) # [B, 32]
        
        logits = self.classifier(context_vector)
        
        if return_features:
            return logits, context_vector
        return logits

# ==============================================================================
# Model Loading & Inference
# ==============================================================================
def load_models(model_dir='.'):
    """
    Load ST-GNN, GB Classifier, and Scaler.
    """
    device = torch.device("cpu") # Inference on CPU is safer for web apps
    
    # 1. Load ST-GNN
    stgnn = TinySTGNN_Optimized()
    stgnn_path = os.path.join(model_dir, 'best_stgnn_hybrid.pth')
    if os.path.exists(stgnn_path):
        stgnn.load_state_dict(torch.load(stgnn_path, map_location=device))
    else:
        print(f"Warning: {stgnn_path} not found.")
    stgnn.to(device)
    stgnn.eval()
    
    # 2. Load GB Classifier
    gb_path = os.path.join(model_dir, 'gb_model.pkl')
    if os.path.exists(gb_path):
        gb_clf = joblib.load(gb_path)
    else:
        print(f"Warning: {gb_path} not found.")
        gb_clf = None
        
    # 3. Load Scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        print(f"Warning: {scaler_path} not found.")
        scaler = None
        
    return stgnn, gb_clf, scaler

def predict_cognitive_state(stgnn, gb_clf, scaler, input_tensor):
    """
    Run hybrid inference.
    input_tensor: [1, 5, 10, 14]
    """
    # 1. ST-GNN Feature Extraction
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_tensor)
        _, latent_features = stgnn(input_tensor, return_features=True)
        latent_features = latent_features.numpy() # [1, 32]
        
    # 2. Raw Features Flattening & Normalization
    # Input [1, 5, 10, 14] -> [1, 700]
    raw_flat = input_tensor.numpy().reshape(1, -1)
    
    if scaler:
        raw_norm = scaler.transform(raw_flat)
    else:
        raw_norm = raw_flat # Fallback if no scaler
        
    # 3. Fusion
    hybrid_features = np.hstack([raw_norm, latent_features]) # [1, 732]
    
    # 4. Prediction
    if gb_clf:
        prediction = gb_clf.predict(hybrid_features)[0]
        proba = gb_clf.predict_proba(hybrid_features)[0]
        return prediction, proba
    else:
        # Fallback to ST-GNN classifier if GB missing
        # This is just a safeguard
        logits = stgnn(input_tensor)
        pred = torch.argmax(logits, dim=1).item()
        return pred, [0.5, 0.5]
