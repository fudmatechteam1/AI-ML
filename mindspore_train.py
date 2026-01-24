"""
MindSpore Training Script for Tech-Trust AI Model
Trains the trust score prediction model using Huawei MindSpore framework
"""
import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import mindspore as ms
from mindspore import nn, Tensor, context
from mindspore.dataset import GeneratorDataset
import model  # Import the model definition and helper functions

# Set context (use GPU if available, otherwise CPU)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # Change to "GPU" if GPU available


class TrustScoreDataset:
    """Dataset generator for MindSpore training"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)


def create_dataset(X, y, batch_size=16, shuffle=True):
    """Create MindSpore dataset from numpy arrays"""
    dataset = TrustScoreDataset(X, y)
    dataset = GeneratorDataset(dataset, column_names=['features', 'labels'], shuffle=shuffle)
    dataset = dataset.batch(batch_size)
    return dataset


def train_mindspore_model(df, model_save_path='trust_model_with_credentials.ckpt', 
                         scaler_save_path='scaler.joblib'):
    """
    Train MindSpore model on the prepared dataframe
    
    Args:
        df: DataFrame with engineered features
        model_save_path: Path to save the trained model checkpoint
        scaler_save_path: Path to save the scaler
    """
    print("\n" + "="*60)
    print("MINSPORE MODEL TRAINING")
    print("="*60 + "\n")
    
    # Extract features (same as original model.py)
    features = [
        'popularity_score',
        'activity_score',
        'collab_score',
        'tech_diversity',
        'language_score',
        'repo_count',
        'credential_score',
        'credential_count',
        'credential_diversity'
    ]
    
    X = df[features].values.astype(np.float32)
    y = df['trust_score'].values.astype(np.float32).reshape(-1, 1)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features: {features}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Save scaler
    joblib.dump(scaler, scaler_save_path)
    print(f"[OK] Scaler saved to {scaler_save_path}\n")
    
    # Create datasets
    train_dataset = create_dataset(X_train_scaled, y_train, batch_size=16, shuffle=True)
    test_dataset = create_dataset(X_test_scaled, y_test, batch_size=16, shuffle=False)
    
    # Build model
    input_dim = X_train.shape[1]
    network = model.TrustScoreModel(input_dim)
    
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # Define optimizer
    optimizer = nn.Adam(network.trainable_params(), learning_rate=0.001)
    
    # Define forward function
    def forward_fn(x, y):
        logits = network(x)
        loss = loss_fn(logits, y)
        return loss
    
    # Define gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
    
    # Define train step
    def train_step(x, y):
        loss, grads = grad_fn(x, y)
        optimizer(grads)
        return loss
    
    print("Starting training...\n")
    
    # Training loop with validation
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    best_weights = None
    
    for epoch in range(150):
        # Train
        network.set_train(True)
        train_losses = []
        for batch in train_dataset.create_dict_iterator():
            features = Tensor(batch['features'])
            labels = Tensor(batch['labels'])
            loss = train_step(features, labels)
            train_losses.append(loss.asnumpy())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validate
        network.set_train(False)
        val_losses = []
        val_maes = []
        for batch in test_dataset.create_dict_iterator():
            features = Tensor(batch['features'])
            labels = Tensor(batch['labels'])
            pred = network(features)
            loss = loss_fn(pred, labels)
            mae = nn.MAE()(pred, labels)
            val_losses.append(loss.asnumpy())
            val_maes.append(mae)
        
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/150]")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f}")
            print(f"  Val MAE:    {avg_val_mae:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best weights
            best_weights = {k: v.asnumpy() for k, v in network.parameters_dict().items()}
            # Save checkpoint
            ms.save_checkpoint(network, model_save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Load best weights
    if best_weights:
        param_dict = {}
        for name, param in network.parameters_dict().items():
            param_dict[name] = ms.Parameter(Tensor(best_weights[name]), name=name)
        ms.load_param_into_net(network, param_dict)
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    network.set_train(False)
    y_pred_list = []
    for batch in test_dataset.create_dict_iterator():
        features = Tensor(batch['features'])
        pred = network(features)
        y_pred_list.append(pred.asnumpy())
    
    y_pred = np.concatenate(y_pred_list, axis=0).flatten()
    y_test_flat = y_test.flatten()
    
    # Calculate metrics
    r2 = r2_score(y_test_flat, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred))
    mae = np.mean(np.abs(y_test_flat - y_pred))
    
    print(f" Test MAE:  {mae:.4f} (Mean Absolute Error)")
    print(f" Test RMSE: {rmse:.4f} (Root Mean Squared Error)")
    print(f" R² Score:  {r2:.4f} (Coefficient of Determination)")
    print(f"{'='*50}\n")
    
    # Save final model
    ms.save_checkpoint(network, model_save_path)
    print(f"[OK] Model saved as {model_save_path}")
    
    # Save metrics
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'test_loss': float(best_val_loss),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': features,
        'trained_on': datetime.now().isoformat(),
        'framework': 'MindSpore',
        'epochs_completed': epoch + 1
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("[OK] Metrics saved to model_metrics.json")
    
    return network, metrics


def main():
    """Main training function"""
    print("\n" + "="*60)
    print("TECH-TRUST AI MODEL TRAINING (MINSPORE)")
    print("="*60 + "\n")
    
    # Check for required files
    if not os.path.exists("github_dataset.csv"):
        print("❌ ERROR: github_dataset.csv not found!")
        print("Please ensure your GitHub dataset is in the current directory.")
        return
    
    # Initialize credential verifier (optional)
    credential_verifier = None
    if os.path.exists('credentials.csv'):
        print("✓ Credentials database found - enabling credential verification")
        credential_verifier = model.CredentialVerifier('credentials.csv')
    else:
        print("⚠ Warning: credentials.csv not found")
        print("  Model will train WITHOUT credential verification features")
        print("  To enable credentials, create credentials.csv with your data\n")
    
    # Check for developer credentials mapping file
    credentials_file = None
    for fname in ['developer_credentials.csv', 'comprehensive_credentials_database.csv']:
        if os.path.exists(fname):
            temp_df = pd.read_csv(fname, nrows=1)
            if 'username' in temp_df.columns and 'credentials' in temp_df.columns:
                credentials_file = fname
                print(f"✓ Developer credentials mapping found: {fname}")
                break
    
    if not credentials_file:
        print("⚠ No developer credentials mapping file found")
        print("  Training with empty credentials for all developers\n")
    
    # Load and prepare data
    df = model.load_and_prepare_data("github_dataset.csv", credentials_file=credentials_file)
    
    # Engineer features
    df = model.engineer_features(df, credential_verifier)
    df.to_csv("developer_training_data_with_credentials.csv", index=False)
    
    print("[OK] Training dataset saved\n")
    
    # Train model
    network, metrics = train_mindspore_model(df)
    
    print("="*60)
    print("TRAINING COMPLETE – MODEL READY FOR DEPLOYMENT")
    print("="*60)
    print(f"Model file: trust_model_with_credentials.ckpt")
    print(f"Scaler file: scaler.joblib")
    print(f"Metrics file: model_metrics.json")
    if credential_verifier:
        print(f"Credentials DB: credentials.csv")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
