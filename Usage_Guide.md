# Kickstarter Project Success Prediction Model Usage Guide

## 1. Environment Configuration

### 1.1 System Requirements
- Python 3.7+
- PyTorch 1.10.0+
- CUDA compatible GPU (optional, but recommended)

### 1.2 Installing Dependencies
```bash
pip install -r requirements.txt
```

### 1.3 Directory Structure
```
.
├── data/                    # Data directory for storing processed data
├── models/                  # Model checkpoint directory
├── src/                     # Source code directory
├── logs/                    # Training logs for TensorBoard
├── evaluation/              # Model evaluation results
├── explanations/            # SHAP explanation visualizations
├── predictions/             # Prediction results for new projects
```

## 2. Data Preparation

### 2.1 Data Format
Input data should be in JSON format, containing the following fields:

#### Embedding Features
- `description_embedding`: 768-dimensional vector
- `blurb_embedding`: 384-dimensional vector
- `risk_embedding`: 384-dimensional vector
- `subcategory_embedding`: 100-dimensional vector
- `category_embedding`: 15-dimensional vector
- `country_embedding`: 100-dimensional vector

#### Numerical Features
- `description_length`: Description length
- `funding_goal`: Funding goal
- `image_count`: Number of images in the project
- `video_count`: Number of videos in the project
- `campaign_duration`: Crowdfunding campaign duration
- `previous_projects_count`: Historical project count
- `previous_success_rate`: Historical success rate
- `previous_pledged`: Historical pledged amount
- `previous_funding_goal`: Historical funding goal

#### Labels
- `state`: 1 for success, 0 for failure

### 2.2 Sample Data Format
```json
{
  "description_embedding": [0.1, 0.2, ..., 0.3],
  "blurb_embedding": [0.4, 0.5, ..., 0.6],
  "risk_embedding": [0.7, 0.8, ..., 0.9],
  "subcategory_embedding": [0.1, 0.2, ..., 0.3],
  "category_embedding": [0.4, 0.5, ..., 0.6],
  "country_embedding": [0.7, 0.8, ..., 0.9],
  "description_length": 1200,
  "funding_goal": 10000,
  "image_count": 5,
  "video_count": 1,
  "campaign_duration": 30,
  "previous_projects_count": 2,
  "previous_success_rate": 0.5,
  "previous_pledged": 12000,
  "previous_funding_goal": 8000,
  "state": 1
}
```

### 2.3 Pre-processing Data
If you need to pre-process raw Kickstarter data:

```bash
python src/data_processor.py --input_file raw_data.json --output_file processed_data.json
```

## 3. Model Training

### 3.1 Basic Training
```bash
python train.py --data_path data/allProcessed.json
```

### 3.2 Advanced Training Options
```bash
python train.py \
    --data_path data/allProcessed.json \
    --batch_size 32 \
    --hidden_dim 256 \
    --dropout_rate 0.3 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --early_stop_patience 10
```

### 3.3 Hyperparameter Optimization
```bash
python train.py --data_path data/allProcessed.json --optimize_hyperparams
```

### 3.4 Resuming Training from Checkpoint
```bash
python train.py --data_path data/allProcessed.json --checkpoint_dir models
```

### 3.5 Monitoring Training
Monitor training progress with TensorBoard:
```bash
tensorboard --logdir logs
```

## 4. Model Prediction

### 4.1 Using the predict.py Script
The simplest way to generate predictions:

```bash
python predict.py --input_json sample_input.json
```

This will:
1. Load the best model
2. Process the input data
3. Generate prediction with probability
4. Calculate SHAP values
5. Save results to the `predictions/` directory

### 4.2 Programmatic Usage
```python
from src.model import KickstarterModel
from src.explainer import KickstarterExplainer
import torch
import json

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KickstarterModel(hidden_dim=256)  # Match the training architecture
checkpoint = torch.load('models/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.to(device)
model.eval()

# Load sample data
with open('sample_input.json', 'r') as f:
    sample = json.load(f)

# Prepare input data
inputs = {
    'description_embedding': torch.tensor([sample['description_embedding']], dtype=torch.float32).to(device),
    'blurb_embedding': torch.tensor([sample['blurb_embedding']], dtype=torch.float32).to(device),
    'risk_embedding': torch.tensor([sample['risk_embedding']], dtype=torch.float32).to(device),
    'subcategory_embedding': torch.tensor([sample['subcategory_embedding']], dtype=torch.float32).to(device),
    'category_embedding': torch.tensor([sample['category_embedding']], dtype=torch.float32).to(device),
    'country_embedding': torch.tensor([sample['country_embedding']], dtype=torch.float32).to(device),
    'numerical_features': torch.tensor([[
        sample['description_length'],
        sample['funding_goal'],
        sample['image_count'],
        sample['video_count'],
        sample['campaign_duration'],
        sample['previous_projects_count'],
        sample['previous_success_rate'],
        sample['previous_pledged'],
        sample['previous_funding_goal']
    ]], dtype=torch.float32).to(device)
}

# Predict
with torch.no_grad():
    probability, _ = model(inputs)
    prediction = probability.item()
    
print(f"Success probability: {prediction:.4f}")
print(f"Predicted outcome: {'Success' if prediction >= 0.5 else 'Failure'}")
```

### 4.3 Batch Prediction
```python
from src.data_processor import KickstarterDataProcessor

# Load data
data_processor = KickstarterDataProcessor('test_data.json')
test_loader = data_processor.get_test_dataloader(batch_size=32)

# Batch prediction
results = []
with torch.no_grad():
    for batch in test_loader:
        # Move data to device
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        
        # Predict
        probabilities, _ = model(inputs)
        
        # Store results
        for i, prob in enumerate(probabilities):
            results.append({
                'id': batch.get('id', i),
                'probability': prob.item(),
                'predicted_outcome': 'Success' if prob.item() >= 0.5 else 'Failure'
            })
```

## 5. Model Evaluation

### 5.1 Running Comprehensive Evaluation
```bash
python testing_analysis.py
```

This will:
1. Load the best model
2. Process the test data
3. Calculate all evaluation metrics
4. Generate visualizations
5. Save results to the `evaluation/` directory

### 5.2 Current Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.961 |
| Brier Score | 0.075 |
| Log Loss | 0.248 |
| Accuracy | 90.24% |
| F1 Score | 0.922 |
| Precision | 0.906 |
| Recall | 0.938 |

### 5.3 Analyzing Evaluation Results
Check the evaluation directory for:
- `accuracy_results.json`: Complete metrics and SHAP analysis
- `roc_curve.png`: ROC curve visualization
- `feature_importance.png`: Feature importance chart

## 6. Model Explanation

### 6.1 Getting SHAP Values
```python
# Create explainer
explainer = KickstarterExplainer(model)

# Generate explanation
prediction, shap_values = explainer.explain_prediction(inputs)

# Print feature contributions
for feature, value in shap_values.items():
    print(f"{feature}: {value:.4f}")
```

### 6.2 Interpreting SHAP Values
- **Positive SHAP values**: Feature pushes prediction toward "success"
- **Negative SHAP values**: Feature pushes prediction toward "failure"
- **SHAP value magnitude**: Strength of feature influence

### 6.3 Key Insights from SHAP Analysis
Based on our latest evaluation, the most impactful features are:

1. **Funding Goal**: The strongest predictor (negative correlation)
2. **Description Quality**: Projects with high-quality descriptions perform better
3. **Campaign Duration**: Optimal duration is around 30 days
4. **Creator History**: Previous success rate strongly influences predictions
5. **Media Content**: Projects with more images and videos tend to perform better

### 6.4 Visualizing SHAP Values
```python
explainer.visualize_shap_values(shap_values, output_path='explanations/shap_values.png')
```

## 7. Advanced Usage

### 7.1 Custom Model Configurations
Modify model architecture by changing parameters:

```python
model = KickstarterModel(
    desc_embedding_dim=768,
    blurb_embedding_dim=384,
    risk_embedding_dim=384,
    subcategory_embedding_dim=100,
    category_embedding_dim=15,
    country_embedding_dim=100,
    numerical_features_dim=9,
    hidden_dim=256,
    dropout_rate=0.3
)
```

### 7.2 Analyzing SHAP Value Distributions
To analyze SHAP value distributions across the dataset:

```bash
python analyze_shap_distribution.py
```

This generates insights about feature importance across different types of projects.

### 7.3 Custom Thresholds
Adjust prediction threshold based on business needs:

```python
# For high-precision predictions
high_precision_threshold = 0.7
prediction = 'Success' if probability >= high_precision_threshold else 'Failure'

# For high-recall predictions
high_recall_threshold = 0.3
prediction = 'Success' if probability >= high_recall_threshold else 'Failure'
```

### 7.4 Using Custom Test Data
```bash
python testing_analysis.py --test_data custom_test_data.json
```