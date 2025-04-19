# Kickstarter Project Success Prediction Model

## Overview

This repository implements a multimodal neural network model for predicting the success probability of Kickstarter crowdfunding projects. The model processes various types of input features including project description embeddings, category embeddings, numerical metrics, and historical creator data to provide accurate success predictions with explainable results using DeepSHAP.

## Key Features

- **Multimodal Architecture**: Processes text embeddings, categorical embeddings, and numerical features
- **Explainability**: Uses DeepSHAP to explain model decisions with feature importance visualization
- **Hyperparameter Optimization**: Automatically tunes model parameters for optimal performance
- **GPU Acceleration**: Supports GPU training for faster model development
- **Evaluation Metrics**: Provides ROC-AUC, Brier Score, and Log Loss metrics for model evaluation

## Repository Structure

```
.
├── data/                    # Data directory for storing processed data
├── models/                  # Model checkpoint directory
│   ├── best_model.pth       # Best model weights
│   └── training_params.json # Training parameters
├── src/                     # Source code directory
│   ├── data_processor.py    # Data loading and processing
│   ├── model.py             # Model architecture definition
│   ├── trainer.py           # Training implementation
│   └── explainer.py         # SHAP explainer implementation
├── logs/                    # Training logs for TensorBoard
├── evaluation/              # Model evaluation results
│   ├── accuracy_results.json # Detailed metrics and SHAP analysis
│   ├── roc_curve.png        # ROC curve visualization
│   └── feature_importance.png # Feature importance visualization
├── explanations/            # SHAP explanation visualizations
├── predictions/             # Prediction results for new projects
├── train.py                 # Main training script
├── predict.py               # Prediction script for new projects
├── analyze_shap_distribution.py  # SHAP value distribution analysis script
├── analyze_success_rate.py       # Project success rate analysis script
├── testing_analysis.py           # Model evaluation pipeline
├── Code_Description.md      # Detailed code documentation
├── Training_Guide.md        # Comprehensive training guide
├── Usage_Guide.md           # Detailed usage instructions
└── requirements.txt         # Dependencies
```

## Installation

1. Clone the repository
2. Install dependencies

```bash
pip install -r requirements.txt
```

## Data

Training data can be downloaded from [this link](https://share.weiyun.com/uHtvjEBY). The data should be placed in the `data/` directory with the filename `allProcessed.json`.

The expected data format is a JSON file with each record containing:
- Embedding vectors (description, blurb, risk, category, subcategory, country)
- Numerical features (description length, funding goal, image count, etc.)
- Target label (state: 1 for success, 0 for failure)

### Data Preparation

The dataset for this project is prepared through data processing methods developed in the [CrowdInsight-DataProcessing](https://github.com/angusf777/CrowdInsight-Data) repository. This companion repository contains specialized tools for processing raw Kickstarter data, including:

- Text processing and embedding generation
- Feature extraction and normalization
- Category and subcategory encoding
- Creator history aggregation
- Data validation and quality assurance

## Quick Start

### Training the Model

For basic model training:

```bash
python train.py --data_path data/allProcessed.json
```

For model training with hyperparameter optimization:

```bash
python train.py --data_path data/allProcessed.json --optimize_hyperparams
```

For more detailed training instructions, refer to the [Training Guide](Training_Guide.md).

### Making Predictions

To predict the success probability of a new project:

```bash
python predict.py --input_json sample_input.json
```

This will generate prediction results and SHAP explanations in the `predictions/` directory.

For more detailed usage instructions, refer to the [Usage Guide](Usage_Guide.md).

### Model Evaluation

To evaluate the model on the test dataset and generate comprehensive metrics:

```bash
python testing_analysis.py
```

This will:
1. Load the trained model
2. Process the test data
3. Calculate performance metrics and SHAP values
4. Generate visualizations
5. Save results to the `evaluation/` directory

## Core Components

### Model Architecture

The model architecture consists of:
1. Separate processing branches for different embedding types
2. Numerical feature processing
3. Feature fusion through concatenation
4. Fully connected layers
5. Output layer for success probability prediction

```python
# Model initialization example
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

### Explainability

The model uses SHAP to provide feature contribution values that explain which features increase or decrease the success probability:

```
{
  "prediction": {
    "success_probability": 0.75,
    "predicted_outcome": "Success"
  },
  "explanation": {
    "shap_values": {
      "description_embedding": 0.2,
      "funding_goal": -0.15,
      "campaign_duration": 0.05,
      "previous_success_rate": 0.1,
      ...
    }
  }
}
```

Positive SHAP values indicate an increase in success probability, while negative values indicate a decrease.

## Model Performance

Current model performance on the test set exceeds all target metrics:

| Metric | Target | Actual Performance |
|--------|--------|-------------------|
| ROC-AUC | ≥ 0.75 | 0.961 |
| Brier Score | ≤ 0.2 | 0.075 |
| Log Loss | ≤ 0.6 | 0.248 |
| Accuracy | - | 90.24% |
| F1 Score | - | 0.922 |
| Precision | - | 0.906 |
| Recall | - | 0.938 |

### Key Insights

Analysis of SHAP values reveals the most influential factors for project success:

1. **Funding Goal**: Has the strongest impact (negative) on success probability - higher goals decrease chances of success
2. **Description Quality**: The second most important feature, indicating content quality matters
3. **Description Length**: Longer, more detailed descriptions tend to correlate with success
4. **Risk Assessment**: How risks are presented and addressed influences success
5. **Category Selection**: Some categories have higher success rates than others

## License

See LICENSE file for details.

## Contact

For questions, please contact angusf777@gmail.com 