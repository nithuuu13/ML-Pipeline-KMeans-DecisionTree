# Machine Learning Pipeline - Clustering & Decision Trees

## Overview
A complete machine learning pipeline implementing KMeans clustering and Decision Tree regression for data analysis and prediction tasks.

## Features
- **Data Loading & Preprocessing**: CSV handling with pandas
- **KMeans Clustering**: Unsupervised learning for pattern discovery (5 clusters)
- **Decision Tree Regression**: Supervised learning for performance prediction
- **End-to-End Pipeline**: Data loading → Training → Prediction → Export

## Implementation Details

### Clustering
- Algorithm: KMeans (k=5)
- Features: `publishedperformance`, `estimatedperformance`
- Output: Cluster labels mapped to categories (a, b, c, d, e)

### Decision Tree
- Algorithm: Decision Tree Regressor (max_depth=5)
- Features: `channelmin`, `channelmax`
- Target: `estimatedperformance`

## Technologies Used
- Python 3
- pandas - Data manipulation
- scikit-learn - Machine learning models
- numpy - Numerical operations

## File Structure
- `Nidharshani_23057532.py` - Main implementation with all functions

## Key Functions
- `load_file()` - Load and structure CSV data
- `train_clustering_model()` - Train KMeans model
- `train_decision_tree()` - Train Decision Tree regressor
- `save_to_file()` - Export results to CSV

## How to Run
```bash
python Nidharshani_23057532.py
