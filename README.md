# Forest Cover Type Prediction System

This repository contains the conceptual design and analytical framework for a **chaos-aware machine learning system** developed to classify forest cover types in the **Roosevelt National Forest** dataset from Kaggle.  
The work expands on prior systems analysis to design a **seven-layer architecture** that integrates ecological modeling, uncertainty quantification, and systems engineering principles for sustainable forest management applications.

## Overview

The system processes **15,120 observations** across **56 cartographic features** to classify **seven distinct forest cover types** at a 30m × 30m spatial resolution.  
It explicitly addresses the chaotic and nonlinear dynamics observed in forest ecosystems—such as elevation thresholds and aspect–elevation couplings—through mechanisms of **sensitivity control** and **uncertainty quantification**.

## Key Features

- **Seven-layer pipeline architecture** ensuring modularity, scalability, and reproducibility  
- **Feature engineering** that includes elevation binning, soil-type consolidation, and trigonometric encoding for aspect  
- **Ensemble modeling** combining Random Forest, XGBoost, and LightGBM via weighted voting (expected 95.2% accuracy)  
- **Dual-mode uncertainty quantification** distinguishing aleatoric (data) and epistemic (model) sources  
- **Chaos and sensitivity control** through threshold detection and drift monitoring (Kullback–Leibler divergence, PSI)  
- **MLOps-ready design** with FastAPI for real-time inference, Redis caching, MLflow tracking, and Kubernetes orchestration  

## Technical Stack

- **Language:** Python  
- **Core Libraries:** NumPy, Pandas, scikit-learn, PyTorch, Optuna, PySpark  
- **MLOps & Storage:** MLflow, PostgreSQL, S3, Grafana  
- **APIs & Serving:** FastAPI, Uvicorn, NGINX  
- **Visualization:** React + Plotly  

## Project Structure

```
/data               -> Datasets and preprocessing scripts
/notebooks          -> Exploratory and analysis notebooks
/src
 ├── ingestion      -> Data ingestion and validation
 ├── features       -> Feature engineering modules
 ├── models         -> Model training and ensemble integration
 ├── monitoring     -> Drift and chaos detection
 └── api            -> Deployment and inference endpoints
/docs               -> Workshop reports and architecture diagrams
```

## Authors

- **Nicolás Martínez Pineda**  
- **Anderson Danilo Martínez Bonilla**  
- **Gabriel Esteban Gutiérrez Calderón**  
- **Jean Paul Contreras Talero**  

**Universidad Distrital Francisco José de Caldas**  

## Reference

This work is based on *“Systems Analysis of Kaggle’s Forest Cover Type Prediction: Elements, Relationships, Sensitivity, and Chaos”* (Workshop #2 Report, 2025).
