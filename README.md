# **Forest Cover Type Prediction System — Robust System & Management**  
### **Workshop 3 — Systems Analysis & Engineering**

This repository contains the refined, production-ready version of the Forest Cover Type Prediction System developed throughout Workshops 1 and 2.  
Workshop 3 consolidates the architecture into a **fault-tolerant, cloud-native, uncertainty-aware ML ecosystem**, aligned with ISO 9000, ISO/IEC 27001, CMMI, and Six Sigma quality frameworks.

The system advances from a research pipeline into an enterprise-grade ML architecture optimized for ecological decision-making in chaotic environmental domains.

---

## 🚀 **Overview**

- Dataset: **15,120 observations**, **56 cartographic features**, **7 forest cover types**  
- Architecture: **4-layer robust system** (replacing the earlier 7-layer pipeline)  
- Accuracy: **95.2%**, with explicit uncertainty quantification  
- Deployment: **Sub-100ms latency**, **99.9% availability**, microservices orchestration  

---

## 🧱 **Key Features**

### **1. Robust 4-Layer Architecture**
- **Data Gate:** ingestion, validation, drift detection, lineage tracking  
- **Feature Engineering:** elevation thresholds, aspect encoding, SMOTEENN, soil consolidation  
- **Model Factory:** ensemble training + Optuna optimization + regression-prevention gates  
- **Inference Core:** ONNX runtime, uncertainty decomposition, threshold-aware confidence  

### **2. Chaos-Aware Prediction**
- Detects critical elevation thresholds at **2400m, 2800m, 3200m**  
- Amplifies uncertainty ×2 within ±50m chaotic zones  
- Separates **aleatoric** and **epistemic** uncertainty  
- Notifies when predictions require ecological manual review  

### **3. Fault Tolerance & Reliability**
- Circuit-breaker validation  
- Automatic rollback if a new model underperforms  
- Redis caching for high-throughput inference  
- Graceful degradation when a model in the ensemble fails  

### **4. Standards-Aligned Quality Framework**
- **ISO 9000:** QMS, documentation, artifact versioning  
- **ISO/IEC 27001:** secure storage, RBAC, encrypted pipelines  
- **CMMI Level 3:** defined, measurable, auditable ML processes  
- **Six Sigma:** DMAIC optimization, drift reduction  
- **CRISP-ML(Q):** monitoring + feedback loops  

---

## 🧰 **Technical Stack**

**Languages & Libraries:**  
Python, NumPy, Pandas, scikit-learn, LightGBM, XGBoost, Optuna, ONNX Runtime  

**MLOps & Infrastructure:**  
FastAPI, Redis, MLflow, PostgreSQL, S3, Kubernetes, NGINX, Grafana, Prometheus  

---

## 🏗️ **Architecture Summary**

### **1. Data Gate**
- Schema validation  
- Data & concept drift detection (PSI, KL-divergence)  
- Stratified train/test split with spatial independence  
- Fault isolation + audit logging  

### **2. Feature Engineering**
- Elevation binning + threshold proximity markers  
- Aspect sine/cosine encoding  
- Soil-type consolidation (40 → 15 classes)  
- SMOTEENN for class imbalance  
- Interaction features (hydrology, fire points, roads)  

### **3. Model Factory**
- Parallel ensemble training (RF + XGB + LGBM)  
- Bayesian optimization  
- Elevation-blocked 5-fold CV  

### **4. Inference Core**
- ONNX-optimized inference  
- Weighted ensemble probabilities  
- Aleatoric + epistemic uncertainty  
- Elevation-aware confidence adjustment  
- Real-time consistency checks across seeds  

---

## ⚠️ **Risk Register (Workshop 3)**

| Risk | Mitigation | Monitoring |
|------|------------|------------|
| Chaotic elevation thresholds | ±50m detection, ×2 uncertainty | Band performance |
| Soil sparsity (73%) | Consolidation, SMOTEENN | Feature-importance drift |
| Geographic brittleness | Elevation-band KPIs | Seasonal variation |
| Data drift | KL/PSI, auto retraining | Alert thresholds |
| Stochastic variance | Seed fixing, blocked CV | CV variance |
| Latency issues | Redis, autoscaling | p95/p99 dashboards |
| Artifact loss | S3 versioning | Integrity checks |
| Security threats | RBAC + encryption | ISMS alerts |

---

## 📅 **Project Management Plan**

### **Roles**
- **Nicolás Martínez** — Architect & ML Engineer  
- **Jean Paul Contreras** — Data Analyst & Feature Engineer  
- **Gabriel Gutiérrez** — Backend & MLOps  
- **Anderson Martínez** — Quality & Risk Manager  

### **Workflow**
- Hybrid Agile–Kanban  
- GitHub Projects + Notion  
- Weekly stand-ups  
- PR-based code review  
- Milestone-driven delivery  

### **Timeline**
- **Week 1:** Architecture refinements  
- **Week 2:** Validation + Feature Engineering  
- **Week 3:** Model training & uncertainty  
- **Week 4:** Deployment + Monitoring  
- **Week 5:** Final documentation  

---

## 📁 **Repository Structure**
~~~
/data                  
/notebooks             
/src
 ├── data_gate         
 ├── features          
 ├── model_factory     
 ├── inference_core    
 ├── monitoring        
 └── api               
/docs

~~~

---

## 👥 **Authors**
- **Nicolás Martínez Pineda**  
- **Anderson Danilo Martínez Bonilla**  
- **Gabriel Esteban Gutiérrez Calderón**  
- **Jean Paul Contreras Talero**

**Universidad Distrital Francisco José de Caldas**




