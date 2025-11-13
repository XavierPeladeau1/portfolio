# Portfolio

This repository contains projects showcasing data science, cloud engineering, and web development skills.
By: Xavier Péladeau

## Projects

### Data Science

#### [Forecasting Interest Rate Fluctuations](Forecasting%20Interest%20Rate%20Fluctuations/)
Multi-horizon forecasting of US 10-year Treasury yields using deep learning and statistical models. Initial work done as part of the **Défi IA 2025 competition** organized by Caisse de Dépôt et Placement du Québec (CDPQ), which my team won in January 2025.

**Tech Stack**: PyTorch (LSTM, Transformers), Darts time series library, TFT, N-HiTS

**Key Features**:
- Multi-step forecasting (1 to 36 months ahead)
- Probabilistic outputs with uncertainty intervals
- Walk-forward validation with baseline comparisons
- Public data via FRED API

#### [Forecasting Electricity Consumption](Forecasting%20Electricity%20Consumption/)
Time series forecasting of electricity consumption using classical and deep learning approaches for Hydro-Québec data.

**Tech Stack**: SARIMAX, Temporal Convolutional Networks (TCN), RNN

**Models**: ARIMA/SARIMAX for statistical modeling, TCN and RNN for neural network-based forecasting

### Cloud Engineering

#### [Prefect](Prefect/)
Production-ready, self-hosted Prefect workflow orchestration platform deployed on Google Cloud Platform with Infrastructure-as-Code provisioning and automated CI/CD.

**Tech Stack**: GCP (Compute Engine, Cloud Run, Cloud Build), Terraform, Docker, Prefect

**Key Features**:
- Complete infrastructure provisioning with Terraform
- VPC networking with private IP addressing and firewall rules
- GitOps CI/CD pipeline with automated Docker builds
- Multi-service Docker Compose setup (PostgreSQL, Redis, Prefect server/worker)
- Serverless workflow execution on Cloud Run

### Web Development

#### [Géolocalisation](Géolocalisation/)
Vanilla JavaScript application for visualizing mobile device location data with interactive maps and animations.

**Tech Stack**: Vanilla JavaScript, AngularJS, Leaflet/Maps API

**Features**: Interactive map visualization, geolocation data parsing, custom animations and styling