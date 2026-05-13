# Spatial-Temporal in vivo Fitness Database for *Vibrio cholerae*

This repository contains an interactive Dash/Plotly web application for exploring spatial-temporal in vivo gene fitness patterns of *Vibrio cholerae*.

The database enables visualization and analysis of genome-wide fitness dynamics across spatial regions and infection timepoints, including clustering analysis, cofitness analysis, similarity profiling, predefined pattern searching, and network-based exploration.

---

# Main Modules

## Home
Overview page for the database, project introduction, and navigation.

## Descriptive Fitness
Visualize spatial-temporal fitness patterns for individual genes or gene sets.

Features include:
- Heatmaps
- Spatial-temporal plots
- Boxplots
- Dotplots
- Downloadable result tables

## Cofitness
Explore cofitness relationships among genes.

Features include:
- Correlation analysis
- Interactive cofitness heatmaps
- Similarity statistics
- Downloadable tables

## Clustering
Cluster genes according to spatial-temporal fitness trajectories.

Features include:
- Adjustable cluster number
- Temporal clustering
- Spatial clustering
- Global 3D clustering
- DTW-based clustering
- Cosine similarity clustering
- Cluster visualization and downloadable tables

## Similarity Profile
Search genes with spatial-temporal fitness profiles similar to a query gene or gene set.

Features include:
- Top-N similar genes
- LOESS smoothing
- Polynomial fitting
- Customizable ranking metrics
- Downloadable result tables

## Network Browser
Explore Gaussian Graphical Model (GGM)-based functional correlation networks.

Features include:
- Interactive gene network visualization
- Partial correlation edges
- Candidate regulator identification
- Subnetwork extraction
- Highlighted functional modules

## Predefined Pattern
Search genes matching user-defined spatial-temporal fitness patterns.

Features include:
- Custom pattern input
- Interactive pattern drawing
- 3D pattern visualization
- Top matched genes
- Downloadable similarity tables

## AI Prediction
Placeholder module for future AI/ML-based prediction models.

Planned directions include:
- Functional prediction
- Latent representation learning
- Environment-aware prediction
- Transfer learning models
- AI-based fitness inference

---

# Repository Structure

```bash
gene_fitness_dash/
├── app/
│   ├── app.py
│   ├── index.py
│   ├── requirements.txt
│   ├── pages/
│   │   ├── home.py
│   │   ├── descriptive_fitness.py
│   │   ├── cofitness.py
│   │   ├── clustering.py
│   │   ├── similarity_profile.py
│   │   ├── network_browser.py
│   │   ├── predefined_pattern.py
│   │   └── ai_prediction.py
│   └── assets/
│
├── data/
│   ├── raw/
│   ├── annotation/
│   ├── clustering/
│   ├── network/
│   ├── gene_sets/
│   └── figure/
│
└── README.md
```

---

# Data Structure

## annotation/
Contains annotation tables including gene names and UniProt annotations.

Example:
- `new_annotations_with_uniprot_names.csv`

## raw/
Contains processed spatial-temporal fitness matrices.

Example:
- `Spatial_temporal_MultiSCAST_FC_final_capping.csv`

## clustering/
Contains precomputed clustering results.

## network/
Contains network edge/node tables and GGM outputs.

## gene_sets/
Contains predefined functional gene lists.

Examples:
- Biotin genes
- Chemotaxis genes
- Flagella genes
- Tcp genes
- O-antigen genes

## figure/
Contains images used in the application.

---

# Running the App Locally

## 1. Activate environment

```bash
conda activate dashdb
```

## 2. Navigate to app directory

```bash
cd gene_fitness_dash/app
```

## 3. Start the Dash app

```bash
python index.py
```

---

# Deployment

Example gunicorn deployment:

```bash
gunicorn app.index:server --bind 0.0.0.0:8050
```

---

# Dependencies

Main dependencies include:

- Dash
- Plotly
- Pandas
- NumPy
- SciPy
- Scikit-learn
- NetworkX
- igraph
- Plotnine
- Flask

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

# Future Directions

Planned future expansions include:

- AI-based functional prediction
- GNN-based regulatory inference
- Dynamic network analysis
- Multi-omics integration
- Interactive 3D visualization
- Cross-condition comparison
- Public online deployment

---

# Citation

If you use this database or codebase, please cite the associated manuscript when available.

---

# Contact

Sida Ye  
Harvard Medical School / HHMI / Brigham and Women's Hospital 
Matthew K. Waldor Lab