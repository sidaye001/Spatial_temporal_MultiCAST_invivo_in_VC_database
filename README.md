# Spatial-Temporal Gene Fitness Database in vivo for Vibrio cholerae

This repository contains a Dash/Plotly web database for exploring spatial-temporal in vivo gene fitness patterns of *Vibrio cholerae*.

## Main modules

- Home
- Descriptive fitness
- Cofitness
- Clustering
- Similarity Profile
- Network Browser
- AI Prediction placeholder

## Local run

```bash
cd gene_fitness_dash
conda activate dashdb
python app/index.pygene_fitness_dash/
└── app/
    ├── app.py
    ├── index.py
    ├── requirements.txt
    ├── pages/
    │   ├── home.py
    │   ├── gene_search.py
    │   ├── similarity_profile.py
    │   ├── network_browser.py
    │   └── downloads.py
    └── assets/
        └── style.css

###To start the interactive app###
cd app
conda activate dashdb
python index.py

###To start the interactive app###
 
