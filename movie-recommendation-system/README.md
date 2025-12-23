# Movie Recommendation System

A collaborative filtering-based movie recommendation system using the MovieLens 100K dataset.

## Overview

This project implements various recommendation algorithms including:
- Baseline models (random, global average, popularity-based)
- Collaborative filtering (user-based and item-based)
- Matrix factorization techniques
- Content-based filtering
- Hybrid approaches

## Dataset Statistics

MovieLens 100K dataset:
- **Total Ratings:** 100,000
- **Users:** 943
- **Movies:** 1,682
- **Rating Scale:** 1-5 stars
- **Sparsity:** 93.7%
- **Average Rating:** 3.53

Genre Distribution:
- Drama: 725 movies
- Comedy: 505 movies
- Action: 251 movies
- Thriller: 251 movies
- Romance: 247 movies

## Project Structure

```
Movie_Recommendation_System/
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks
├── results/               # Model results and visualizations
│   └── visualizations/    # Charts and plots
└── models/                # Saved models
```

## Installation

```bash
pip install -r requirements.txt
```

## Status

Dataset loaded and explored. EDA complete with visualizations.
