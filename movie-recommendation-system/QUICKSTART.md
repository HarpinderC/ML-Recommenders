# ⚡ Quick Start Guide

**Get the movie recommender running in 5 minutes!**

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB free disk space

### Step 1: Clone Repository
```bash
git clone https://github.com/HarpinderC/movie-recommender-system.git
cd movie-recommender-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- `scikit-surprise` - Collaborative filtering
- `streamlit` - Web app framework
- `plotly` - Interactive visualizations
- `pandas`, `numpy` - Data processing

---

## 🎬 Running the App

### Launch Streamlit
```bash
streamlit run app/app.py
```

The app will automatically open in your browser at `http://localhost:8501`

**Can't find the browser?** Manually navigate to the URL shown in the terminal.

---

## 🎯 Using the App

### Tab 1: Get Recommendations

**🎬 Similar Movies**
1. Select a movie from the dropdown (e.g., "Toy Story")
2. Click "Find Similar Movies"
3. See 10 genre-similar recommendations with scores

**👤 Personalized Picks**
1. Enter a User ID (1-943)
2. Adjust the model balance slider
3. Click "Get My Recommendations"
4. See top 10 personalized suggestions

### Tab 2: Model Comparison
- View performance metrics for all 12 models
- Interactive charts showing development progress
- Detailed results table

### Tab 3: About
- Project overview
- Technical details
- Model descriptions

---

## 📊 Exploring the Notebooks

### View Development Journey
```bash
# Navigate to notebooks directory
cd notebooks

# Open in Jupyter
jupyter notebook
```

**Notebooks in order:**
1. `01_data_loading_eda_baseline.ipynb` - Data exploration + baselines
2. `02_collaborative_filtering.ipynb` - User & Item-Based CF
3. `03_matrix_factorization.ipynb` - SVD, SVD++, NMF
4. `04_content_hybrid_coldstart.ipynb` - Content-based + Hybrid

---

## 🛠️ Common Issues

### Issue: "ModuleNotFoundError: No module named 'surprise'"
**Solution:**
```bash
pip install scikit-surprise
```

### Issue: "Port 8501 is already in use"
**Solution:**
```bash
streamlit run app/app.py --server.port 8502
```

### Issue: "FileNotFoundError: movies_metadata.csv"
**Solution:** Make sure you're running from the project root directory:
```bash
cd movie-recommender-system
streamlit run app/app.py
```

---

## 📝 Quick Results Summary

| Model | RMSE | Improvement |
|-------|------|-------------|
| Random Baseline | 1.8882 | - |
| Popularity | 1.0210 | 45.9% |
| Item-Based CF | 0.9402 | 50.2% |
| **SVD++ (Best)** | **0.9200** | **51.3%** |

---

## 🎓 What You're Looking At

**This project demonstrates:**
- ✅ 12 recommendation algorithms evaluated
- ✅ Collaborative, content-based, and hybrid approaches
- ✅ Cold start handling for new users/items
- ✅ Production-ready deployment with Streamlit

**Key Insight:** SVD++ (matrix factorization with implicit feedback) outperformed all other approaches, achieving 0.9200 RMSE—a 51.3% improvement over random baseline.

---

## 📚 Want More Details?

**See [README.md](README.md) for:**
- Complete technical deep dive
- Methodology explanations
- Model comparison details
- Future improvements

---

## 📞 Need Help?

**Found a bug or have questions?**
- Open an issue on [GitHub](https://github.com/HarpinderC/movie-recommender-system/issues)
- Email: aekas142@gmail.com

---

**Happy recommending! 🎬**
