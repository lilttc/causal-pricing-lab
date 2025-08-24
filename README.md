# Causal Pricing Lab

Demand forecasting and price elasticity estimation using **Double Machine Learning (DML)** and transaction-level e-commerce data.

This project explores how to quantify **own-price** and **cross-price elasticities** for electronic products, simulate pricing strategies, and forecast demand under different scenarios.  
The workflow is inspired by practical challenges in retail and refurbished electronics markets, where **price sensitivity is high** and **portfolio effects (cannibalization, substitution) matter**.

---

## 📌 Project Highlights

- **EDA & Demand Curves** – visual exploration of sales trends, seasonality, and revenue drivers.  
- **Elasticity Estimation**  
  - Baseline log-log OLS for own-price elasticity.  
  - **Double Machine Learning (DML)** to remove confounding bias (controls: seasonality, geography, etc.).  
- **Cross-Price Elasticities** – substitution & complementarity across products.  
- **Pricing Simulation**  
  - Apply Lerner’s condition for optimal pricing.  
  - Portfolio optimization & cannibalization scenarios.  
- **Demand Forecasting** – ARIMA / Prophet / XGBoost models for short-term forecasts, combined with elasticity estimates for pricing “what-ifs.”  

---

## 📂 Repository Structure

```

causal-pricing-lab/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ pyproject.toml / environment.yml # reproducible environment (optional)
├─ .gitignore
├─ configs/
│ ├─ params.yaml
│ └─ paths.yaml
├─ data/ (gitignored)
│ ├─ raw/ # monthly CSVs (2019)
│ ├─ interim/ # cleaned parquet
│ └─ processed/ # feature-engineered tables
├─ notebooks/
│ ├─ 01_data_preparation.ipynb
│ ├─ 02_exploratory_analysis.ipynb
│ ├─ 03_elasticity_baseline.ipynb
│ ├─ 04_elasticity_dml.ipynb
│ ├─ 05_cross_price_elasticity.ipynb
│ └─ 06_pricing_simulation.ipynb
├─ src/causal_pricing/
│ ├─ __init__.py
│ ├─ cleaning.py
│ ├─ features.py
│ ├─ viz.py
│ ├─ models/
│ │ ├─ elasticity_baseline.py
│ │ ├─ elasticity_dml.py
│ │ ├─ cross_price.py
│ │ └─ forecasting.py
│ └─ pricing.py
└─ reports/
├─ figures/
└─ slides/pricing_insights.pdf

```
---

## ⚙️ Setup

Clone the repo:

```bash
git clone https://github.com/lilttc/causal-pricing-lab.git
cd causal-pricing-lab
```

Install dependencies (choose one):

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Mac/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install package itself (editable mode)
pip install -e .

```

🚀 How to Run

1. Prepare data
Place the monthly sales CSVs into data/raw/.

2. Run notebooks sequentially (notebooks/01... → 06...)
Each notebook is self-contained with explanations and visuals.

3. Optional automation
Key steps (data prep, elasticity estimation, pricing simulation) are wrapped in scripts under scripts//.

📊 Example Outputs

- Elasticity Estimates
- Cross-Price Heatmap
- Pricing Scenarios
Simulations of ±10% price changes on top-selling products with revenue/margin trade-offs.

📚 Methods
- Econometrics: log-log OLS, Lerner’s condition.
- Causal ML: Double Machine Learning (doubleml, econml).
- Forecasting: Prophet, ARIMA, XGBoost.
- Visualization: seaborn, matplotlib, plotly.

⚠️ Limitations
- Dataset is synthetic Kaggle e-commerce sales (electronics, 2019).
- Elasticities are illustrative, not business-ready.
- No competitor pricing or marketing spend included
- Transferable methodology: can be applied to real-world domains (e.g., smartphones, fashion, groceries).

🙌 Acknowledgments

- Kaggle dataset: [Sales Dataset of Ecommerce (Electronic Products)](https://www.kaggle.com/datasets/deepanshuverma0154/sales-dataset-of-ecommerce-electronic-products?resource=download).
- References: Chernozhukov et al. (2018) on Double Machine Learning.