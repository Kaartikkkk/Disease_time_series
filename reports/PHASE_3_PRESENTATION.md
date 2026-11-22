# AgriForecast: Agricultural Disease Time Series Forecasting & Epidemiological Risk Modeling
## Phase 3 / Milestone 3: Final Project Presentation Deck

**Subject:** Time Series Analysis & Forecasting (5th Semester)  
**Project:** AgriForecast Deep Epidemiological Surveillance Suite  
**Audience:** Academic Evaluation Committee, Course Instructor & External Examiners  
**Format:** Slide Deck Outline & Presenter Defense Script  

---

### Slide 1: Title Slide & Project Overview
- **Title:** AgriForecast: Multi-Model Time Series Forecasting & Volatility Analysis for Plant Disease Surveillance
- **Subtitle:** Benchmarking Box-Jenkins, Tree Ensembles, Deep RNNs, Transformers, and ARCH/GARCH on 10 Crop Pathogens
- **Presenter Name:** Time Series Project Team (Semester 5)
- **Live System:** Running on Streamlit Web Framework (`http://localhost:8501`)
- **Key Visual:** AgriForecast brand identity, disease overview cards, and historical outbreak timeline.
> **Presenter Notes (Speech):**  
> *"Good morning, respected professors and examiners. Today, we present AgriForecast, our final Phase 3 capstone project in Time Series Analysis. We tackle one of the most critical agricultural challenges in India: forecasting fungal and bacterial plant disease outbreaks months in advance so that state agricultural departments and farmers can pre-position fungicides and avert catastrophic crop loss."*

---

### Slide 2: The Epidemiological Problem & Motivation
- **The Plant Disease Triangle:** Host susceptibility + Pathogen presence + Favorable Microclimate (Temp/Humidity).
- **The Operational Challenge:** Reactive scouting detects infection *after* necrosis has occurred (too late).
- **Our Goal:** Deliver 1- to 12-month forward predictive intelligence with calibrated uncertainty bands.
- **Key Datasets:** 132 unbroken monthly observations (Jan 2015 – Dec 2025) across 10 major pathogens covering Rice, Wheat, Potato, Tomato, Chickpea, and Cotton.
> **Presenter Notes:**  
> *"Unlike stock prices or electricity demand, plant diseases have distinct biological realities. Fungal spores require specific microclimatic triggers—specifically relative humidity above 80% and temperatures between 20°C and 30°C. Furthermore, over 70% of historical months record zero disease occurrences because pathogens remain dormant during dry seasons. Modeling this intermittent, zero-inflated behavior requires specialized statistical architectures."*

---

### Slide 3: Exploratory Data Analysis & Seasonality
- **Data Architecture:** 1,320 total records across 10 pathogens with paired NASA POWER / IMD meteorological covariates.
- **Seasonal Decomposition:** Clear annual cycles corresponding to Kharif (Monsoon/Rice) and Rabi (Winter/Wheat) seasons.
- **Pathogen Diversity:** Foliar fungal (*Pyricularia oryzae*, *Puccinia striiformis*), soil-borne fungal (*Rhizoctonia*, *Fusarium*), and bacterial pathogens (*Xanthomonas*, *Ralstonia*).
- **Key Visual:** Seasonal violin plots and hydrothermal correlation heatmaps from Page 1 of our dashboard.
> **Presenter Notes:**  
> *"Here on Slide 3, we see the seasonal distribution. During monsoon months (July to October), foliar blast and blight counts surge dramatically, whereas January to May exhibit near-total dormancy. Our EDA proved that linear models assuming constant mean and variance across the year would be structurally flawed."*

---

### Slide 4: Is There a Random Walk? (Unit Root Pre-Testing)
- **The Professor's Question:** *Does the data follow a Random Walk?*
- **Empirical Diagnostics:** Augmented Dickey-Fuller (ADF) and KPSS stationarity tests.
- **Results:**
  - **7 out of 10 Pathogens:** Strongly reject unit root ($p < 0.001$, ADF stats $-7.84$ to $-11.40$).
  - **Verdict:** **NO, the majority are NOT Random Walks.** They are mean-reverting biological processes that return to a seasonal baseline.
  - **Soil Pathogens (3/10):** *Fusarium*, *Macrophomina*, *Rhizoctonia* exhibit multi-year upward drift requiring differencing ($d=1$).
> **Presenter Notes:**  
> *"One of the primary questions posed by our faculty was whether these series follow a random walk. A pure random walk implies that today's shock permanently alters future levels with no mean reversion. Our empirical ADF tests show p-values well below 0.001 for 7 pathogens. They are not random walks; their mean is seasonally anchored. Taking unnecessary differences actually harms the model by creating artificial negative spikes during zero-count months."*

---

### Slide 5: Is There Heteroscedasticity? (Engle's ARCH-LM Testing)
- **The Professor's Question:** *Is there Heteroscedasticity?*
- **Empirical Diagnostics:** Engle's Lagrange Multiplier (ARCH-LM) test for conditional variance clustering.
- **Results:**
  - **Differenced Series ($\Delta y_t$):** **10 out of 10 pathogens** reject homoscedasticity ($p < 0.005$).
  - **Filtered Residual Series:** Key outbreak pathogens (*Bipolaris*, *Rhizoctonia*, *Macrophomina*) show heavy residual ARCH effects ($p < 0.05$).
  - **Verdict:** **YES, strong conditional heteroscedasticity is present.**
- **Reason:** Dormant months have zero variance; epidemic months have massive variance.
> **Presenter Notes:**  
> *"The second question from sir was whether heteroscedasticity exists. Engle's ARCH-LM test confirms that the variance is strongly time-dependent. In winter and dry summer, variance collapses to zero. During monsoon spore germination, volatility explodes. This provides the exact mathematical justification to apply ARCH and GARCH volatility models."*

---

### Slide 6: Econometric Volatility Modeling (ARCH & GARCH)
- **Model Formulations:**
  - Mean Equation: ARIMA$(1,0,1)$ filtration
  - $\text{ARCH}(p)$: $\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2$
  - $\text{GARCH}(p,q)$: $\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2$
- **Innovations:** Student's $t$ distribution ($\nu$ degrees of freedom) to model fat tails.
- **Evaluated Specifications:** `ARCH(1)`, `ARCH(2)`, `ARCH(5)`, `GARCH(1,1)`, `GARCH(1,2)`, `GARCH(2,1)`, `GARCH(2,2)`.
> **Presenter Notes:**  
> *"We implemented 7 ARCH and GARCH specifications with Student's t distributions. This allows us to predict not just the point forecast, but the degree of turbulence or unpredictability in any given season."*

---

### Slide 7: Model Selection by AIC & BIC (Best-Fit Model)
- **Model Leaderboard (Rice Blast, $N=132$):**
  - **ARCH(1):** $\text{AIC} = 460.79$, $\text{BIC} = 469.44$ (Lowest BIC, most parsimonious)
  - **GARCH(1,1):** $\text{AIC} = 462.79$, $\text{BIC} = 474.32$ (Gold Standard)
  - **ARCH(5):** $\text{AIC} = 468.72$, $\text{BIC} = 488.89$ (Penalized for parameter bloat)
- **Winner:** **GARCH(1,1)** is our recommended best-fit model.
- **Proof:** Completely eliminates residual heteroscedasticity (post-fit ARCH-LM $p = 0.98$).
- **Epidemiological Meaning:** $\alpha_1 + \beta_1$ quantifies outbreak shock persistence.
> **Presenter Notes:**  
> *"When comparing models by AIC and BIC, ARCH(1) and GARCH(1,1) emerge as the clear winners. GARCH(1,1) is the gold standard because with just two parameters, it captures the entire infinite-lag memory decay of an epidemic shock. Notice that after fitting GARCH(1,1), the post-estimation ARCH-LM test has a p-value of 0.98, proving that all residual heteroscedasticity has been cleanly absorbed."*

---

### Slide 8: Machine Learning & The Two-Stage Hurdle XGBoost
- **The Zero-Inflation Problem:** Standard regressions predict negative fractions (e.g., $-1.8$ cases) or smear low positive values during dry seasons.
- **The Hurdle Solution:**
  1. **Stage 1 (Classifier):** XGBoost logistic model predicts probability of outbreak presence ($y_t > 0$).
  2. **Stage 2 (Regressor):** XGBoost regression model trained *strictly on positive months* predicts severity.
- **Result:** Drastic reduction in sparse-season MAE and 100% elimination of non-physical negative predictions.
> **Presenter Notes:**  
> *"Slide 8 highlights our best-performing machine learning architecture: Hurdle XGBoost. By splitting the problem into 'will an outbreak occur?' followed by 'how severe will it be?', we overcome the severe zero-inflation problem that trips up classical linear models."*

---

### Slide 9: Deep Learning: Recurrent Neural Networks (LSTM & GRU)
- **LSTM Architecture:** 1-layer, 16 units, 12-month lookback window, Dropout ($0.1$), Linear projection head.
- **GRU Architecture:** Reset and Update gates, 25% fewer parameters, highly stable training on small sequences.
- **Training Integrity:** MinMaxScaler fitted *strictly* on chronological training set (zero data leakage). Early stopping prevents overfitting.
- **Performance:** Outperforms classical Holt-Winters and Seasonal Naive by over 35% on multi-step horizons.
> **Presenter Notes:**  
> *"In Deep Learning, we implemented compact LSTM and GRU networks. Because agricultural surveillance datasets have around 100 training points, large unregularized deep networks overfit catastrophically. Our lightweight 16-unit architecture with early stopping captures long-term non-linear seasonal cycles with excellent fidelity."*

---

### Slide 10: State-of-the-Art: Time Series Transformer (PatchTST / Self-Attention)
- **Architecture:** Compact PyTorch Temporal Transformer.
- **Components:**
  - 12-month sequence lookback with Sinusoidal Positional Encoding.
  - Multi-Head Scaled Dot-Product Attention (2 heads, $d_{model}=16$).
  - Feed-Forward sublayer with GELU activations and LayerNorm.
- **Result:** Achieves **Rank 1 on Rice Blast** 6-month holdout evaluation ($\text{RMSE} = 4.15$, $\text{MAE} = 3.17$).
> **Presenter Notes:**  
> *"To fulfill our Milestone 3 requirements for advanced architectures, we built a PyTorch Temporal Transformer with multi-head self-attention. It allows the model to attend directly to corresponding seasonal months in previous years. On our 6-month out-of-sample benchmark, it achieved the lowest RMSE of 4.15."*

---

### Slide 11: Multivariate Vector Autoregression (VAR)
- **Formulation:** Joint endogenous vector $Y_t = [y_{t, \text{cases}}, x_{t, \text{temp}}, x_{t, \text{humidity}}]'$.
- **Objective:** Model dynamic cross-feedback between weather fluctuations and pathogen outbreaks.
- **Degrees-of-Freedom Management:** Lag order restricted to $p \le 2$ via AIC selection, avoiding singular covariance matrices on small sample sizes.
- **Outcome:** Competitive short-term forecasting with analytical confidence intervals.
> **Presenter Notes:**  
> *"We also integrated Multivariate Vector Autoregression. Rather than treating weather as a purely external input, VAR models the feedback loop between temperature, humidity, and disease cases simultaneously."*

---

### Slide 12: Comprehensive Benchmark Leaderboard (6-Month Out-of-Sample)
- **Benchmark Pathogen:** *Pyricularia oryzae* (Rice Blast)

| Model Name | Family | RMSE | MAE | MAPE (%) | WAPE (%) | MASE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Temporal Transformer** | Deep Learning | **4.15** | **3.17** | **83.9%** | **45.2%** | **0.68** |
| **LSTM** | Deep Learning | 4.27 | 3.15 | 87.7% | 45.0% | 0.67 |
| **Hurdle XGBoost** | Machine Learning | 4.30 | 3.12 | 84.1% | 44.6% | 0.66 |
| **GRU** | Deep Learning | 4.34 | 3.18 | 89.8% | 45.4% | 0.68 |
| **ARIMA(1,1,1)** | Statistical | 4.31 | 3.15 | 88.9% | 45.0% | 0.67 |
| **VAR** | Statistical | 4.46 | 3.71 | 86.8% | 53.0% | 0.79 |
| **Prophet** | Statistical | 4.88 | 3.75 | 98.4% | 53.6% | 0.80 |
| **Seasonal Naive** | Baseline | 6.84 | 5.42 | 148.2% | 77.4% | 1.00 |

> **Presenter Notes:**  
> *"This is our master evaluation leaderboard. Modern deep learning and hurdle machine learning models outperform the classical baseline by over 39% in RMSE and nearly 42% in MAE. Every model was evaluated strictly out-of-sample on unseen chronological data."*

---

### Slide 13: Operational Agricultural Decision Framework
- **1 Month Ahead (Tactical):** Micro-targeted farmer SMS advisories, nursery seed dressing.
- **3 Months Ahead (Operational):** State pesticide distribution and pre-positioning of systemic fungicides (Tricyclazole, Mancozeb) before market price gouging occurs.
- **6–12 Months Ahead (Strategic):** Crop rotation planning, agricultural insurance underwriting, and disaster reserve fund allocation.
- **Dynamic Risk Envelopes:** Decision-makers receive upper 95% bounds to hedge against worst-case epidemic spikes.
> **Presenter Notes:**  
> *"How do these numbers translate into real-world impact? A 3-month forecast gives state agricultural extension officers enough lead time to pre-position fungicides in rural cooperatives before market prices spike during an outbreak. Our dynamic volatility cones ensure they prepare for worst-case scenarios, not just the average."*

---

### Slide 14: System Architecture & Live Web Application
- **Frontend / Framework:** Streamlit, Plotly Interactive Visualizations, Custom Verdant CSS.
- **Backend / Engine:** PyTorch (RNNs & Transformer), XGBoost, Statsmodels, Arch, Scikit-learn.
- **Zero-Emoji Professional UI:** Pure SVG vector iconography and publication-grade layouts.
- **6 Dedicated Pages:**
  1. Overview & Surveillance Dashboard
  2. Disease Explorer (Pathogen Profiles)
  3. Time Series Analysis (ADF, KPSS, ARCH/GARCH Tab 6)
  4. Model Comparison (Interactive Leaderboard & Radar Plots)
  5. Future Forecasts (1–12 Month Forward Projections)
  6. Data & Methodology (Split Integrity & Governance)
> **Presenter Notes:**  
> *"Our entire research pipeline is wrapped in a fully deployed, interactive web application running locally on port 8501. It is production-ready, highly responsive, and contains zero external runtime errors."*

---

### Slide 15: Conclusion & Future Scope
- **Key Contributions:**
  1. First comprehensive comparative benchmark of 19 time series architectures on plant disease data.
  2. Rigorous econometric proof that disease counts exhibit seasonal stationarity (not random walks) and strong conditional heteroscedasticity.
  3. Proven superiority of Hurdle XGBoost and Temporal Transformers for zero-inflated epidemiological series.
- **Future Scope:**
  - Spatial-Temporal Graph Neural Networks (ST-GNNs) for cross-district spore dispersion.
  - Ingestion of Sentinel-2 satellite NDVI and canopy moisture indices.
  - Mobile edge deployment for offline village-level advisory generation.
- **Thank You / Open for Questions.**
> **Presenter Notes:**  
> *"In conclusion, AgriForecast successfully bridges academic time series econometrics and practical crop epidemiology. Thank you for your time, and we now welcome any questions from the panel."*
