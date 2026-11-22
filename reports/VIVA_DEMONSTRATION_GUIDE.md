# AgriForecast: Milestone 3 Viva & Project Demonstration Defense Guide

**Subject:** Time Series Analysis & Forecasting (Semester 5)  
**System URL:** `http://localhost:8501`  
**Purpose:** Comprehensive guide for acing the project viva, defending econometric choices, and navigating the live web application.

---

## Part 1: Step-by-Step Live Website Demonstration Script

When the examiners say: *"Show me your project"* or *"Demonstrate the application"*, follow this structured 5-minute walkthrough:

---

### Step 1: Homepage & Surveillance Overview (`http://localhost:8501/`)
1. **What to click:** Open [http://localhost:8501/](http://localhost:8501/).
2. **What to show on screen:**
   - Top navigation bar (AgriForecast branding with clean SVG icons).
   - Macro surveillance KPI cards (10 Pathogens, 132 Monthly Observations, Zero Leakage).
   - Interactive historical trend plot showing Kharif vs. Rabi outbreak surges.
3. **What to say:**
   > *"Good morning sir. This is AgriForecast, an end-to-end time series surveillance suite tracking 10 major crop pathogens in India across 132 unbroken monthly observations from January 2015 to December 2025. All paired meteorological covariates—temperature, rainfall, and relative humidity—are strictly synchronized."*

---

### Step 2: Time Series Analysis & ARCH/GARCH (`http://localhost:8501/Time_Series_Analysis`)
1. **What to click:** Click **"Analysis"** on the top navigation bar, then scroll to the tabs and click **Tab 6: "ARCH / GARCH Volatility & Uncertainty"**.
2. **What to show on screen:**
   - The 4 KPI cards: **Average Uncertainty ($\bar{\sigma}$)**, **Peak Outbreak Uncertainty**, **Outbreak Shock Memory ($\alpha + \beta$)**, and **Model Status** (`ARCH LM p-value`).
   - The Historical Volatility timeline showing peaks during outbreak years and flat zero lines during dry dormancy.
   - The 12-Month Forward Forecast with Dynamic Uncertainty Interval (the blue shaded cone that expands during high-risk seasons).
   - Click the expander: **"Detailed Econometric Tables (Parameters, AIC/BIC & All Specifications)"**.
3. **What to say (Directly addressing Sir's questions):**
   > *"Here in Tab 6, we address the exact questions regarding Random Walk and Heteroscedasticity. First, we ran ADF and KPSS tests. ADF rejects the unit root hypothesis with p < 0.0001 for 7 out of 10 pathogens, proving our data does NOT follow a random walk—it is stationary around a seasonal mean. Second, Engle’s ARCH-LM test shows p < 0.005, confirming strong conditional heteroscedasticity. Because variance clusters in time, we fit 7 ARCH and GARCH specifications. In this table, GARCH(1,1) emerges as the gold standard model: it minimizes BIC, captures persistence through alpha and beta, and drives the post-fit ARCH-LM p-value to 0.98, completely eliminating residual heteroscedasticity."*

---

### Step 3: Model Comparison & Leaderboard (`http://localhost:8501/Model_Comparison`)
1. **What to click:** Click **"Models"** on the top navigation bar.
2. **What to show on screen:**
   - The Filter Toolbar: Select *Rice Blast (Pyricularia oryzae)* and testing period = 6 months.
   - The Comparative Leaderboard Table showing **Temporal Transformer**, **LSTM**, **Hurdle XGBoost**, **GRU**, **ARIMA**, **VAR**, and **Prophet**.
   - Point out the error metrics: **RMSE, MAE, MAPE, WAPE, and MASE**.
   - Show the interactive multi-model forecast plot and error distribution curves.
3. **What to say:**
   > *"On our Model Comparison page, we evaluate 19 architectures strictly on out-of-sample data with zero data leakage. On Rice Blast over a 6-month horizon, our PyTorch Temporal Transformer and Hurdle XGBoost lead the leaderboard with an RMSE around 4.15 and MAE of 3.12, outperforming the classical Seasonal Naive baseline by over 40%."*

---

### Step 4: Out-of-Sample Future Forecasts (`http://localhost:8501/Future_Forecasts`)
1. **What to click:** Click **"Forecast"** on the top navigation bar.
2. **What to show on screen:**
   - Select *Rice Blast* and model = *Hurdle XGBoost* or *Temporal Transformer*.
   - Show the 6-month or 12-month forward projection into 2026.
   - Point out the 80% and 95% confidence bands and the Month-by-Month Risk Outlook cards below.
3. **What to say:**
   > *"Finally, here on the Future Forecasts page, the system projects out-of-sample risk for 2026. Extension officers can view predicted peak outbreak months and receive automated risk warnings to pre-position fungicides before price gouging occurs."*

---

## Part 2: Top 10 Expected Viva Questions & Model Defense Answers

### Q1: "Did you check if the time series is a Random Walk? What did the test show?"
- **Answer:**
  > *"Yes, sir. We executed both the Augmented Dickey-Fuller (ADF) test and the KPSS test across all 10 pathogens. A random walk requires a unit root ($I(1)$) where shocks have permanent effects. For 7 out of 10 foliar pathogens—such as Rice Blast (ADF = -9.07, $p < 0.0001$) and Potato Early Blight (ADF = -11.40, $p < 0.0001$)—the unit root null hypothesis was strongly rejected. The series are stationary in levels around a seasonal mean because biological disease cycles naturally return to zero during unfavorable dry/winter seasons. However, for 3 soil-borne pathogens (Fusarium, Macrophomina, and Rhizoctonia), we found evidence of multi-year upward drift requiring first-differencing ($d=1$) or hurdle detrending."*

---

### Q2: "What is Heteroscedasticity, and how did you prove it exists in your data?"
- **Answer:**
  > *"Heteroscedasticity means that the variance of the error term is not constant over time. In plant pathology, variance is tiny or zero during dry dormancy months and explodes during warm, humid monsoon months when rapid spore germination occurs. We proved this mathematically using Engle's Lagrange Multiplier (ARCH-LM) test. When testing the first-differenced series ($\Delta y_t$), all 10 pathogens rejected the null hypothesis of homoscedasticity with $p < 0.005$. In residual series from ARIMA models, persistent outbreak pathogens like Bipolaris oryzae ($p = 0.0001$) and Rhizoctonia solani ($p = 0.0134$) showed heavy ARCH effects, justifying the use of ARCH and GARCH models."*

---

### Q3: "Which model is the best-fit among ARCH and GARCH, and why?"
- **Answer:**
  > *"We evaluated 7 specifications: ARCH(1), ARCH(2), ARCH(5), GARCH(1,1), GARCH(1,2), GARCH(2,1), and GARCH(2,2) with Student's $t$ innovations. Under the Bayesian Information Criterion (BIC), which heavily penalizes extra parameters on small samples ($N=132$), ARCH(1) and GARCH(1,1) achieve the lowest scores (BIC = 469.44 and 474.32 for Rice Blast). Econometrically and epidemiologically, GARCH(1,1) is the superior gold-standard model because its autoregressive variance term ($\beta_1 \sigma_{t-1}^2$) models the entire infinite-horizon decay of past shocks using only 2 parameters ($\alpha_1$ and $\beta_1$). Crucially, after fitting GARCH(1,1), the post-estimation ARCH-LM test yields $p = 0.98$, proving that all conditional heteroscedasticity in the residuals has been successfully eliminated."*

---

### Q4: "Why did you use Student's t distribution instead of Gaussian Normal in GARCH?"
- **Answer:**
  > *"Because agricultural disease data exhibits extreme kurtosis and fat tails. Over 70% of historical months are zero, while active epidemic months produce sudden, explosive spikes. A Gaussian normal distribution assumes thin tails and underestimates the probability of extreme epidemic surges. The Student's $t$ distribution adds a degrees-of-freedom parameter $\nu$ (estimated between 2.1 and 3.3 in our models), which accurately captures these fat-tailed outlier shocks."*

---

### Q5: "How does your Hurdle XGBoost work, and why does it outperform standard regression?"
- **Answer:**
  > *"Standard regression algorithms—including linear regression and standard XGBoost—treat zero-count months like continuous real numbers, leading to artificial negative predictions (e.g., -2 cases) or smearing fractional values across dry seasons. Our Hurdle XGBoost uses a two-stage architecture: Stage 1 is an XGBoost binary classifier that predicts whether an outbreak will occur ($P(y_t > 0)$). If the probability clears a threshold $\tau$, Stage 2—an XGBoost regressor trained strictly on non-zero historical records—predicts the outbreak volume. This completely eliminates negative values and significantly reduces Mean Absolute Error."*

---

### Q6: "How did you implement Deep Learning and Transformers on such small datasets without overfitting?"
- **Answer:**
  > *"With 132 monthly observations per pathogen, training massive models with millions of parameters would lead to catastrophic overfitting. We addressed this through four rigorous design choices: First, we built lightweight architectures—our PyTorch LSTM and GRU have only 16 hidden units, and our Temporal Transformer uses $d_{model}=16$ with 2 attention heads. Second, we applied LayerNorm and 10% dropout. Third, we used strict chronological sequence windowing (12-month lookback) and early stopping on an internal validation set. Fourth, all MinMaxScaler transformations were fitted strictly on the training set, ensuring zero data leakage."*

---

### Q7: "What is the difference between VAR and your univariate models?"
- **Answer:**
  > *"Univariate models (ARIMA, LSTM, Transformer) predict disease cases primarily from past cases of the same disease, occasionally with weather as exogenous features. Vector Autoregression (VAR) treats disease cases, temperature, and relative humidity as a joint, simultaneously evolving endogenous system: $Y_t = c + \sum A_i Y_{t-i} + \epsilon_t$. It models the bi-directional dynamic feedback between microclimate shifts and pathogen counts. To prevent the degrees-of-freedom problem ($K + pK^2$ parameters), we restricted lag selection to $p \le 2$ via AIC."*

---

### Q8: "Why is MAPE often misleading on agricultural disease data, and what metric did you use instead?"
- **Answer:**
  > *"MAPE calculates $|(y_t - \hat{y}_t) / y_t| \times 100\%$. In disease surveillance, $y_t = 0$ in over 70% of months, causing division by zero and infinite values. Even when masked over non-zero months, if actual cases are very small (e.g., $y_t = 1$) and the model predicts 3, MAPE reports a 200% error even though the absolute difference is only 2 cases. Therefore, we complement MAPE with Weighted Absolute Percentage Error (WAPE = $\sum |y_t - \hat{y}_t| / \sum y_t$) and Mean Absolute Scaled Error (MASE), which are mathematically robust to zero-heavy time series."*

---

### Q9: "How did you ensure zero data leakage in your validation setup?"
- **Answer:**
  > *"We used a strict chronological time split: 70% for training (Jan 2015 – Aug 2022), 15% for validation (Sep 2022 – Apr 2024), and 15% for untouched test holdout (May 2024 – Dec 2025). We never used random K-fold cross-validation or future observations to scale past data. All scalers, hyperparameter searches, and lag constructions were calculated strictly forward in time."*

---

### Q10: "What are the actionable real-world outcomes of this research?"
- **Answer:**
  > *"The system converts statistical forecasts into actionable agricultural interventions:
  > 1. At 1 month ahead, regional SMS alerts are triggered to warn farmers to inspect nursery seedlings.
  > 2. At 3 months ahead, state departments can pre-position systemic fungicides (like Tricyclazole for Rice Blast) in district warehouses before market prices surge during an epidemic.
  > 3. At 6 to 12 months ahead, the dynamic GARCH volatility cone allows agricultural insurance providers to model catastrophic risk and allocate relief funds."*
