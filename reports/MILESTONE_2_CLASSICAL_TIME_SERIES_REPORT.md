# Classical Time Series Analysis and Forecasting of Agricultural Plant Diseases — Milestone 2 Evaluation Report

**Course:** CSET 369 — Time Series Analysis & Forecasting  
**Author:** Kartik Lamba  
**Enrollment Number:** S24CSEU0642  
**Email:** lambakartik2006@gmail.com  
**Department:** School of Computer Science, Engineering & Technology (SCSET)  
**Institution:** Bennett University, Greater Noida, India  
**Course Coordinator:** Anurag Choubey  
**Academic Year:** 2026–2027 (Semester 5)  
**Milestone Focus:** Milestone 2 — Classical Time Series Econometrics & Statistical Baselines  
**Format:** IEEE Research Paper Standard (Pages $\le 10$, Exact 4 Pages)  

---

### Abstract
*Fungal and bacterial crop diseases cause severe harvest losses across India, hurting farmer income and national food security. Today, farmers detect disease only after visible leaf damage appears, which is often too late to save the harvest. Milestone 2 of the AgriForecast project establishes a clear, classical statistical baseline to forecast monthly disease outbreaks across 10 major Indian crop diseases. We use an unbroken 11-year monthly dataset (January 2015 to December 2025, 1,320 total records) combining GBIF field observations with NASA satellite weather data. We implement and test nine classical models: Autoregressive (AR), Moving Average (MA), ARMA, ARIMA, Seasonal ARIMA (SARIMA), ARIMAX with weather features, Holt-Winters exponential smoothing, and a simple Seasonal Naive benchmark. Statistical tests (ADF and KPSS) show that raw disease counts need first differencing ($d=1$) to remove seasonal trends. In out-of-sample testing on unseen data, simple linear models performed best: ARMA achieved the lowest test error (MAE: 0.4844, RMSE: 0.5854), followed closely by AR(1) (MAE: 0.4856) and ARIMA(1,1,1) (MAE: 0.4953). Residual checks with the Ljung-Box test confirm that errors are random white noise. However, classical models cannot handle the fact that 71.8% of months have zero cases (often predicting negative disease counts), showing why advanced machine learning and deep learning models are needed in Milestone 3.*

**Index Terms—** Crop disease forecasting, classical time series, Box-Jenkins ARIMA, SARIMAX, Holt-Winters, stationarity testing, Bennett University.

---

### I. Introduction

#### A. Background & Problem Statement
Farming in India is heavily impacted by fungal and bacterial plant diseases. For example, Rice Blast (*Pyricularia oryzae*) and Sheath Blight (*Rhizoctonia solani*) routinely destroy 20% to 45% of rice harvests in affected areas. Wheat Rust threatens northern grain belts, while Potato Late Blight can ruin an entire field in a matter of days. Today, farmers only discover disease after seeing yellowing leaves and dead spots. By that time, fungal spores have already multiplied, making chemical sprays expensive and far less effective. If agricultural departments can forecast disease outbreaks 1 to 6 months in advance, they can issue early warnings, order preventive fungicides, and protect food supplies.

#### B. The Classical Time Series Approach
In Milestone 2, we establish a classical statistical baseline using the Box-Jenkins methodology. Classical time-series models break down disease numbers into three logical parts:
1. **Autoregression (AR):** How strongly today's cases depend on last month's cases.
2. **Moving Average (MA):** How recent unexpected shocks affect the trend.
3. **Seasonality:** How the annual monsoon cycle repeats every 12 months.

#### C. Objectives of Milestone 2
1. Clean and combine 11 years of monthly disease and satellite weather data into a single verified dataset.
2. Check stationarity with standard statistical tests (ADF and KPSS).
3. Fit and compare classical models (AR, MA, ARIMA, SARIMA, Holt-Winters).
4. Highlight why simple linear models struggle when 70%+ of months have zero cases.

---

### II. Literature Review

Classical time series forecasting has evolved through several key milestones:
1. **Box-Jenkins ARIMA:** Box and Jenkins [1] showed that many real-world series depend strongly on their own past values and recent errors.
2. **Seasonal SARIMA:** Box, Jenkins, and Reinsel [2] added seasonal terms to repeat patterns every 12 months, matching India's rainy Kharif and dry Rabi seasons.
3. **Holt-Winters Smoothing:** Holt [3] and Winters [4] created exponential smoothing to update trends and seasonality quickly.
4. **Weather Drivers (The Disease Triangle):** Madden et al. [5] showed that disease outbreaks occur only when active spores, a vulnerable plant, and warm humid weather coincide. ARIMAX models include these weather variables directly as external regressors.

---

### III. Dataset Description & Preprocessing

Our master dataset (`data/disease_timeseries_master.csv`) brings together verified field disease observations from GBIF/EPPO and monthly surface weather data (temperature, rainfall, humidity) from NASA POWER satellites. It contains 1,320 monthly records across 10 crops over 132 unbroken months (January 2015 to December 2025). We predict `documented_occurrence_count`, which averages 1.34 cases per month with a maximum of 74 cases. Exactly 71.82% of months have zero cases due to dry off-season weather.

**Preventing Data Leakage:** To make sure our models never cheat by looking into the future, we split our data chronologically:
- **Training Set (70%):** Jan 2015 to Aug 2022 (92 months).
- **Validation Set (15%):** Sep 2022 to Apr 2024 (20 months).
- **Holdout Test Set (15%):** May 2024 to Dec 2025 (20 months).
We only use weather data from 1 and 2 months ago ($t-1, t-2$) because real-life agricultural officers only have past weather when forecasting upcoming months.

---

### IV. Exploratory Data Analysis & Stationarity Testing

A time series is stationary when its average and variance remain stable over time. If a series has strong trends or wanders randomly, classical models struggle to make reliable predictions. We used two standard statistical tests: the Augmented Dickey-Fuller (ADF) test ($H_0$: unit root present) and the KPSS test ($H_0$: stationary around mean/trend).

| Pathogen Taxon | Raw ADF ($p$-val) | Raw KPSS ($p$-val) | Decision | Differenced ADF ($p$-val) |
| :--- | :---: | :---: | :---: | :---: |
| *Pyricularia oryzae* (Rice Blast) | -1.842 ($p=0.3598$) | 0.8421 ($p=0.0100$) | **Non-Stationary** | **-8.721 ($p<0.0001$)** |
| *Macrophomina phaseolina* | +0.0915 ($p=0.9655$) | 1.1553 ($p=0.0100$) | **Non-Stationary** | **-9.482 ($p<0.0001$)** |
| *Fusarium oxysporum* | -2.110 ($p=0.2403$) | 0.9812 ($p=0.0100$) | **Non-Stationary** | **-7.894 ($p<0.0001$)** |
| *Puccinia striiformis* (Wheat Rust) | -1.954 ($p=0.3069$) | 0.7891 ($p=0.0100$) | **Non-Stationary** | **-8.125 ($p<0.0001$)** |
| *Phytophthora infestans* (Late Blight)| -2.041 ($p=0.2688$) | 0.9124 ($p=0.0100$) | **Non-Stationary** | **-8.451 ($p<0.0001$)** |

*Table I. Stationarity test results showing raw data becomes stationary after first differencing ($d=1$).*

---

### V. Classical Models Explained in Simple Terms

We tested eight classical statistical setups:
1. **$AR(1)$:** Predicts today's cases based on last month's cases: $y_t = c + \phi_1 y_{t-1} + \varepsilon_t$.
2. **$MA(1)$:** Adjusts today's forecast based on last month's unexpected forecast error: $y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1}$.
3. **$ARMA(1,1)$:** Combines both past cases and past error adjustments: $y_t = c + \phi_1 y_{t-1} + \varepsilon_t + \theta_1 \varepsilon_{t-1}$.
4. **$ARIMA(1,1,1)$:** Differences the series once to remove trends, then applies AR and MA.
5. **$SARIMA(1,0,1)(1,0,1)_{12}$:** Repeats patterns every 12 months to match the annual monsoon cycle.
6. **$ARIMAX$ & $SARIMAX$:** Adds external weather variables like last month's temperature and humidity directly into the regression.
7. **Holt-Winters Smoothing:** Automatically tracks level, trend, and seasonal patterns using exponential weights.
8. **Seasonal Naive Baseline:** Simply assumes this month's cases will equal the cases from the exact same month last year.

---

### VI. Model Selection & Checking Residuals

We found the best model orders by looking for the lowest AIC and BIC scores (for ARIMA(1,1,1), $\text{AIC} = 220.4$). Next, we ran the Ljung-Box test on the model errors to make sure no patterns were left behind. The test showed $Q(12) = 14.21$ ($p = 0.221$). Because $p > 0.05$, the errors are pure random white noise, confirming the model captured all predictable patterns.

---

### VII. Results & Comparison on Unseen Data

All models were tested on unseen holdout test data. Table II shows the accuracy ranking across 11 classical models. Simple models like ARMA(1,1) (MAE: 0.4844) and AR(1) (MAE: 0.4856) performed best because they only estimate 1 or 2 parameters and do not overfit. In contrast, complex seasonal models performed worse because monsoon arrival dates shift by several weeks from year to year.

| Rank | Model Architecture | MAE | RMSE | MAPE (%) | sMAPE (%) | $R^2$ | MASE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **ARMA(1, 1)** | **0.4844** | **0.5854** | **59.34%** | **80.44%** | **0.1461** | **0.3918** |
| 2 | AR(1) | 0.4856 | 0.5868 | 63.01% | 80.51% | 0.1052 | 0.3928 |
| 3 | ARIMA(1, 1, 1) | 0.4953 | 0.5976 | 58.37% | 84.63% | 0.1315 | 0.4006 |
| 4 | VAR(2) | 0.5029 | 0.6038 | 68.38% | 80.25% | 0.0158 | 0.4069 |
| 5 | ARIMAX | 0.5112 | 0.6259 | 57.54% | 87.61% | 0.0601 | 0.4135 |
| 6 | SARIMAX | 0.5596 | 0.7391 | 67.61% | 94.09% | -0.3021 | 0.4528 |
| 7 | SARIMA | 0.5607 | 0.7320 | 64.90% | 94.09% | -0.2778 | 0.4536 |
| 8 | Holt-Winters | 0.6406 | 0.8492 | 89.27% | 109.19% | -0.7180 | 0.5182 |
| 9 | Moving Average | 0.6410 | 0.7489 | 67.50% | 91.56% | -0.3602 | 0.5188 |
| 10 | Meta Prophet | 0.6832 | 0.7762 | 75.15% | 86.29% | -0.4770 | 0.5524 |
| 11 | Seasonal Naive | 0.8869 | 1.0205 | 109.08% | 124.36% | -1.5086 | 0.7178 |

*Table II. Performance comparison of classical models on unseen test data.*

---

### VIII. Practical Agricultural Lessons Learned

1. **Monsoon Rain Spikes (Kharif Season):** Rainfall and high humidity in July and August correlate strongly with Rice Blast outbreaks 1 month later ($r = +0.58$). This 1-month delay matches the biological incubation time needed for fungal spores to land, germinate, and produce visible spots on leaves.
2. **Winter Cool Season (Rabi Season):** Wheat Stripe Rust correlates negatively with temperature ($r = -0.42$), meaning it thrives during cold winter months (10°C to 18°C) in northern India.

---

### IX. Why Classical Models are Not Enough (The Need for Milestone 3)

While classical models are simple and fast, they have three major weaknesses when dealing with real-world plant diseases:
1. **Predicting Negative Disease Cases:** Classical models assume data follows a symmetric bell curve. During dry summer months, they often predict absurd negative numbers like -1.5 disease cases.
2. **The 71.8% Zero-Case Challenge:** Over 70% of months have zero cases. Simple linear models struggle to stay flat at zero without creating artificial noise.
3. **Changing Forecast Uncertainty:** Classical models assume forecast error is constant all year. In reality, uncertainty is almost zero in summer and explodes during the rainy monsoon.

**What We Build in Milestone 3:** To fix these weaknesses, Milestone 3 introduces two-stage Hurdle XGBoost (which never predicts negative numbers), deep neural networks (PyTorch LSTM, GRU), self-attention Temporal Transformers (PatchTST), and GARCH volatility cones.

---

### X. Conclusion

Milestone 2 successfully created an honest classical baseline for Indian crop disease forecasting. Statistical tests proved that differencing stabilizes the series. Simple models like ARMA(1,1) and AR(1) performed best among linear methods, keeping average error (MAE) under 0.50 cases. These classical results give us a clear standard to beat when building advanced machine learning and deep learning models in Milestone 3.

---

### XI. References

[1] G. E. P. Box and G. M. Jenkins, *Time Series Analysis: Forecasting and Control*. San Francisco, CA: Holden-Day, 1970.  
[2] G. E. P. Box, G. M. Jenkins, and G. C. Reinsel, *Time Series Analysis: Forecasting and Control*, 4th ed. Hoboken, NJ: John Wiley & Sons, 2008.  
[3] C. C. Holt, "Forecasting seasonals and trends by exponentially weighted moving averages," *Int. J. Forecast.*, vol. 20, no. 1, pp. 5–10, 2004.  
[4] P. R. Winters, "Forecasting sales by exponentially weighted moving averages," *Manage. Sci.*, vol. 6, no. 3, pp. 324–342, 1960.  
[5] L. V. Madden, G. Hughes, and F. van den Bosch, *The Study of Plant Disease Epidemics*. St. Paul, MN: APS Press, 2007.  
[6] S. J. Taylor and B. Letham, "Forecasting at scale," *The American Statistician*, vol. 72, no. 1, pp. 37–45, 2018.  
[7] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne, Australia: OTexts, 2021.  
[8] R. F. Engle, "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation," *Econometrica*, vol. 50, no. 4, pp. 987–1007, 1982.  
[9] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *J. Econometrics*, vol. 31, no. 3, pp. 307–327, 1986.  
[10] J. D. Hamilton, *Time Series Analysis*. Princeton, NJ: Princeton University Press, 1994.  
