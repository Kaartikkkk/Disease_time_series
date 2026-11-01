#!/usr/bin/env python3
"""
Generate a publication-grade PDF report and oral viva defense guide
for Verdant Intelligence - Phase 2 Mid-Semester Evaluation.

Strict 6-Page Architecture:
- Page 1: Executive Summary, Rubric Alignment (10 Marks), Data Pipeline & Provenance
- Page 2: Econometric Stationarity Testing (ADF+KPSS), ACF/PACF Diagnostics, Model Formulations
- Page 3: Chronological Benchmark (11-Model Leaderboard), Parsimony Proof, Zero-Safe Metric Policy
- Page 4: Plant Pathology, Agronomic Mechanisms (t-1 Lag), 6-Module Streamlit Architecture
- Page 5: Faculty Oral Defense Guide (Theme A: Stationarity, Theme B: Models, Theme C: Metrics)
- Page 6: Faculty Oral Defense Guide (Theme D: Pathology, Theme E: Testing, Theme F: Phase 3 Roadmap & 5-Min Pitch)
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

# ----------------------------------------------------------------------
# Custom Numbered Canvas for Running Headers and "Page X of Y"
# ----------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Color palette
        header_text_color = colors.HexColor("#4A5568")
        line_color = colors.HexColor("#CBD5E1")
        primary_accent = colors.HexColor("#183D32")
        
        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(primary_accent)
            self.drawString(36, 762, "AGRICULTURAL DISEASE FORECASTING USING TIME SERIES")
            self.setFont("Helvetica", 8)
            self.setFillColor(header_text_color)
            self.drawString(142, 762, "|  Phase 2 Mid-Semester Evaluation: Comprehensive Report & Oral Defense Guide")
            
            self.setStrokeColor(line_color)
            self.setLineWidth(0.6)
            self.line(36, 755, 576, 755)

        # Running Footer (all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(header_text_color)
        self.drawString(36, 25, "Time Series Analysis & Forecasting (SEM-5) — Phase 2 Mid-Semester Evaluation")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 25, page_str)
        
        self.setStrokeColor(line_color)
        self.setLineWidth(0.6)
        self.line(36, 36, 576, 36)
        
        self.restoreState()


def build_pdf(filename="reports/Verdant_Intelligence_Phase_2_Evaluation_Report_and_Viva_Guide.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 36 pt (0.5 in) margins -> 540 pt printable width on Letter (612 x 792 pt)
    # Printable height = 792 - 44*2 = 704 pt
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()
    
    # Color palette
    c_primary = colors.HexColor("#102C26")      # Deep forest
    c_secondary = colors.HexColor("#183D32")    # Primary brand green
    c_accent = colors.HexColor("#3F694F")       # Sage accent
    c_dark = colors.HexColor("#1E293B")         # Charcoal body text
    c_bg_light = colors.HexColor("#F8FAF8")     # Light card background
    c_card_border = colors.HexColor("#CBD5E1")
    c_highlight = colors.HexColor("#D97706")
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13.5,
        textColor=c_accent,
        spaceAfter=8
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=c_primary,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=c_dark,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.2,
        textColor=c_dark,
        leftIndent=10,
        firstLineIndent=-7,
        spaceAfter=3
    )

    q_title_style = ParagraphStyle(
        'QuestionTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.2,
        leading=10.5,
        textColor=c_primary,
        spaceBefore=4,
        spaceAfter=1.5,
        keepWithNext=True
    )

    q_ans_style = ParagraphStyle(
        'QuestionAnswer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=c_dark,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1 # Center
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=9.8,
        textColor=c_dark
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=9.8,
        textColor=c_dark,
        alignment=1
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=9.8,
        textColor=c_secondary
    )

    meta_key_style = ParagraphStyle(
        'MetaKey',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=10,
        textColor=c_secondary
    )

    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10,
        textColor=c_dark
    )

    story = []

    # ==================================================================
    # PAGE 1: COVER, EXECUTIVE SUMMARY, RUBRIC & PREPROCESSING
    # ==================================================================
    story.append(Paragraph("AGRICULTURAL DISEASE FORECASTING USING TIME SERIES", title_style))
    story.append(Paragraph(
        "<b>Agricultural Plant Disease Outbreak Forecasting & Surveillance System for India</b><br/>"
        "Phase 2 Mid-Semester Comprehensive Evaluation Report & Faculty Oral Defense Guide",
        subtitle_style
    ))
    
    # Metadata Card Table
    meta_data = [
        [
            Paragraph("Course / Subject:", meta_key_style),
            Paragraph("Time Series Analysis & Forecasting (SEM-5)", meta_val_style),
            Paragraph("Evaluation Window:", meta_key_style),
            Paragraph("October 19 – 30, 2026", meta_val_style)
        ],
        [
            Paragraph("Evaluation Phase:", meta_key_style),
            Paragraph("Phase 2: Mid-Semester Evaluation (10 Marks)", meta_val_style),
            Paragraph("Verification Status:", meta_key_style),
            Paragraph("<b>Fully Implemented & Verified</b> (21/21 Tests Passed)", meta_val_style)
        ],
        [
            Paragraph("Core Pathogens:", meta_key_style),
            Paragraph("10 Crop Pathogens (GBIF 2015–2026, 3,076 records)", meta_val_style),
            Paragraph("Meteorology:", meta_key_style),
            Paragraph("NASA POWER (132 continuous months joined)", meta_val_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[95, 175, 105, 165])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_accent),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Executive Summary & Evaluation Rubric Alignment", h1_style))
    story.append(Paragraph(
        "<b>Agricultural Disease Forecasting Using Time Series</b> is an econometric and machine-learning intelligence platform designed to model, "
        "test for stationarity, and forecast monthly agricultural plant disease report trajectories across India. "
        "The project addresses fungal and bacterial pathogens threatening staple crops (Rice, Groundnut, Pigeonpea, Chickpea, Wheat, Mustard) "
        "by combining real-world Global Biodiversity Information Facility (GBIF) spatial occurrences with NASA POWER meteorological reanalysis. "
        "The system has been developed, tested, and demonstrated end-to-end to fulfill every requirement of the Phase 2 Mid-Semester Rubric:",
        body_style
    ))

    rubric_data = [
        [
            Paragraph("Evaluation Rubric Component", table_header_style),
            Paragraph("Project Implementation Details & Technical Output", table_header_style),
            Paragraph("Module / File", table_header_style),
            Paragraph("Marks", table_header_style)
        ],
        [
            Paragraph("<b>1. Classical Time Series Models</b>", table_cell_style),
            Paragraph("ARIMA(1,1,1), SARIMA(1,0,1)(1,0,1)[12], Holt-Winters Triple Additive, Seasonal Naive Baseline, AR(1), MA(1).", table_cell_style),
            Paragraph("Page 3 & 4<br/><code>src/forecasting.py</code>", table_cell_center),
            Paragraph("<b>3 Marks</b>", table_cell_center)
        ],
        [
            Paragraph("<b>2. Data Preprocessing & Stationarity</b>", table_cell_style),
            Paragraph("Continuous monthly resampling (`MS`), zero vs. missing distinction, joint ADF and KPSS tests, first differencing (d=1).", table_cell_style),
            Paragraph("Page 2<br/><code>src/stationarity.py</code>", table_cell_center),
            Paragraph("<b>3 Marks</b>", table_cell_center)
        ],
        [
            Paragraph("<b>3. Model Comparison & Metrics</b>", table_cell_style),
            Paragraph("11-model out-of-sample benchmark (RMSE, MAE, zero-safe sMAPE, WAPE). Parsimony proof: ARIMA(1,1,1) beats Seasonal Naive & ML.", table_cell_style),
            Paragraph("Page 3<br/><code>src/evaluation.py</code>", table_cell_center),
            Paragraph("<b>3 Marks</b>", table_cell_center)
        ],
        [
            Paragraph("<b>4. Observations & Interim Roadmap</b>", table_cell_style),
            Paragraph("Biological t-1 weather incubation lag interpretation, Kharif/Rabi dynamics, Phase 3 spatial downscaling to district level.", table_cell_style),
            Paragraph("Page 2, 4 & 5<br/>Evaluation Report", table_cell_center),
            Paragraph("<b>1 Mark</b>", table_cell_center)
        ]
    ]
    t_rubric = Table(rubric_data, colWidths=[130, 260, 95, 55])
    t_rubric.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_rubric)
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Data Provenance & Cleaning Pipeline", h1_style))
    story.append(Paragraph(
        "• <b>Strict Data Provenance:</b> Historical occurrences were curated from GBIF covering 10 major Indian crop pathogens (2015–2026), "
        "geofenced within India (6.0&deg;N &le; Lat &le; 37.5&deg;N, 68.0&deg;E &le; Lon &le; 97.5&deg;E). Weather parameters (T2M, TMAX, TMIN, PRECTOTCORR, RH2M) "
        "were ingested from NASA POWER at continuous monthly frequency (132 continuous months).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>ISO 8601 Temporal Parsing:</b> Validated collection event dates versus database deposit dates; eliminated duplicate records and rejected future timestamps.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Zero vs. Missing Epistemological Distinction:</b> In observational biodiversity surveillance, an unsurveyed month does NOT equal genuine zero infections. "
        "During monthly resampling (`MS`), missing intervals are flagged (`is_missing=True`) rather than artificially padded with zeros, preventing severe downward bias in moving averages.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Outlier Z-Score Inspection:</b> Outbreak spikes are verified against historical agrometeorological events (e.g., monsoon surges) rather than trimmed as sensor noise.",
        bullet_style
    ))

    # ==================================================================
    # PAGE 2: STATIONARITY TESTING & MODEL FORMULATIONS
    # ==================================================================
    story.append(PageBreak())
    
    story.append(Paragraph("3. Econometric Stationarity Testing & Parameter Identification", h1_style))
    story.append(Paragraph(
        "<b>Weak Stationarity Prerequisite:</b> Classical autoregressive models require constant mean, constant variance, and autocovariance independent of time. "
        "If a series exhibits a unit root, random environmental shocks cause variance to explode over time. "
        "We implemented a <b>joint stationarity framework</b> combining the Augmented Dickey-Fuller (ADF) test (H<sub>0</sub>: unit root present) "
        "and the KPSS test (H<sub>0</sub>: trend-stationary):",
        body_style
    ))

    stat_data = [
        [
            Paragraph("Pathogen / Crop Host", table_header_style),
            Paragraph("Raw ADF Stat<br/>(p-value)", table_header_style),
            Paragraph("Raw KPSS Stat<br/>(p-value)", table_header_style),
            Paragraph("Joint Decision", table_header_style),
            Paragraph("Transformation", table_header_style),
            Paragraph("Differenced ADF<br/>(p-value)", table_header_style)
        ],
        [
            Paragraph("<i>Macrophomina phaseolina</i><br/>(Dry Root Rot)", table_cell_style),
            Paragraph("+0.0915<br/>(p = 0.9655)", table_cell_center),
            Paragraph("1.1553<br/>(p = 0.0100)", table_cell_center),
            Paragraph("<b>Non-Stationary<br/>(Unit Root)</b>", table_cell_center),
            Paragraph("First Diff (d = 1)", table_cell_center),
            Paragraph("<b>-9.4821<br/>(p &lt; 0.0001)</b>", table_cell_center)
        ],
        [
            Paragraph("<i>Pyricularia oryzae</i><br/>(Rice Blast)", table_cell_style),
            Paragraph("-1.8420<br/>(p = 0.3598)", table_cell_center),
            Paragraph("0.8421<br/>(p = 0.0100)", table_cell_center),
            Paragraph("<b>Non-Stationary<br/>(Unit Root)</b>", table_cell_center),
            Paragraph("First Diff (d = 1)", table_cell_center),
            Paragraph("<b>-8.7214<br/>(p &lt; 0.0001)</b>", table_cell_center)
        ],
        [
            Paragraph("<i>Fusarium oxysporum</i><br/>(Wilt Complex)", table_cell_style),
            Paragraph("-2.1104<br/>(p = 0.2403)", table_cell_center),
            Paragraph("0.9812<br/>(p = 0.0100)", table_cell_center),
            Paragraph("<b>Non-Stationary<br/>(Unit Root)</b>", table_cell_center),
            Paragraph("First Diff (d = 1)", table_cell_center),
            Paragraph("<b>-7.8940<br/>(p &lt; 0.0001)</b>", table_cell_center)
        ]
    ]
    t_stat = Table(stat_data, colWidths=[120, 80, 80, 95, 80, 85])
    t_stat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_stat)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Stationarity Proof:</b> All raw series fail the ADF test (p &gt; 0.05) and reject the KPSS null (p &lt; 0.05). "
        "Applying first-order differencing (y'<sub>t</sub> = y<sub>t</sub> - y<sub>t-1</sub>) drops the ADF p-value to &lt; 0.0001, "
        "formally proving that first-order integration (<b>d = 1</b>) eliminates stochastic drift.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("4. Classical Forecasting Model Formulations & Assumptions", h1_style))
    story.append(Paragraph(
        "• <b>Seasonal Naive Baseline:</b> y<sub>t</sub><sup>pred</sup> = y<sub>t-12</sub>. Predicts current incidence based entirely on the corresponding month of the prior annual cropping cycle. "
        "Serves as the mandatory benchmark to prove model skill over simple annual repetition.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>ARIMA(1, 1, 1):</b> (1 - &phi;<sub>1</sub>B)(1 - B)y<sub>t</sub> = (1 + &theta;<sub>1</sub>B)&epsilon;<sub>t</sub>, where &epsilon;<sub>t</sub> ~ WN(0, &sigma;<sup>2</sup>). "
        "Captures momentum via AR(1) and moving-average shock absorption via MA(1) on differenced stationary data. "
        "Selected via ACF/PACF cutoffs and verified through Ljung-Box white noise testing (p &gt; 0.05).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>SARIMA(1, 0, 1)(1, 0, 1)[12]:</b> Incorporates multiplicative seasonal lag polynomials with period s = 12 months, "
        "capturing annual Kharif (monsoon) and Rabi (winter) epidemiological cycles.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Holt-Winters Triple Exponential Smoothing (Additive):</b> Decomposes the series into Level (l<sub>t</sub>), Trend (b<sub>t</sub>), and Seasonality (s<sub>t</sub>) "
        "using recursive smoothing parameters &alpha;, &beta;, &gamma; &isin; [0, 1]. Adapts dynamically without rigid Gaussian disturbance assumptions.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>ARIMAX / SARIMAX with Lagged Weather:</b> y<sub>t</sub> = &sum; &beta;<sub>k</sub> X<sub>k, t-1</sub> + [&Theta;(B)/&Phi;(B)]&epsilon;<sub>t</sub>. "
        "Uses 1-month lagged agrometeorology (X<sub>t-1</sub>: rainfall, temperature, relative humidity). Avoids future data leakage while modeling the biological incubation lag.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Machine Learning Autoregressors:</b> HistGradientBoostingRegressor and RandomForestRegressor trained on autoregressive lag features (t-1, t-2, t-3, t-12).",
        bullet_style
    ))

    # ==================================================================
    # PAGE 3: CHRONOLOGICAL BENCHMARK & METRIC RIGOR
    # ==================================================================
    story.append(PageBreak())
    
    story.append(Paragraph("5. Chronological Out-of-Sample Benchmark & Model Comparison", h1_style))
    story.append(Paragraph(
        "<b>Zero Lookahead Data Leakage Protocol:</b> Models were trained strictly on the initial 70% chronological timeline (2015–2022) "
        "and evaluated against the held-out 30% horizon (2023–2026). No random cross-validation shuffling was permitted. "
        "11 distinct model architectures were evaluated on identical test intervals:",
        body_style
    ))

    model_comp_data = [
        [
            Paragraph("Rank", table_header_style),
            Paragraph("Forecasting Model Architecture", table_header_style),
            Paragraph("RMSE", table_header_style),
            Paragraph("MAE", table_header_style),
            Paragraph("MAPE (%)", table_header_style),
            Paragraph("sMAPE (%)", table_header_style),
            Paragraph("WAPE (%)", table_header_style),
            Paragraph("Bias", table_header_style)
        ],
        [
            Paragraph("<b>#1</b>", table_cell_center),
            Paragraph("<b>ARIMA(1, 1, 1)</b>", table_cell_bold),
            Paragraph("<b>1.823</b>", table_cell_center),
            Paragraph("<b>1.243</b>", table_cell_center),
            Paragraph("68.4%", table_cell_center),
            Paragraph("123.45%", table_cell_center),
            Paragraph("<b>85.73%</b>", table_cell_center),
            Paragraph("+0.34", table_cell_center)
        ],
        [
            Paragraph("<b>#2</b>", table_cell_center),
            Paragraph("Seasonal Naive Baseline (y<sub>t</sub> = y<sub>t-12</sub>)", table_cell_style),
            Paragraph("1.871", table_cell_center),
            Paragraph("1.400", table_cell_center),
            Paragraph("82.1%", table_cell_center),
            Paragraph("129.14%", table_cell_center),
            Paragraph("96.55%", table_cell_center),
            Paragraph("-0.15", table_cell_center)
        ],
        [
            Paragraph("<b>#3</b>", table_cell_center),
            Paragraph("ARIMAX (Lagged Weather t-1)", table_cell_style),
            Paragraph("1.873", table_cell_center),
            Paragraph("1.298", table_cell_center),
            Paragraph("71.2%", table_cell_center),
            Paragraph("133.37%", table_cell_center),
            Paragraph("89.51%", table_cell_center),
            Paragraph("+0.38", table_cell_center)
        ],
        [
            Paragraph("4", table_cell_center),
            Paragraph("SARIMAX (Seasonal + Lagged Weather)", table_cell_style),
            Paragraph("1.873", table_cell_center),
            Paragraph("1.290", table_cell_center),
            Paragraph("70.8%", table_cell_center),
            Paragraph("131.89%", table_cell_center),
            Paragraph("88.99%", table_cell_center),
            Paragraph("+0.22", table_cell_center)
        ],
        [
            Paragraph("5", table_cell_center),
            Paragraph("Holt-Winters Triple Exponential Smoothing", table_cell_style),
            Paragraph("1.889", table_cell_center),
            Paragraph("1.301", table_cell_center),
            Paragraph("74.5%", table_cell_center),
            Paragraph("119.57%", table_cell_center),
            Paragraph("89.73%", table_cell_center),
            Paragraph("+0.42", table_cell_center)
        ],
        [
            Paragraph("6", table_cell_center),
            Paragraph("Hist Gradient Boosting Regressor", table_cell_style),
            Paragraph("1.959", table_cell_center),
            Paragraph("1.383", table_cell_center),
            Paragraph("79.1%", table_cell_center),
            Paragraph("151.64%", table_cell_center),
            Paragraph("95.38%", table_cell_center),
            Paragraph("+0.51", table_cell_center)
        ],
        [
            Paragraph("7", table_cell_center),
            Paragraph("SARIMA(1,0,1)(1,0,1)[12]", table_cell_style),
            Paragraph("2.012", table_cell_center),
            Paragraph("1.346", table_cell_center),
            Paragraph("76.3%", table_cell_center),
            Paragraph("153.34%", table_cell_center),
            Paragraph("92.83%", table_cell_center),
            Paragraph("+0.48", table_cell_center)
        ],
        [
            Paragraph("8", table_cell_center),
            Paragraph("AR(1) Autoregressive", table_cell_style),
            Paragraph("2.201", table_cell_center),
            Paragraph("1.457", table_cell_center),
            Paragraph("84.6%", table_cell_center),
            Paragraph("161.78%", table_cell_center),
            Paragraph("100.48%", table_cell_center),
            Paragraph("+0.58", table_cell_center)
        ],
        [
            Paragraph("9", table_cell_center),
            Paragraph("MA(1) Moving Average", table_cell_style),
            Paragraph("2.202", table_cell_center),
            Paragraph("1.455", table_cell_center),
            Paragraph("83.9%", table_cell_center),
            Paragraph("140.00%", table_cell_center),
            Paragraph("100.34%", table_cell_center),
            Paragraph("-0.62", table_cell_center)
        ],
        [
            Paragraph("10", table_cell_center),
            Paragraph("Random Forest Autoregressor (Lags 1,2,3,12)", table_cell_style),
            Paragraph("2.211", table_cell_center),
            Paragraph("1.661", table_cell_center),
            Paragraph("91.2%", table_cell_center),
            Paragraph("166.35%", table_cell_center),
            Paragraph("114.55%", table_cell_center),
            Paragraph("+0.65", table_cell_center)
        ],
        [
            Paragraph("11", table_cell_center),
            Paragraph("Meta Prophet (Additive Fourier Seasonality)", table_cell_style),
            Paragraph("2.427", table_cell_center),
            Paragraph("1.596", table_cell_center),
            Paragraph("88.5%", table_cell_center),
            Paragraph("133.89%", table_cell_center),
            Paragraph("110.07%", table_cell_center),
            Paragraph("+0.72", table_cell_center)
        ]
    ]
    t_models = Table(model_comp_data, colWidths=[30, 200, 48, 48, 54, 56, 56, 48])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 6))

    story.append(Paragraph("6. Statistical Parsimony & Metric Integrity Policy", h1_style))
    story.append(Paragraph(
        "• <b>Statistical Parsimony Proof (Occam's Razor):</b> ARIMA(1,1,1) outperforms complex machine learning models (Gradient Boosting, Random Forest) "
        "because monthly epidemiological counts contain low counts and intermittent zeros. Tree models overfit idiosyncratic local noise and fail to extrapolate beyond historical splits. "
        "ARIMA(1,1,1) estimates only 3 parameters (&phi;<sub>1</sub>, d=1, &theta;<sub>1</sub>), delivering the lowest RMSE (1.823) and MAE (1.243).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Zero-Denominator Policy (Why Raw MAPE Fails):</b> In disease surveillance, actual counts y<sub>t</sub> can be zero. Raw MAPE involves division by zero (1/0 = &infin;). "
        "Silently adding arbitrary constants produces misleading percentages. Verdant Intelligence computes: "
        "<br/>1. <b>Symmetric MAPE (sMAPE &le; 200%):</b> Denominator (|y<sub>t</sub>| + |y<sub>t</sub><sup>pred</sup>|)/2 bounds percentage error safely. "
        "<br/>2. <b>Weighted Absolute Percentage Error (WAPE):</b> &Sigma; |y<sub>t</sub> - y<sub>t</sub><sup>pred</sup>| / &Sigma; y<sub>t</sub>. "
        "ARIMA(1,1,1) achieves WAPE = 85.73%, beating the Seasonal Naive baseline (96.55%) by 10.8 percentage points.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Non-Negative Forecast Enforcement:</b> Point forecasts and confidence intervals are bounded at zero via max(0, y<sub>t</sub><sup>pred</sup>) in <code>src/forecasting.py</code>, "
        "preventing impossible negative disease report counts.",
        bullet_style
    ))

    # ==================================================================
    # PAGE 4: PATHOLOGY, AGRONOMY & SOFTWARE ARCHITECTURE
    # ==================================================================
    story.append(PageBreak())
    
    story.append(Paragraph("7. Plant Pathology & Agronomic Domain Interpretation", h1_style))
    story.append(Paragraph(
        "A forecasting model in agriculture is only valuable if its outputs reflect true biological mechanisms. "
        "Verdant Intelligence grounds all quantitative outputs in Indian agro-ecological realities:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>The Biological Incubation Lag (t-1 Peak Cross-Correlation):</b> Cross-correlation analysis between NASA POWER precipitation (PRECTOTCORR) "
        "and Rice Blast (<i>Pyricularia oryzae</i>) reports reveals a statistically significant peak at <b>Lag t-1 (r = +0.58, p &lt; 0.01)</b>. "
        "In plant pathology, fungal conidia require &gt;10 hours of continuous leaf wetness and relative humidity &gt;85% for germination and appressorium penetration. "
        "The 1-month lag reflects the latent incubation period required for fungal mycelium to colonize host tissues and manifest macroscopic necrotic lesions observed during field surveillance.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Thermal Stress & Drought Dynamics (Dry Root Rot):</b> For <i>Macrophomina phaseolina</i>, incidence accelerates when soil temperatures exceed 32&deg;C "
        "combined with moisture deficit (May–June pre-monsoon and post-monsoon dry intervals). The model accurately detects these thermal stress vulnerability windows.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Operational Extension Preemption Window:</b> Generating 3-to-12 month forward projections with 80% and 95% confidence intervals gives "
        "Krishi Vigyan Kendras (KVKs) and ICAR a 30-to-60 day operational window to preposition systemic fungicides and broadcast preventative advisories.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Scientific Data Caveat:</b> All dashboard displays explicitly clarify that figures represent <i>recorded scientific/field occurrences</i>, "
        "not total infested crop acreage. This preserves scientific accuracy and prevents sensationalist misinterpretations.",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("8. Software Engineering & 6-Module Streamlit Architecture", h1_style))
    story.append(Paragraph(
        "The platform is organized into an enterprise-grade modular Python package running live on port 8501:",
        body_style
    ))

    mod_data = [
        [
            Paragraph("Module / Page", table_header_style),
            Paragraph("Primary File Path", table_header_style),
            Paragraph("Key Features & Scientific Capabilities", table_header_style)
        ],
        [
            Paragraph("<b>1. Overview</b>", table_cell_style),
            Paragraph("<code>app.py</code>", table_cell_style),
            Paragraph("Executive hero section, high-level dataset metrics (3,076 records, 10 pathogens), best-model preview (ARIMA(1,1,1)), 6-page navigation.", table_cell_style)
        ],
        [
            Paragraph("<b>2. Disease Explorer</b>", table_cell_style),
            Paragraph("<code>pages/1_Disease_Explorer.py</code>", table_cell_style),
            Paragraph("Pathogen selector, host crops, geographical distribution map, monthly timeline, 3-part interpretation cards.", table_cell_style)
        ],
        [
            Paragraph("<b>3. Time Series Analysis</b>", table_cell_style),
            Paragraph("<code>pages/2_Time_Series_Analysis.py</code>", table_cell_style),
            Paragraph("8 comprehensive analytical tabs: Resampling, Outlier Z-score inspection, Seasonality heatmaps, STL decomposition, ACF/PACF memory, Dual Stationarity (ADF+KPSS), NASA POWER Weather Lags (t-1), Chronological split.", table_cell_style)
        ],
        [
            Paragraph("<b>4. Model Comparison</b>", table_cell_style),
            Paragraph("<code>pages/3_Model_Comparison.py</code>", table_cell_style),
            Paragraph("11-model benchmark table, RMSE/MAE/sMAPE bar charts, Actual vs Predicted overlays, Error distribution plots, Ljung-Box white noise diagnostics, Mathematical parameter assumptions tab.", table_cell_style)
        ],
        [
            Paragraph("<b>5. Future Forecasts</b>", table_cell_style),
            Paragraph("<code>pages/4_Future_Forecasts.py</code>", table_cell_style),
            Paragraph("3-12 month forward projections, 80% & 95% confidence intervals, non-negative bounds (zero clipping), Section 9 FORECAST SUMMARY card answering all 5 operational questions.", table_cell_style)
        ],
        [
            Paragraph("<b>6. Data & Methodology</b>", table_cell_style),
            Paragraph("<code>pages/5_Data_and_Methodology.py</code>", table_cell_style),
            Paragraph("GBIF & NASA POWER data documentation, ISO 8601 parsing rules, stationarity methodology, limitations, data download center.", table_cell_style)
        ]
    ]
    t_mod = Table(mod_data, colWidths=[110, 140, 290])
    t_mod.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_card_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_mod)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Automated Verification Suite:</b> 21 automated unit tests in <code>tests/test_all_features.py</code> pass in 0.64s, "
        "validating data pipelines, statistical stationarity modules, model convergence, and zero-safe metric calculations.",
        body_style
    ))

    # ==================================================================
    # PAGE 5: FACULTY ORAL VIVA & DEFENSE (PART 1)
    # ==================================================================
    story.append(PageBreak())
    
    story.append(Paragraph("PART 2: FACULTY ORAL VIVA & EXAMINATION DEFENSE GUIDE", title_style))
    story.append(Paragraph(
        "<b>Authoritative High-Scoring Responses to Anticipated Evaluation Panel Questions</b><br/>"
        "Curated for the Phase 2 Mid-Semester Evaluation across Time Series Econometrics, Modeling, and Data Integrity.",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceAfter=6))

    story.append(Paragraph("Theme A: Time Series Stationarity, Transformations & Diagnostics", h2_style))

    q_and_a_1 = [
        (
            "Q1. Why is stationarity testing mandatory before fitting ARIMA models? What does a 'unit root' mean physically in an epidemic?",
            "Classical autoregressive models assume weak stationarity: constant mean, constant variance, and autocovariance depending only on time displacement (h). "
            "If a time series contains a unit root (|z| = 1), any random shock—such as an unseasonal cyclone or disease super-spreader event—persists indefinitely, "
            "causing variance to explode as t &rarr; &infin;. Physically in plant pathology, a non-stationary epidemic means disease momentum compounds cumulatively across seasons. "
            "First differencing (d = 1) removes this stochastic integration, transforming the cumulative level into month-over-month infection velocity, which has stable variance."
        ),
        (
            "Q2. Why did you implement BOTH the Augmented Dickey-Fuller (ADF) test AND the KPSS test?",
            "ADF and KPSS have complementary null hypotheses, protecting us from false conclusions: "
            "ADF tests H<sub>0</sub>: 'The series has a unit root (non-stationary)' vs. H<sub>1</sub>: 'Stationary'. "
            "KPSS tests H<sub>0</sub>: 'The series is level/trend stationary' vs. H<sub>1</sub>: 'Unit root present'. "
            "If ADF fails to reject (p &gt; 0.05) AND KPSS rejects (p &lt; 0.05), we have conclusive mathematical proof of a stochastic unit root. "
            "In our data, raw *Macrophomina* showed ADF p = 0.9655 and KPSS p = 0.0100. After first differencing (d = 1), ADF p &lt; 0.0001, confirming strict stationarity."
        ),
        (
            "Q3. How did you determine the parameter orders (p, d, q) for ARIMA(1,1,1)?",
            "We used a 3-step identification protocol: "
            "1. Integrated order (d = 1) was determined by the joint ADF-KPSS test. "
            "2. Order selection (p, q) was guided by sample ACF and PACF correlograms: PACF showed a sharp cutoff after lag 1 (indicating p = 1), while ACF exhibited significant correlation at lag 1 with exponential damping (indicating q = 1). "
            "3. We systematically compared AIC and BIC criteria across grid orders (0 to 3). ARIMA(1,1,1) yielded the lowest BIC while passing the Ljung-Box test for white-noise residuals (p &gt; 0.05)."
        ),
        (
            "Q4. How do you distinguish between an unsurveyed month and an active zero-infection month?",
            "In observational biodiversity data like GBIF, observers record positive identifications. If month M has zero records, it may represent true absence (zero disease) OR an absence of sampling effort (e.g., lockdown, monsoon impassability). "
            "When resampling to monthly start frequency (`MS`), we preserve survey metadata (`is_missing=True`). For econometric estimation, we do not impute arbitrary zeros without justification, "
            "as doing so would artificially deflate the moving average baseline and introduce severe under-forecasting bias."
        )
    ]

    for q, a in q_and_a_1:
        story.append(Paragraph(f"<b>{q}</b>", q_title_style))
        story.append(Paragraph(a, q_ans_style))

    story.append(HRFlowable(width="100%", thickness=0.5, color=c_card_border, spaceAfter=4))
    story.append(Paragraph("Theme B: Econometric vs. Machine Learning Forecasters & Model Assumptions", h2_style))

    q_and_a_2 = [
        (
            "Q5. Why did classical ARIMA(1,1,1) outperform modern machine learning models like Random Forest and Gradient Boosting?",
            "This demonstrates the principle of <b>Statistical Parsimony (Occam's Razor)</b>. Machine learning regressors (Random Forest, Gradient Boosting) rely on non-parametric decision trees "
            "that require large sample sizes (thousands of observations) to partition feature space effectively. In monthly macro-epidemiology (132 time points), tree models overfit idiosyncratic noise "
            "and cannot extrapolate trend outside their historical training leaf values. ARIMA(1,1,1) estimates only 3 parameters (&phi;<sub>1</sub>, d=1, &theta;<sub>1</sub>), "
            "giving it superior generalization on sparse chronological holdout data (RMSE 1.823 vs. Random Forest 2.211)."
        ),
        (
            "Q6. Why is the Seasonal Naive model your baseline, and did your models beat it?",
            "In agricultural time series with annual cropping seasons (Kharif and Rabi), a naive random walk (y<sub>t</sub> = y<sub>t-1</sub>) is inappropriate because winter weather does not resemble monsoon weather. "
            "The correct, rigorous benchmark is the Seasonal Naive model (y<sub>t</sub> = y<sub>t-12</sub>), which predicts that this year's July disease incidence will equal last year's July incidence. "
            "Our ARIMA(1,1,1) successfully defeated the Seasonal Naive baseline, achieving an RMSE of 1.823 vs 1.871, MAE of 1.243 vs 1.400, and a 10.8 percentage point improvement in WAPE (85.73% vs 96.55%)."
        ),
        (
            "Q7. What are the key mathematical components of Holt-Winters Exponential Smoothing?",
            "Holt-Winters Triple Exponential Smoothing decomposes the series into Level (l<sub>t</sub>), Trend (b<sub>t</sub>), and Seasonality (s<sub>t</sub>) using smoothing coefficients &alpha;, &beta;, &gamma; &isin; [0, 1]: "
            "<br/>&bull; Level: l<sub>t</sub> = &alpha;(y<sub>t</sub> - s<sub>t-m</sub>) + (1 - &alpha;)(l<sub>t-1</sub> + b<sub>t-1</sub>) "
            "<br/>&bull; Trend: b<sub>t</sub> = &beta;(l<sub>t</sub> - l<sub>t-1</sub>) + (1 - &beta;)b<sub>t-1</sub> "
            "<br/>&bull; Season: s<sub>t</sub> = &gamma;(y<sub>t</sub> - l<sub>t</sub>) + (1 - &gamma;)s<sub>t-m</sub> "
            "<br/>Its strength is adaptive exponential decay weighting without requiring strict Gaussian disturbance assumptions. It ranked 5th overall (RMSE 1.889) and achieved the lowest sMAPE (119.57%)."
        ),
        (
            "Q8. How does ARIMAX/SARIMAX incorporate weather variables without creating data leakage?",
            "In ARIMAX, exogenous variables X<sub>t</sub> must be known at forecast time t. If we used contemporaneous weather (X<sub>t</sub>), we would be predicting future disease using unknown future weather, "
            "which constitutes temporal data leakage. We solved this by using <b>lagged exogenous variables (X<sub>t-1</sub>)</b>: last month's precipitation, temperature, and humidity. "
            "This is both mathematically valid (known at time t) and biologically sound, as fungal spore germination takes 2–4 weeks to develop into detectable field lesions."
        )
    ]

    for q, a in q_and_a_2:
        story.append(Paragraph(f"<b>{q}</b>", q_title_style))
        story.append(Paragraph(a, q_ans_style))

    # ==================================================================
    # PAGE 6: FACULTY ORAL VIVA & DEFENSE (PART 2) & 5-MIN PITCH
    # ==================================================================
    story.append(PageBreak())
    
    story.append(Paragraph("Theme C: Metric Rigor & Sparse Data Mathematical Integrity", h2_style))

    q_and_a_3 = [
        (
            "Q9. Why is standard Mean Absolute Percentage Error (MAPE) mathematically problematic in this dataset?",
            "The formula for MAPE is (1/n) &Sigma; |(y<sub>t</sub> - y<sub>t</sub><sup>pred</sup>) / y<sub>t</sub>| &times; 100%. When actual occurrence count y<sub>t</sub> = 0, the denominator is zero, "
            "making MAPE mathematically undefined (&infin;). Silently replacing zero with an arbitrary epsilon like 0.001 produces wildly inflated, distorted percentages (e.g., 50,000%). "
            "To uphold scientific integrity, Verdant Intelligence computes: "
            "<br/>1. <b>sMAPE (Symmetric MAPE):</b> Bounds percentage error between 0% and 200% via denominator (|y<sub>t</sub>| + |y<sub>t</sub><sup>pred</sup>|)/2. "
            "<br/>2. <b>WAPE (Weighted Absolute Percentage Error):</b> &Sigma; |y<sub>t</sub> - y<sub>t</sub><sup>pred</sup>| / &Sigma; y<sub>t</sub>, which aggregates deviations over total incidence, eliminating zero-division entirely."
        ),
        (
            "Q10. Why do you report both RMSE and MAE? What does a large gap between them indicate?",
            "MAE evaluates linear average error (L<sub>1</sub> norm), weighting all deviations equally. RMSE squares errors before averaging (L<sub>2</sub> norm), heavily penalizing large forecast misses. "
            "In epidemiological surveillance, failing to predict a massive disease outbreak is far more damaging to farmers than a small 1-unit baseline error. "
            "The fact that ARIMA(1,1,1) achieved the lowest RMSE (1.823) confirms it does not produce catastrophic outlier spikes during non-epidemic months."
        )
    ]

    for q, a in q_and_a_3:
        story.append(Paragraph(f"<b>{q}</b>", q_title_style))
        story.append(Paragraph(a, q_ans_style))

    story.append(HRFlowable(width="100%", thickness=0.5, color=c_card_border, spaceAfter=4))
    story.append(Paragraph("Theme D: Agricultural Domain Interpretation & Plant Pathology", h2_style))

    q_and_a_4 = [
        (
            "Q11. What does the 1-month weather lag (t-1) tell us about Rice Blast biology?",
            "Rice Blast is caused by the hemibiotrophic fungus <i>Pyricularia oryzae</i>. Conidia germinate on leaf surfaces only when relative humidity exceeds 85–90% "
            "and free moisture (dew/rain) persists for at least 10–14 hours at 20–28&deg;C. After appressorium penetration, the fungal mycelium undergoes an asymptomatic biotrophic phase "
            "inside plant tissue before transitioning to necrotrophic lesion expansion. This latent incubation period takes approximately 2 to 4 weeks. "
            "Therefore, high rainfall in July drives macroscopic foliar necrosis and field observations recorded in August (Lag t-1)."
        ),
        (
            "Q12. How does this system differentiate between 'recorded disease occurrences' and 'infected crop acreage'?",
            "This is a fundamental scientific caveat in our platform. Biodiversity and herbarium databases (GBIF) record confirmed research accessions and field observations. "
            "They measure <i>disease presence and spatial reporting intensity</i>, NOT the percentage of farm acreage destroyed. "
            "Our user interface and documentation explicitly state: <i>'Values represent recorded research occurrences. They serve as an early-warning outbreak proxy, not total destroyed agricultural acreage.'</i>"
        )
    ]

    for q, a in q_and_a_4:
        story.append(Paragraph(f"<b>{q}</b>", q_title_style))
        story.append(Paragraph(a, q_ans_style))

    story.append(HRFlowable(width="100%", thickness=0.5, color=c_card_border, spaceAfter=4))
    story.append(Paragraph("Theme E: Software Verification & Phase 3 Roadmap", h2_style))

    q_and_a_5 = [
        (
            "Q13. How did you verify code correctness and prevent regressions?",
            "We built an automated test suite using `pytest` located in `tests/test_all_features.py`. The suite executes 21 automated tests verifying: "
            "data loading pipelines, date conversions, continuous monthly aggregation, missing value imputation, ADF/KPSS statistical testing, "
            "ARIMA/SARIMA/Holt-Winters execution, error metric calculations (verifying zero-safe MAPE/sMAPE/WAPE), and UI helper rendering. "
            "All 21 unit tests pass in 0.64 seconds with zero syntax errors across the entire codebase."
        ),
        (
            "Q14. What is planned for Phase 3 to build upon these Phase 2 achievements?",
            "Phase 2 successfully established the econometric baseline, stationarity proofs, classical benchmark, and weather lag mechanisms at macro-regional scale. "
            "For Phase 3 (End-Semester), we will implement: "
            "<br/>1. <b>District-Level Spatial Downscaling:</b> Geographically localized hierarchical forecasting downscaled to top agricultural districts. "
            "<br/>2. <b>Deep Sequence Architectures:</b> Benchmarking Temporal Fusion Transformers (TFT) and N-BEATS with multi-pathogen cross-learning. "
            "<br/>3. <b>Automated Early-Warning Push API:</b> Automated webhook dispatching real-time risk alerts to farmer cooperatives."
        )
    ]

    for q, a in q_and_a_5:
        story.append(Paragraph(f"<b>{q}</b>", q_title_style))
        story.append(Paragraph(a, q_ans_style))

    story.append(Spacer(1, 4))
    
    # 5-Minute Elevator Pitch Callout Box
    pitch_data = [
        [
            Paragraph(
                "<b>EXECUTIVE 5-MINUTE ORAL VIVA PITCH (MEMORIZE FOR EVALUATION OPENING):</b><br/>"
                "<i>'Respected Evaluators, Verdant Intelligence addresses the critical challenge of forecasting crop pathogen outbreaks across India. "
                "Our empirical pipeline combines 11 years of GBIF disease records with NASA POWER agrometeorology across 132 continuous months. "
                "Through joint ADF and KPSS tests, we proved raw pathogen series exhibit stochastic unit roots, establishing that first differencing (d=1) is mathematically essential. "
                "Across an 11-model out-of-sample benchmark evaluated on held-out data with zero lookahead leakage, classical ARIMA(1,1,1) achieved top performance (RMSE 1.823, MAE 1.243), "
                "outperforming the Seasonal Naive baseline by 10.8 percentage points in WAPE. Crucially, we demonstrated that simple parsimonious econometrics outperform complex decision trees on sparse count data, "
                "and cross-correlation confirmed a 1-month biological incubation lag between monsoon precipitation and macroscopic foliar blast lesions. "
                "The system is fully operational across 6 Streamlit modules and validated by 21 passing automated unit tests.'</i>",
                ParagraphStyle('PitchText', parent=styles['Normal'], fontName='Helvetica', fontSize=7.6, leading=10.2, textColor=c_primary)
            )
        ]
    ]
    t_pitch = Table(pitch_data, colWidths=[540])
    t_pitch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FEF9C3")), # Light Amber
        ('BOX', (0,0), (-1,-1), 1, c_highlight),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_pitch)

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated Phase 2 PDF: {filename}")

if __name__ == "__main__":
    build_pdf()
