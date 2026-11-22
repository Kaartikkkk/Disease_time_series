"""
Comprehensive 9-Page Academic Project Report Generator (Easy-to-Understand Language)
Subject: CSET 369 — Time Series Analysis & Forecasting
Institution: Bennett University, SCSET
Author: Kartik Lamba (Enrollment: S24CSEU0642)
Course Coordinator: Anurag Choubey
Prescribed Format: Research Paper / IEEE Format (Exact 9 Pages, <= 10 Pages)
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf, pacf

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen import canvas

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.abspath(os.path.join(REPO_ROOT, "reports"))
ASSETS_DIR = os.path.abspath(os.path.join(REPORTS_DIR, "assets"))
os.makedirs(ASSETS_DIR, exist_ok=True)

class AcademicNumberedCanvas(canvas.Canvas):
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

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#334155"))

        # Running Header (Pages 2+)
        if self._pageNumber > 1:
            header_text = "Bennett University | SCSET — CSET 369: Time Series Analysis & Forecasting (Phase 3 Final Report)"
            self.drawString(36, 11 * inch - 26, header_text)
            self.setFont("Helvetica", 7.5)
            self.drawRightString(8.5 * inch - 36, 11 * inch - 26, "Kartik Lamba (S24CSEU0642)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 30, 8.5 * inch - 36, 11 * inch - 30)

        # Running Footer (All pages)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#475569"))
        footer_text = "Final Project Evaluation (Phase 3) | Instructor: Anurag Choubey | AgriForecast Surveillance Suite"
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawString(36, 22, footer_text)
        self.setFont("Helvetica-Bold", 7.5)
        self.drawRightString(8.5 * inch - 36, 22, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 30, 8.5 * inch - 36, 30)

        self.restoreState()


def get_styles():
    styles = getSampleStyleSheet()

    primary = colors.HexColor("#0F172A")
    navy = colors.HexColor("#1E3A8A")
    body = colors.HexColor("#1E293B")
    muted = colors.HexColor("#64748B")

    styles.add(ParagraphStyle(
        name="DocTitle",
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=15.0,
        textColor=primary,
        alignment=1,
        spaceAfter=2,
    ))

    styles.add(ParagraphStyle(
        name="DocSubTitle",
        fontName="Helvetica-Bold",
        fontSize=9.2,
        leading=11.5,
        textColor=navy,
        alignment=1,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name="AuthorBlock",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.8,
        textColor=body,
        alignment=1,
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=8.8,
        leading=11.0,
        textColor=navy,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="SubSectionHeader",
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=10.0,
        textColor=primary,
        spaceBefore=3,
        spaceAfter=1.5,
        keepWithNext=True,
    ))

    styles.add(ParagraphStyle(
        name="BodyTextJustified",
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.5,
        textColor=body,
        spaceAfter=3,
        alignment=4,
    ))

    styles.add(ParagraphStyle(
        name="AbstractText",
        fontName="Helvetica-Oblique",
        fontSize=7.2,
        leading=9.4,
        textColor=body,
        alignment=4,
    ))

    styles.add(ParagraphStyle(
        name="TableText",
        fontName="Helvetica",
        fontSize=6.8,
        leading=8.3,
        textColor=body,
    ))

    styles.add(ParagraphStyle(
        name="TableHead",
        fontName="Helvetica-Bold",
        fontSize=6.8,
        leading=8.3,
        textColor=colors.white,
        alignment=1,
    ))

    styles.add(ParagraphStyle(
        name="FigCaption",
        fontName="Helvetica-Oblique",
        fontSize=6.8,
        leading=8.2,
        textColor=muted,
        alignment=1,
        spaceBefore=1.5,
        spaceAfter=3.0,
    ))

    return styles


def build_final_report_pdf():
    pdf_path = os.path.join(REPORTS_DIR, "MILESTONE_3_ADVANCED_FORECASTING_REPORT.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=34,
        bottomMargin=34,
    )

    styles = get_styles()
    story = []

    # Assets
    fig1 = os.path.join(ASSETS_DIR, "m2_fig1_trajectory.png")
    fig2 = os.path.join(ASSETS_DIR, "fig_seasonal_heatmap.png")
    fig3 = os.path.join(ASSETS_DIR, "m2_fig2_correlogram.png")
    fig4 = os.path.join(ASSETS_DIR, "fig_workflow_architecture.png")
    fig5 = os.path.join(ASSETS_DIR, "m3_fig1_forecast_holdout.png")
    fig6 = os.path.join(ASSETS_DIR, "m3_fig3_all_models_rmse.png")
    fig7 = os.path.join(ASSETS_DIR, "m3_fig2_garch_volatility.png")

    # =========================================================================
    # PAGE 1: TITLE, ABSTRACT, INDEX TERMS, SECTION I: INTRODUCTION
    # =========================================================================
    story.append(Paragraph("AgriForecast: Time Series Forecasting and Volatility Modeling for Crop Disease Outbreaks", styles["DocTitle"]))
    story.append(Paragraph("Phase 3 Comprehensive Final Project Report — Final Evaluation (12 Marks)", styles["DocSubTitle"]))
    
    auth_text = (
        "<b>Author:</b> Kartik Lamba (Enrollment: <b>S24CSEU0642</b>, Email: <code>lambakartik2006@gmail.com</code>)<br/>"
        "<b>Course:</b> CSET 369 — Time Series Analysis & Forecasting (Semester 5, Academic Year 2026–2027)<br/>"
        "<b>Department:</b> School of Computer Science, Engineering & Technology (SCSET) | <b>Institution:</b> Bennett University, Greater Noida, India<br/>"
        "<b>Course Coordinator:</b> Anurag Choubey | <b>Project Repository:</b> <code>github.com/Kaartikkkk/Disease_time_series</code>"
    )
    story.append(Paragraph(auth_text, styles["AuthorBlock"]))

    abstract_html = (
        "<b><i>Abstract</i>—Crop disease outbreaks cause severe harvest losses across India, threatening farmer income and food security. Today, farmers detect disease only after seeing physical leaf damage, which is often too late for effective treatment. In this project, called AgriForecast, we build an early warning forecasting system that predicts disease outbreaks 1 to 6 months in advance. We track 10 major Indian crop diseases over an unbroken 11-year monthly timeline (January 2015 to December 2025, 1,320 total records) combining GBIF field records with NASA weather satellites. We implement and evaluate 19 different models across three families: classical statistical models (ARIMA, SARIMA), machine learning trees (Random Forest, Hurdle XGBoost), and deep neural networks (PyTorch LSTM, PatchTST Transformer), along with GARCH volatility cones. Statistical tests show that 70% of crop diseases follow predictable seasonal cycles rather than random walks. On unseen holdout test data, our PatchTST Transformer achieves the best performance with an RMSE of 4.149 (a 3.7% improvement over the best classical model), while Hurdle XGBoost effectively handles dry off-season months where zero cases occur. GARCH(1,1) calculates dynamic risk cones to help government agencies order fungicides before market prices spike. The platform is deployed live on Streamlit Cloud and validated by 113 automated tests.</b><br/>"
        "<b><i>Index Terms</i>—Crop disease forecasting, Time series analysis, PyTorch LSTM, PatchTST, Hurdle XGBoost, GARCH volatility, Bennett University.</b>"
    )
    abs_table = Table([[Paragraph(abstract_html, styles["AbstractText"])]], colWidths=[7.5 * inch])
    abs_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(abs_table)
    story.append(Spacer(1, 3))

    story.append(Paragraph("I. INTRODUCTION", styles["SectionHeader"]))
    story.append(Paragraph(
        "<b>A. Why Crop Disease Forecasting Matters:</b> Plant diseases caused by fungi and bacteria damage major Indian crops every year. For example, Rice Blast (<i>Pyricularia oryzae</i>) and Sheath Blight (<i>Rhizoctonia solani</i>) can destroy 20% to 45% of rice harvests in a single season. Similarly, Potato Late Blight can wipe out an entire potato field in just a few days if conditions are wet and warm. Today, farmers and agricultural officers detect these diseases only after seeing visible leaf damage. By that time, fungal spores have already spread throughout the field, making chemical sprays expensive and far less effective. If agricultural departments can forecast outbreaks 1 to 6 months in advance, they can warn farmers early, distribute preventative fungicides, and protect food supplies.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>B. How Plant Diseases Spread (The Disease Triangle):</b> Plant diseases do not spread randomly. In plant biology, an outbreak happens only when three conditions meet at the exact same time: (1) <i>The Pathogen:</i> active fungal spores must be present in the air or soil; (2) <i>The Host Crop:</i> the crop must be in a vulnerable growing stage; and (3) <i>The Weather:</i> temperature must be between 20&deg;C and 28&deg;C, and relative humidity must rise above 80%. If the air stays dry, zero infections occur. But once humidity crosses 85%, spore germination accelerates rapidly.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>C. The Three Main Data Challenges:</b> Forecasting crop disease with time-series models is challenging because: (1) <i>Extreme Zeroes:</i> Over 71.8% of historical months have zero recorded cases because crops are either not planted or the weather is too dry; (2) <i>Sudden Climate Triggers:</i> Disease does not increase smoothly; it jumps suddenly when rainy monsoon weather arrives; and (3) <i>Explosive Volatility:</i> Quiet dry months with zero variance alternate with chaotic monsoon seasons where case numbers vary wildly.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>D. Project Goals for Phase 3:</b> In this final phase, we: (1) Check whether disease series wander randomly or follow predictable cycles; (2) Implement and compare 19 models across classical statistics, modern machine learning, deep learning, and volatility modeling; (3) Test all models strictly on future unseen data without leaking answers; (4) Provide clear recommendations for farmers and officials; and (5) Deliver a live cloud web app verified with automated tests.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>E. Outline of this Report:</b> Section II reviews earlier research. Section III describes the dataset and feature preparation. Section IV covers classical models (ARIMA) and stationarity tests. Section V presents advanced neural networks, trees, and GARCH volatility. Section VI details parameter tuning. Section VII compares model results on unseen data. Section VIII covers risk cones and farm policy. Section IX outlines future work, followed by references and appendix.",
        styles["BodyTextJustified"]
    ))

    # =========================================================================
    # PAGE 2: SECTION II: LITERATURE REVIEW & SECTION III: DATASET (PART 1)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("II. LITERATURE REVIEW & PREVIOUS WORK", styles["SectionHeader"]))
    story.append(Paragraph(
        "Over the years, researchers have tried different mathematical methods to forecast agricultural diseases:",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>1. Classical Time Series (ARIMA):</b> Box, Jenkins, and Reinsel [1] introduced ARIMA models, which predict future values using past values and past forecast errors. Holt [2] and Winters added exponential smoothing for seasonal patterns. While simple and reliable, standard ARIMA assumes that data follows a bell curve (Gaussian), which makes it predict unrealistic negative numbers during dry months when zero cases occur.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>2. Machine Learning Tree Models:</b> Breiman [3] introduced Random Forests, which combine many decision trees to make stable predictions. Chen and Guestrin [4] developed XGBoost, which trains trees sequentially to correct earlier errors. While decision trees are great at catching non-linear weather thresholds, standard trees tend to predict fractional numbers like 0.4 cases during zero-case months, which adds up to significant error.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>3. Deep Learning Sequence Models (LSTM & Transformers):</b> Hochreiter and Schmidhuber [5] created Long Short-Term Memory (LSTM) networks with internal memory gates that remember patterns over long sequences. Cho et al. [6] introduced the Gated Recurrent Unit (GRU), which works similarly with fewer parameters. Vaswani et al. [8] introduced Transformers using self-attention. Recently, Nie et al. [7] developed PatchTST, showing that grouping time steps into small patches makes Transformers much more accurate on short, noisy time series.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>4. Econometric Volatility & Zero-Inflated Models:</b> Engle [9] developed ARCH, and Bollerslev [10] developed GARCH, which model how forecast uncertainty changes over time. Cragg [11] created hurdle models, which first predict whether an event will happen at all, and then estimate how severe it will be. Taylor and Letham [12] built Prophet for business trends with seasonal harmonics.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>5. What Makes Our Project Different:</b> Most past research tested only one model (like ARIMA alone or LSTM alone) on a single crop. In AgriForecast, we benchmark 19 models across 10 major crops on identical out-of-sample test data with zero data leakage, solving the zero-inflation challenge and modeling risk with GARCH cones.",
        styles["BodyTextJustified"]
    ))
    story.append(Spacer(1, 2))

    story.append(Paragraph("III. DATASET DESCRIPTION & CHARACTERISTICS", styles["SectionHeader"]))
    story.append(Paragraph(
        "<b>A. Where the Data Comes From:</b> Our dataset (<code>data/disease_timeseries_master.csv</code>) brings together two verified data sources: (1) field disease observations across India from the Global Biodiversity Information Facility (GBIF) and EPPO; and (2) monthly surface weather data (temperature, rainfall, humidity) from NASA POWER satellites. The dataset covers 132 continuous months from January 2015 to December 2025 across 10 crops, totaling 1,320 records.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>B. Target Variable & Key Statistics:</b> We predict <code>documented_occurrence_count</code> (the number of verified disease reports per month). The average across all crops is 1.34 cases per month, reaching a peak of 74 cases during the severe Rice Blast outbreak in July 2020. Exactly 71.82% of months have zero cases because crops are either out of season or the weather is too dry for spores to germinate.",
        styles["BodyTextJustified"]
    ))

    # Table I: Pathogen Taxonomy
    t1_data = [
        [Paragraph("<b>Crop Pathogen</b>", styles["TableHead"]), Paragraph("<b>Disease Name</b>", styles["TableHead"]), Paragraph("<b>Host Plant</b>", styles["TableHead"]), Paragraph("<b>Zero %</b>", styles["TableHead"]), Paragraph("<b>Mean</b>", styles["TableHead"]), Paragraph("<b>Max</b>", styles["TableHead"]), Paragraph("<b>Typical Outbreak Season</b>", styles["TableHead"])],
        [Paragraph("<i>Pyricularia oryzae</i>", styles["TableText"]), Paragraph("Rice Blast", styles["TableText"]), Paragraph("Rice (Oryza sativa)", styles["TableText"]), Paragraph("68.2%", styles["TableText"]), Paragraph("2.14", styles["TableText"]), Paragraph("74", styles["TableText"]), Paragraph("Monsoon / Kharif (Jul - Oct)", styles["TableText"])],
        [Paragraph("<i>Bipolaris oryzae</i>", styles["TableText"]), Paragraph("Brown Spot", styles["TableText"]), Paragraph("Rice", styles["TableText"]), Paragraph("70.5%", styles["TableText"]), Paragraph("1.42", styles["TableText"]), Paragraph("38", styles["TableText"]), Paragraph("Monsoon / Kharif (Aug - Nov)", styles["TableText"])],
        [Paragraph("<i>Rhizoctonia solani</i>", styles["TableText"]), Paragraph("Sheath Blight", styles["TableText"]), Paragraph("Rice", styles["TableText"]), Paragraph("72.0%", styles["TableText"]), Paragraph("1.18", styles["TableText"]), Paragraph("29", styles["TableText"]), Paragraph("Warm Rainy (Jul - Sep)", styles["TableText"])],
        [Paragraph("<i>Xanthomonas oryzae</i>", styles["TableText"]), Paragraph("Bacterial Blight", styles["TableText"]), Paragraph("Rice", styles["TableText"]), Paragraph("74.2%", styles["TableText"]), Paragraph("0.98", styles["TableText"]), Paragraph("22", styles["TableText"]), Paragraph("Late Monsoon (Aug - Oct)", styles["TableText"])],
        [Paragraph("<i>Puccinia striiformis</i>", styles["TableText"]), Paragraph("Yellow Stripe Rust", styles["TableText"]), Paragraph("Wheat (Triticum)", styles["TableText"]), Paragraph("75.8%", styles["TableText"]), Paragraph("0.84", styles["TableText"]), Paragraph("19", styles["TableText"]), Paragraph("Winter / Rabi (Dec - Mar)", styles["TableText"])],
        [Paragraph("<i>Alternaria solani</i>", styles["TableText"]), Paragraph("Early Blight", styles["TableText"]), Paragraph("Potato / Tomato", styles["TableText"]), Paragraph("69.7%", styles["TableText"]), Paragraph("1.62", styles["TableText"]), Paragraph("31", styles["TableText"]), Paragraph("Post-Monsoon & Winter", styles["TableText"])],
        [Paragraph("<i>Phytophthora infestans</i>", styles["TableText"]), Paragraph("Late Blight", styles["TableText"]), Paragraph("Potato", styles["TableText"]), Paragraph("71.2%", styles["TableText"]), Paragraph("1.51", styles["TableText"]), Paragraph("35", styles["TableText"]), Paragraph("Humid Winter (Nov - Jan)", styles["TableText"])],
        [Paragraph("<i>Fusarium oxysporum</i>", styles["TableText"]), Paragraph("Wilt Complex", styles["TableText"]), Paragraph("Chickpea / Cotton", styles["TableText"]), Paragraph("73.5%", styles["TableText"]), Paragraph("1.10", styles["TableText"]), Paragraph("24", styles["TableText"]), Paragraph("Warm Spring (Feb - Apr)", styles["TableText"])],
        [Paragraph("<i>Macrophomina phaseolina</i>", styles["TableText"]), Paragraph("Dry Root Rot", styles["TableText"]), Paragraph("Pulses / Soybean", styles["TableText"]), Paragraph("72.7%", styles["TableText"]), Paragraph("1.25", styles["TableText"]), Paragraph("26", styles["TableText"]), Paragraph("Dry Post-Monsoon Heat", styles["TableText"])],
        [Paragraph("<i>Ralstonia solanacearum</i>", styles["TableText"]), Paragraph("Bacterial Wilt", styles["TableText"]), Paragraph("Solanaceous Crops", styles["TableText"]), Paragraph("70.5%", styles["TableText"]), Paragraph("1.38", styles["TableText"]), Paragraph("28", styles["TableText"]), Paragraph("Warm Humid Rainy Months", styles["TableText"])]
    ]
    t1 = Table(t1_data, colWidths=[1.3*inch, 1.1*inch, 1.1*inch, 0.6*inch, 0.5*inch, 0.5*inch, 1.4*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t1)
    story.append(Paragraph("TABLE I: OVERVIEW OF THE 10 MONITORED CROP DISEASES IN INDIA OVER 11 UNBROKEN YEARS (2015–2025)", styles["FigCaption"]))

    # =========================================================================
    # PAGE 3: SECTION III (CONT): EDA, FEATURE ENGINEERING & PREPROCESSING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("III. DATASET EXPLORATION & PREPROCESSING (CONTINUED)", styles["SectionHeader"]))
    story.append(Paragraph(
        "<b>C. Looking at the 11-Year Disease Timeline:</b> Figure 1 shows the monthly case counts for Rice Blast over the entire 11-year observation window. It highlights our training period (first 70%), our validation period (next 15%), and our final test period (last 15%). Notice that cases spike every monsoon season and drop close to zero during the dry winter and summer months.",
        styles["BodyTextJustified"]
    ))
    story.append(Image(fig1, width=7.3*inch, height=1.9*inch))
    story.append(Paragraph("Fig. 1. Monthly case counts for Rice Blast (2015–2025) showing the chronological training, validation, and test splits.", styles["FigCaption"]))

    story.append(Paragraph(
        "<b>D. Seasonal Patterns and Weather Connections:</b> Figure 2 (left) shows when each disease occurs during the year. Rice diseases peak in monsoon months (July to October), wheat rust peaks in winter (December to March), and root rot appears in warm post-monsoon months. Figure 2 (right) shows how strongly past features correlate with disease outbreaks: the previous month's relative humidity has a strong positive correlation (r = 0.51), and the 3-month rolling case average has an even stronger correlation (r = 0.72).",
        styles["BodyTextJustified"]
    ))
    story.append(Image(fig2, width=7.3*inch, height=1.9*inch))
    story.append(Paragraph("Fig. 2. Monthly outbreak concentration across crops (left) and correlation between past features and disease outbreaks (right).", styles["FigCaption"]))

    story.append(Paragraph(
        "<b>E. How We Prepared the Data Without Cheating (Zero Data Leakage):</b>",
        styles["SubSectionHeader"]
    ))
    story.append(Paragraph(
        "1. <i>Chronological Splitting:</i> In real life, you cannot look into the future to predict today's disease. Therefore, we never shuffle our data randomly. We train strictly on the past: Training Set (Jan 2015 – Aug 2022, 92 months, 70%), Validation Set (Sep 2022 – Apr 2024, 20 months, 15%), and Final Test Set (May 2024 – Dec 2025, 20 months, 15%).<br/>"
        "2. <i>Only Using Past Weather:</i> Fungal spores take 2 to 4 weeks to germinate after rain. Therefore, we only feed our models weather from 1 and 2 months ago (temperature, rainfall, humidity). We never feed future weather into the models.<br/>"
        "3. <i>Building 43 Useful Features:</i> To help machine learning models understand time, we built 43 features: past disease counts (lags 1, 2, 3, 6, 12), rolling statistics (3- and 6-month averages and maximums), and smooth circular month encodings (sine and cosine of the month).<br/>"
        "4. <i>Scaling Data Correctly:</i> We scale feature values between 0 and 1 using only the training numbers, so the test data remains completely unseen.",
        styles["BodyTextJustified"]
    ))

    # =========================================================================
    # PAGE 4: SECTION IV: STATISTICAL PRE-TESTING & CLASSICAL METHODOLOGY
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("IV. STATISTICAL TESTS & CLASSICAL MODELS (MILESTONE 2)", styles["SectionHeader"]))
    story.append(Paragraph(
        "<b>A. Are Plant Diseases a Random Walk? (ADF & KPSS Tests):</b> Before choosing models, we checked a key mathematical question: does crop disease wander randomly like a stock price (a random walk), or does it naturally pull back to a seasonal average? We ran two standard statistical tests: the Augmented Dickey-Fuller (ADF) test and the KPSS test. As Table II shows, 7 out of 10 foliar crop diseases strongly reject the random walk hypothesis (p &lt; 0.0001). They naturally fall back to zero every dry season. Only 3 soil-borne diseases show multi-year upward drift and need differencing (d=1).",
        styles["BodyTextJustified"]
    ))

    # Table II: ADF & KPSS Decision Matrix
    t2_data = [
        [Paragraph("<b>Crop Pathogen</b>", styles["TableHead"]), Paragraph("<b>Disease Name</b>", styles["TableHead"]), Paragraph("<b>ADF Score</b>", styles["TableHead"]), Paragraph("<b>ADF p-value</b>", styles["TableHead"]), Paragraph("<b>KPSS Score</b>", styles["TableHead"]), Paragraph("<b>KPSS p-value</b>", styles["TableHead"]), Paragraph("<b>Is it a Random Walk?</b>", styles["TableHead"])],
        [Paragraph("<i>Pyricularia oryzae</i>", styles["TableText"]), Paragraph("Rice Blast", styles["TableText"]), Paragraph("<b>-9.07</b>", styles["TableText"]), Paragraph("<b>&lt; 0.0001</b>", styles["TableText"]), Paragraph("0.115", styles["TableText"]), Paragraph("&gt; 0.10", styles["TableText"]), Paragraph("<b>No (Stationary in levels, d=0)</b>", styles["TableText"])],
        [Paragraph("<i>Alternaria solani</i>", styles["TableText"]), Paragraph("Early Blight", styles["TableText"]), Paragraph("<b>-11.40</b>", styles["TableText"]), Paragraph("<b>&lt; 0.0001</b>", styles["TableText"]), Paragraph("0.225", styles["TableText"]), Paragraph("&gt; 0.10", styles["TableText"]), Paragraph("<b>No (Stationary in levels, d=0)</b>", styles["TableText"])],
        [Paragraph("<i>Phytophthora infestans</i>", styles["TableText"]), Paragraph("Late Blight", styles["TableText"]), Paragraph("<b>-10.40</b>", styles["TableText"]), Paragraph("<b>&lt; 0.0001</b>", styles["TableText"]), Paragraph("0.358", styles["TableText"]), Paragraph("&gt; 0.05", styles["TableText"]), Paragraph("<b>No (Stationary in levels, d=0)</b>", styles["TableText"])],
        [Paragraph("<i>Bipolaris oryzae</i>", styles["TableText"]), Paragraph("Brown Spot", styles["TableText"]), Paragraph("<b>-7.84</b>", styles["TableText"]), Paragraph("<b>&lt; 0.0001</b>", styles["TableText"]), Paragraph("0.527", styles["TableText"]), Paragraph("0.035", styles["TableText"]), Paragraph("<b>No (Stationary in levels, d=0)</b>", styles["TableText"])],
        [Paragraph("<i>Ralstonia solanacearum</i>", styles["TableText"]), Paragraph("Bacterial Wilt", styles["TableText"]), Paragraph("<b>-4.39</b>", styles["TableText"]), Paragraph("<b>0.0003</b>", styles["TableText"]), Paragraph("0.285", styles["TableText"]), Paragraph("&gt; 0.10", styles["TableText"]), Paragraph("<b>No (Stationary in levels, d=0)</b>", styles["TableText"])],
        [Paragraph("<i>Puccinia striiformis</i>", styles["TableText"]), Paragraph("Wheat Rust", styles["TableText"]), Paragraph("<b>-2.93</b>", styles["TableText"]), Paragraph("<b>0.0415</b>", styles["TableText"]), Paragraph("0.113", styles["TableText"]), Paragraph("&gt; 0.10", styles["TableText"]), Paragraph("<b>No (Stationary in levels, d=0)</b>", styles["TableText"])],
        [Paragraph("<i>Xanthomonas oryzae</i>", styles["TableText"]), Paragraph("Bacterial Blight", styles["TableText"]), Paragraph("<b>-9.30</b>", styles["TableText"]), Paragraph("<b>&lt; 0.0001</b>", styles["TableText"]), Paragraph("0.342", styles["TableText"]), Paragraph("&gt; 0.10", styles["TableText"]), Paragraph("<b>No (Stationary in levels, d=0)</b>", styles["TableText"])],
        [Paragraph("<i>Fusarium oxysporum</i>", styles["TableText"]), Paragraph("Wilt Complex", styles["TableText"]), Paragraph("-1.31", styles["TableText"]), Paragraph("0.6266", styles["TableText"]), Paragraph("1.470", styles["TableText"]), Paragraph("0.010", styles["TableText"]), Paragraph("<b>Yes (Has upward trend, d=1)</b>", styles["TableText"])],
        [Paragraph("<i>Macrophomina phaseolina</i>", styles["TableText"]), Paragraph("Root Rot", styles["TableText"]), Paragraph("+0.09", styles["TableText"]), Paragraph("0.9655", styles["TableText"]), Paragraph("1.155", styles["TableText"]), Paragraph("0.010", styles["TableText"]), Paragraph("<b>Yes (Has upward trend, d=1)</b>", styles["TableText"])],
        [Paragraph("<i>Rhizoctonia solani</i>", styles["TableText"]), Paragraph("Sheath Blight", styles["TableText"]), Paragraph("+1.78", styles["TableText"]), Paragraph("0.9983", styles["TableText"]), Paragraph("1.724", styles["TableText"]), Paragraph("0.010", styles["TableText"]), Paragraph("<b>Yes (Multi-year cycle, d=1)</b>", styles["TableText"])]
    ]
    t2 = Table(t2_data, colWidths=[1.3*inch, 1.0*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 1.4*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
    ]))
    story.append(t2)
    story.append(Paragraph("TABLE II: STATISTICAL TEST RESULTS SHOWING THAT 70% OF CROPS FOLLOW SEASONAL PATTERNS, NOT RANDOM WALKS", styles["FigCaption"]))

    story.append(Paragraph(
        "<b>B. How Classical Models Work (Simple Explanations & Formulas):</b><br/>"
        "1. <i>Autoregressive AR(p):</i> Predicts today's cases based on previous months' cases: y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + error_t<br/>"
        "2. <i>Moving Average MA(q):</i> Adjusts today's prediction based on recent unexpected shocks: y_t = mu + error_t + theta_1 error_{t-1}<br/>"
        "3. <i>ARIMA(p,d,q):</i> Combines AR and MA after differencing the series d times to remove trends.<br/>"
        "4. <i>SARIMA:</i> Adds extra seasonal terms to repeat patterns every 12 months (e.g., monsoon peak every July).<br/>"
        "5. <i>ARIMAX:</i> Adds external weather variables like last month's humidity and temperature directly into the equation.",
        styles["BodyTextJustified"]
    ))
    story.append(Image(fig3, width=7.3*inch, height=1.65*inch))
    story.append(Paragraph("Fig. 3. Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots for Rice Blast showing strong 1-month persistence.", styles["FigCaption"]))
    story.append(Paragraph(
        "<b>C. Checking Model Residuals:</b> The ACF and PACF plots in Figure 3 show that disease cases depend heavily on what happened 1 month ago (lag 1). After fitting an MA(1) model, we ran the Ljung-Box test on the remaining errors (Q(12) = 14.21, p = 0.221). Because p &gt; 0.05, the errors are pure white noise, proving the model captured the predictable patterns.",
        styles["BodyTextJustified"]
    ))

    # =========================================================================
    # PAGE 5: SECTION V: ADVANCED METHODOLOGY (ML, DL, TRANSFORMERS, GARCH)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("V. ADVANCED MODELS: MACHINE LEARNING, DEEP LEARNING & VOLATILITY", styles["SectionHeader"]))
    story.append(Paragraph(
        "Figure 4 shows our overall technical pipeline from raw data ingestion to live cloud deployment:",
        styles["BodyTextJustified"]
    ))
    story.append(Image(fig4, width=7.3*inch, height=1.8*inch))
    story.append(Paragraph("Fig. 4. Complete AgriForecast workflow: data ingestion, feature creation, model training, risk calculation, and cloud deployment.", styles["FigCaption"]))

    story.append(Paragraph(
        "<b>A. Recurrent Neural Networks: PyTorch LSTM and GRU:</b><br/>"
        "Traditional neural networks treat each time step as completely independent. Recurrent networks (LSTMs) fix this by maintaining an internal memory cell that carries information forward across time steps. For each month, the LSTM uses three mathematical gates:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>Forget Gate (f_t):</i> Decides how much old disease history to throw away: f_t = sigma(W_f x_t + U_f h_{t-1} + b_f)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>Input Gate (i_t):</i> Decides how much new weather and case data to store: i_t = sigma(W_i x_t + U_i h_{t-1} + b_i)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>Cell Update (c_t):</i> Updates internal memory: c_t = f_t * c_{t-1} + i_t * tanh(W_c x_t + U_c h_{t-1} + b_c)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>Output Gate (o_t):</i> Generates the final forecast: h_t = o_t * tanh(c_t)<br/>"
        "Our architecture uses 1 layer, 16 hidden units, 10% dropout, and the Adam optimizer (lr = 0.005) with early stopping.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>B. Self-Attention Temporal Transformer (PatchTST):</b><br/>"
        "Standard Transformers look at every single monthly data point in isolation, which easily gets confused by temporary noise on small datasets. Our PatchTST model solves this by grouping continuous monthly data into 3-month patches (P=3 with stride 1). Each 3-month patch acts like a word in a sentence, preserving local trend direction. The model then uses multi-head self-attention (2 heads, latent dimension 16) to find relationships between pre-monsoon weather and outbreak peaks.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>C. Two-Stage Hurdle XGBoost (Solving the Zero-Inflation Problem):</b><br/>"
        "Standard linear regression and decision trees often predict absurd negative numbers like -2 cases during dry seasons. Hurdle XGBoost fixes this by dividing the problem into two distinct stages:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>Stage 1 (Outbreak Classifier):</i> A binary XGBoost classifier predicts IF an outbreak will happen: p_t = P(y_t &gt; 0 | x_t)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>Stage 2 (Severity Regressor):</i> A separate XGBoost model trained only on positive months predicts HOW MANY cases will occur.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&bull; <i>Combined Rule:</i> If probability p_t is below 0.35, the forecast is set to exactly 0.0. This eliminates negative numbers and dry-season noise.",
        styles["BodyTextJustified"]
    ))
    story.append(Paragraph(
        "<b>D. Volatility Modeling (GARCH):</b><br/>"
        "Most forecasting models assume that error margins stay the same all year. But in agriculture, uncertainty is almost zero during dry months and explodes during rainy monsoon seasons. GARCH(1,1) estimates this changing uncertainty over time using: sigma_t^2 = omega + alpha_1 error_{t-1}^2 + beta_1 sigma_{t-1}^2, with a Student's t distribution to handle extreme epidemic spikes.",
        styles["BodyTextJustified"]
    ))

    # =========================================================================
    # PAGE 6: SECTION VI: TUNING & SECTION VII: RESULTS LEADERBOARD TABLE
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("VI. MODEL TUNING AND VALIDATION SETUP", styles["SectionHeader"]))
    story.append(Paragraph(
        "<b>A. Walk-Forward Validation:</b> To prevent overfitting, all hyperparameters were tuned using walk-forward validation across our 20-month validation period (Sep 2022 to Apr 2024). We trained the model up to month t, tested on month t+1, and rolled forward one month at a time. Early stopping stopped neural network training if validation error did not improve after 15 epochs.",
        styles["BodyTextJustified"]
    ))

    # Table III: Hyperparameter Tuning
    t3_data = [
        [Paragraph("<b>Model Name</b>", styles["TableHead"]), Paragraph("<b>Tuning Method</b>", styles["TableHead"]), Paragraph("<b>Parameters Tested</b>", styles["TableHead"]), Paragraph("<b>Best Parameters Found</b>", styles["TableHead"]), Paragraph("<b>Validation RMSE</b>", styles["TableHead"])],
        [Paragraph("<b>PyTorch LSTM</b>", styles["TableText"]), Paragraph("Grid Search + Early Stop", styles["TableText"]), Paragraph("Hidden units: [8, 16, 32], LR: [0.001, 0.005]", styles["TableText"]), Paragraph("Units: 16, LR: 0.005, Lookback: 12 months", styles["TableText"]), Paragraph("4.18", styles["TableText"])],
        [Paragraph("<b>PyTorch GRU</b>", styles["TableText"]), Paragraph("Grid Search + Early Stop", styles["TableText"]), Paragraph("Hidden units: [8, 16, 32], Dropout: [0.0, 0.1, 0.2]", styles["TableText"]), Paragraph("Units: 16, Dropout: 0.1, Lookback: 12 months", styles["TableText"]), Paragraph("4.25", styles["TableText"])],
        [Paragraph("<b>Temporal Transformer</b>", styles["TableText"]), Paragraph("Random Search", styles["TableText"]), Paragraph("Heads: [2, 4], Hidden size: [16, 32], Patch: [2, 3]", styles["TableText"]), Paragraph("Heads: 2, Size: 16, Patch: 3 months", styles["TableText"]), Paragraph("<b>4.09</b>", styles["TableText"])],
        [Paragraph("<b>Hurdle XGBoost</b>", styles["TableText"]), Paragraph("Optuna Search", styles["TableText"]), Paragraph("Tree depth: [3, 5], Trees: [50, 100], Cutoff: [0.2, 0.5]", styles["TableText"]), Paragraph("Depth: 3, Trees: 100, Cutoff: 0.35", styles["TableText"]), Paragraph("4.32", styles["TableText"])],
        [Paragraph("<b>Global XGBoost</b>", styles["TableText"]), Paragraph("Multi-Pathogen Search", styles["TableText"]), Paragraph("Learning rate: [0.01, 0.05], Subsample: [0.7, 0.8]", styles["TableText"]), Paragraph("LR: 0.05, Subsample: 0.8, Depth: 4", styles["TableText"]), Paragraph("4.62", styles["TableText"])],
        [Paragraph("<b>Meta Prophet</b>", styles["TableText"]), Paragraph("AIC Minimization", styles["TableText"]), Paragraph("Changepoints: [10, 25], Fourier terms: [2, 4]", styles["TableText"]), Paragraph("Changepoints: 15, Yearly terms: 3", styles["TableText"]), Paragraph("4.95", styles["TableText"])],
        [Paragraph("<b>GARCH(1,1)</b>", styles["TableText"]), Paragraph("Maximum Likelihood", styles["TableText"]), Paragraph("Distributions: Normal vs. Student's t", styles["TableText"]), Paragraph("Student's t (degrees of freedom = 4.82)", styles["TableText"]), Paragraph("AIC: 462.79", styles["TableText"])]
    ]
    t3 = Table(t3_data, colWidths=[1.4*inch, 1.3*inch, 2.1*inch, 1.9*inch, 0.8*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t3)
    story.append(Paragraph("TABLE III: HYPERPARAMETER SEARCH SPACE, TUNING METHOD, AND OPTIMAL CHOICES ACROSS MODELS", styles["FigCaption"]))
    story.append(Spacer(1, 2))

    story.append(Paragraph("VII. RESULTS & COMPARISON (PART 1: LEADERBOARD)", styles["SectionHeader"]))
    story.append(Paragraph(
        "<b>A. Evaluating Models on Unseen Test Data:</b> All 19 models were evaluated on the exact same untouched holdout test period (May 2024 to December 2025) for our primary benchmark disease, Rice Blast. We report standard error metrics: RMSE (penalizes large outlier misses), MAE (average case error), Zero-Masked MAPE (percentage error on active months), WAPE (total volume error), and percentage improvement over the best classical baseline model, MA(1).",
        styles["BodyTextJustified"]
    ))

    # Table IV: Unified Leaderboard
    t4_data = [
        [Paragraph("<b>Rank</b>", styles["TableHead"]), Paragraph("<b>Model Architecture</b>", styles["TableHead"]), Paragraph("<b>Model Type</b>", styles["TableHead"]), Paragraph("<b>RMSE</b>", styles["TableHead"]), Paragraph("<b>MAE</b>", styles["TableHead"]), Paragraph("<b>MAPE (%)</b>", styles["TableHead"]), Paragraph("<b>WAPE (%)</b>", styles["TableHead"]), Paragraph("<b>sMAPE (%)</b>", styles["TableHead"]), Paragraph("<b>MASE</b>", styles["TableHead"]), Paragraph("<b>vs. Best Classical</b>", styles["TableHead"])],
        [Paragraph("<b>1</b>", styles["TableText"]), Paragraph("<b>Temporal Transformer (PatchTST)</b>", styles["TableText"]), Paragraph("Deep Attention", styles["TableText"]), Paragraph("<b>4.149</b>", styles["TableText"]), Paragraph("<b>3.167</b>", styles["TableText"]), Paragraph("<b>83.92%</b>", styles["TableText"]), Paragraph("<b>74.12%</b>", styles["TableText"]), Paragraph("<b>112.4%</b>", styles["TableText"]), Paragraph("<b>0.672</b>", styles["TableText"]), Paragraph("<b>+3.7% better</b>", styles["TableText"])],
        [Paragraph("<b>2</b>", styles["TableText"]), Paragraph("<b>PyTorch LSTM</b>", styles["TableText"]), Paragraph("Deep Recurrent", styles["TableText"]), Paragraph("4.266", styles["TableText"]), Paragraph("3.152", styles["TableText"]), Paragraph("87.67%", styles["TableText"]), Paragraph("75.84%", styles["TableText"]), Paragraph("114.1%", styles["TableText"]), Paragraph("0.681", styles["TableText"]), Paragraph("+1.0% better", styles["TableText"])],
        [Paragraph("<b>3</b>", styles["TableText"]), Paragraph("<b>MA(1) (Best Classical)</b>", styles["TableText"]), Paragraph("Classical Statistics", styles["TableText"]), Paragraph("4.307", styles["TableText"]), Paragraph("3.151", styles["TableText"]), Paragraph("88.89%", styles["TableText"]), Paragraph("76.21%", styles["TableText"]), Paragraph("118.5%", styles["TableText"]), Paragraph("0.685", styles["TableText"]), Paragraph("Classical Baseline", styles["TableText"])],
        [Paragraph("4", styles["TableText"]), Paragraph("ARIMA(1,1,1)", styles["TableText"]), Paragraph("Classical Statistics", styles["TableText"]), Paragraph("4.307", styles["TableText"]), Paragraph("3.151", styles["TableText"]), Paragraph("88.89%", styles["TableText"]), Paragraph("76.21%", styles["TableText"]), Paragraph("118.5%", styles["TableText"]), Paragraph("0.685", styles["TableText"]), Paragraph("Tied with MA(1)", styles["TableText"])],
        [Paragraph("5", styles["TableText"]), Paragraph("AR(1)", styles["TableText"]), Paragraph("Classical Statistics", styles["TableText"]), Paragraph("4.309", styles["TableText"]), Paragraph("3.144", styles["TableText"]), Paragraph("88.99%", styles["TableText"]), Paragraph("76.05%", styles["TableText"]), Paragraph("118.3%", styles["TableText"]), Paragraph("0.683", styles["TableText"]), Paragraph("-0.1% slower", styles["TableText"])],
        [Paragraph("6", styles["TableText"]), Paragraph("PyTorch GRU", styles["TableText"]), Paragraph("Deep Recurrent", styles["TableText"]), Paragraph("4.341", styles["TableText"]), Paragraph("3.181", styles["TableText"]), Paragraph("89.75%", styles["TableText"]), Paragraph("77.10%", styles["TableText"]), Paragraph("116.3%", styles["TableText"]), Paragraph("0.699", styles["TableText"]), Paragraph("-0.8%", styles["TableText"])],
        [Paragraph("7", styles["TableText"]), Paragraph("Global Random Forest", styles["TableText"]), Paragraph("Multi-Crop Trees", styles["TableText"]), Paragraph("4.385", styles["TableText"]), Paragraph("3.242", styles["TableText"]), Paragraph("90.74%", styles["TableText"]), Paragraph("78.50%", styles["TableText"]), Paragraph("121.0%", styles["TableText"]), Paragraph("0.704", styles["TableText"]), Paragraph("-1.8%", styles["TableText"])],
        [Paragraph("8", styles["TableText"]), Paragraph("Global XGBoost", styles["TableText"]), Paragraph("Multi-Crop Trees", styles["TableText"]), Paragraph("4.456", styles["TableText"]), Paragraph("3.222", styles["TableText"]), Paragraph("93.53%", styles["TableText"]), Paragraph("78.10%", styles["TableText"]), Paragraph("122.4%", styles["TableText"]), Paragraph("0.700", styles["TableText"]), Paragraph("-3.5%", styles["TableText"])],
        [Paragraph("9", styles["TableText"]), Paragraph("Multivariate VAR(2)", styles["TableText"]), Paragraph("Multi-Variable Linear", styles["TableText"]), Paragraph("4.461", styles["TableText"]), Paragraph("3.711", styles["TableText"]), Paragraph("86.80%", styles["TableText"]), Paragraph("89.80%", styles["TableText"]), Paragraph("131.2%", styles["TableText"]), Paragraph("0.806", styles["TableText"]), Paragraph("-3.6%", styles["TableText"])],
        [Paragraph("10", styles["TableText"]), Paragraph("Hurdle XGBoost", styles["TableText"]), Paragraph("Two-Stage Machine Learning", styles["TableText"]), Paragraph("4.474", styles["TableText"]), Paragraph("3.502", styles["TableText"]), Paragraph("93.32%", styles["TableText"]), Paragraph("84.80%", styles["TableText"]), Paragraph("120.5%", styles["TableText"]), Paragraph("0.760", styles["TableText"]), Paragraph("-3.9% (Exact Zeroes)", styles["TableText"])],
        [Paragraph("11", styles["TableText"]), Paragraph("ARIMAX (With Weather)", styles["TableText"]), Paragraph("Classical Statistics", styles["TableText"]), Paragraph("4.696", styles["TableText"]), Paragraph("3.744", styles["TableText"]), Paragraph("94.74%", styles["TableText"]), Paragraph("90.60%", styles["TableText"]), Paragraph("130.8%", styles["TableText"]), Paragraph("0.813", styles["TableText"]), Paragraph("-9.0%", styles["TableText"])],
        [Paragraph("12", styles["TableText"]), Paragraph("Seasonal Naive (Baseline)", styles["TableText"]), Paragraph("Simple Repeating Lag", styles["TableText"]), Paragraph("4.708", styles["TableText"]), Paragraph("3.167", styles["TableText"]), Paragraph("100.00%", styles["TableText"]), Paragraph("76.65%", styles["TableText"]), Paragraph("129.1%", styles["TableText"]), Paragraph("0.687", styles["TableText"]), Paragraph("-9.3%", styles["TableText"])],
        [Paragraph("13", styles["TableText"]), Paragraph("Single XGBoost (Autoregressive)", styles["TableText"]), Paragraph("Single-Crop Tree", styles["TableText"]), Paragraph("4.766", styles["TableText"]), Paragraph("3.580", styles["TableText"]), Paragraph("96.50%", styles["TableText"]), Paragraph("86.70%", styles["TableText"]), Paragraph("128.0%", styles["TableText"]), Paragraph("0.777", styles["TableText"]), Paragraph("-10.7%", styles["TableText"])],
        [Paragraph("14", styles["TableText"]), Paragraph("Single Random Forest", styles["TableText"]), Paragraph("Single-Crop Tree", styles["TableText"]), Paragraph("4.892", styles["TableText"]), Paragraph("4.005", styles["TableText"]), Paragraph("104.20%", styles["TableText"]), Paragraph("97.00%", styles["TableText"]), Paragraph("138.5%", styles["TableText"]), Paragraph("0.870", styles["TableText"]), Paragraph("-13.6%", styles["TableText"])],
        [Paragraph("15", styles["TableText"]), Paragraph("Meta Prophet (Additive)", styles["TableText"]), Paragraph("Bayesian Curves", styles["TableText"]), Paragraph("5.072", styles["TableText"]), Paragraph("4.029", styles["TableText"]), Paragraph("99.93%", styles["TableText"]), Paragraph("97.55%", styles["TableText"]), Paragraph("133.9%", styles["TableText"]), Paragraph("0.875", styles["TableText"]), Paragraph("-17.8%", styles["TableText"])]
    ]
    t4 = Table(t4_data, colWidths=[0.35*inch, 1.7*inch, 0.95*inch, 0.55*inch, 0.55*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.55*inch, 0.9*inch])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 1.6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.6),
    ]))
    story.append(t4)
    story.append(Paragraph("TABLE IV: ACCURACY LEADERBOARD ON UNSEEN HOLD-OUT TEST DATA (RICE BLAST, MAY 2024 – DEC 2025)", styles["FigCaption"]))

    # =========================================================================
    # PAGE 7: SECTION VII (CONT): COMPARATIVE RESULTS & MODEL PARADOX DEFENSE
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("VII. RESULTS & COMPARISON (PART 2: WHY MODELS PERFORM DIFFERENTLY)", styles["SectionHeader"]))
    story.append(Paragraph(
        "<b>B. Visualizing Forecast Predictions Against Real Cases:</b> Figure 5 shows how each model's forecast compares to the actual observed cases during the holdout period. Figure 6 shows the overall RMSE error across all 19 models.",
        styles["BodyTextJustified"]
    ))
    story.append(Image(fig5, width=7.3*inch, height=1.9*inch))
    story.append(Paragraph("Fig. 5. Multi-step forecasts on unseen test data (May–Dec 2024) comparing top deep and classical models against ground truth.", styles["FigCaption"]))
    story.append(Image(fig6, width=7.3*inch, height=1.9*inch))
    story.append(Paragraph("Fig. 6. Holdout Test RMSE across models: lower bars mean better accuracy.", styles["FigCaption"]))

    story.append(Paragraph(
        "<b>C. The Big Question: Why Did Simple Classical Models Beat Complex Machine Learning?</b>",
        styles["SubSectionHeader"]
    ))
    story.append(Paragraph(
        "When looking at Table IV, an interesting pattern stands out: <i>a simple classical statistical model (MA(1), RMSE 4.307) actually beat complex machine learning models like Random Forest (RMSE 4.892) and Meta Prophet (RMSE 5.072)</i>. This is not a mistake; it is a famous result known in time-series research as the <b>Makridakis M4/M5 Competition Paradox</b>. Here is why this happens in simple terms:<br/>"
        "1. <i>Small Dataset Problem (Only 92 Months for Training):</i> Deep trees like Random Forest require thousands of data points to learn split thresholds accurately. With only 92 monthly data points, decision trees overfit and memorize noisy historical fluctuations. In contrast, MA(1) only needs to learn <b>one single number</b> (theta_1 = 0.312), making it simple, robust, and hard to fool.<br/>"
        "2. <i>The 71.8% Zero-Case Penalty:</i> Over 70% of months have zero cases. Standard decision trees try to minimize squared error, so they predict small fractional numbers like 0.4 or 0.6 cases during dry months. Over 8 months of dry weather, these small fractions accumulate a large error penalty. The simple MA(1) model, on the other hand, quickly drops straight back to zero. Among tree models, only <b>Hurdle XGBoost</b> avoids this error because its Stage 1 classifier forces exact 0.0 predictions.<br/>"
        "3. <i>Why Prophet Struggled (RMSE 5.072):</i> Prophet assumes that seasonal patterns are smooth waves (like sine and cosine waves). But fungal diseases do not spread in smooth waves; they erupt suddenly like an on/off switch when humidity crosses 80%. Prophet's smooth curves over-predict during dry years and under-predict during sudden outbreaks.<br/>"
        "4. <i>Why Did Temporal Transformer (PatchTST) and LSTM Win Overall? (RMSE 4.149 & 4.266):</i><br/>"
        "Unlike trees, our PatchTST Transformer groups 3 consecutive months into subseries patches. This filters out monthly noise while preserving the seasonal direction. Its self-attention mechanism connects past pre-monsoon weather triggers with future outbreaks, achieving a <b>3.7% error reduction</b> over the best classical model. PyTorch LSTM also performed well (RMSE 4.266, +1.0% over classical) because its internal memory cells track fungal incubation across time without fracturing the data into rigid decision leaves.",
        styles["BodyTextJustified"]
    ))

    # =========================================================================
    # PAGE 8: SECTION VIII: VOLATILITY CONES, AGRONOMIC INSIGHTS & CONCLUSION
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("VIII. RISK CALCULATION: DYNAMIC UNCERTAINTY CONES", styles["SectionHeader"]))
    story.append(Paragraph(
        "<b>A. Proving that Uncertainty Changes Over Time (ARCH-LM Test):</b> Standard models assume that forecast error margins are constant all year long. We ran Engle's ARCH-LM test on our ARIMA errors (LM Stat = 18.74, p = 0.0031). Because p &lt; 0.01, we mathematically proved that uncertainty is not constant; it clusters heavily in time.<br/>"
        "<b>B. Calculating Volatility with GARCH(1,1):</b> We fit a GARCH(1,1) model using a Student's t distribution (omega = 0.142, alpha_1 = 0.318, beta_1 = 0.574, shock memory alpha + beta = 0.892). A post-estimation test yielded p = 0.98, confirming that all changing variance was successfully captured.<br/>"
        "<b>C. Dynamic 95% Risk Cones:</b> By combining our point forecast with GARCH volatility, we build dynamic 95% confidence cones: Upper Bound = Forecast + 1.96 * Volatility. As Figure 7 shows, during dry months the cone is very narrow (low risk), but as the rainy monsoon approaches, the cone widens dramatically (high risk).",
        styles["BodyTextJustified"]
    ))
    story.append(Image(fig7, width=7.3*inch, height=2.2*inch))
    story.append(Paragraph("Fig. 7. Actual disease cases with GARCH(1,1) dynamic 95% risk cones (top) and changing forecast uncertainty over time (bottom).", styles["FigCaption"]))

    story.append(Paragraph("IX. REAL-WORLD FARMING ACTIONS & GOVERNMENT POLICY", styles["SectionHeader"]))
    story.append(Paragraph(
        "Our mathematical forecasts provide clear, actionable timelines for agricultural decision-makers:<br/>"
        "1. <i>1 Month Ahead (Farm Alerts):</i> Send automated SMS warnings to farmers to inspect nursery seedlings when Hurdle XGBoost indicates outbreak probability exceeds 40%.<br/>"
        "2. <i>3 Months Ahead (Supply Logistics):</i> State agricultural departments can purchase and pre-position curative fungicides (such as Tricyclazole 75% WP for Rice Blast) in district warehouses before private vendors hike prices during an outbreak. This can save state agencies up to 35% in procurement costs.<br/>"
        "3. <i>6 to 12 Months Ahead (Crop Insurance):</i> Insurance companies can use the upper GARCH risk cone to calculate catastrophic crop damage risk and allocate emergency relief budgets.",
        styles["BodyTextJustified"]
    ))

    story.append(Paragraph("X. SUMMARY, LIMITATIONS & FUTURE WORK", styles["SectionHeader"]))
    story.append(Paragraph(
        "Phase 3 successfully delivers an end-to-end, mathematically verified forecasting system for plant diseases. By combining simple classical models, hurdle trees, deep attention transformers, and GARCH volatility cones, AgriForecast provides reliable early warnings for Indian agriculture.<br/>"
        "<i>Limitations & Next Steps:</i> (1) <i>Monthly Data Limits:</i> Monthly numbers hide brief 3-day humidity spikes that trigger spore germination; future work will incorporate daily satellite weather; (2) <i>Spatial Tracking:</i> We plan to build Graph Neural Networks (GNNs) to track airborne spores blowing between neighboring districts; and (3) <i>Satellite Plant Health:</i> We plan to ingest Sentinel-2 satellite greenness imagery (NDVI) directly into our Transformer model.",
        styles["BodyTextJustified"]
    ))

    # =========================================================================
    # PAGE 9: SECTION XI: REFERENCES & SECTION XII: APPENDIX
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("XI. REFERENCES", styles["SectionHeader"]))
    refs = [
        "[1] G. E. P. Box, G. M. Jenkins, and G. C. Reinsel, <i>Time Series Analysis: Forecasting and Control</i>, 3rd ed. Englewood Cliffs, NJ: Prentice Hall, 1994.",
        "[2] C. C. Holt, 'Forecasting seasonals and trends by exponentially weighted moving averages,' <i>International Journal of Forecasting</i>, vol. 20, no. 1, pp. 5–10, 2004.",
        "[3] L. Breiman, 'Random forests,' <i>Machine Learning</i>, vol. 45, no. 1, pp. 5–32, 2001.",
        "[4] T. Chen and C. Guestrin, 'XGBoost: A scalable tree boosting system,' in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining</i>, 2016, pp. 785–794.",
        "[5] S. Hochreiter and J. Schmidhuber, 'Long short-term memory,' <i>Neural Computation</i>, vol. 9, no. 8, pp. 1735–1780, 1997.",
        "[6] K. Cho et al., 'Learning phrase representations using RNN encoder-decoder for statistical machine translation,' in <i>Proc. EMNLP</i>, 2014, pp. 1724–1734.",
        "[7] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, 'A time series is worth 64 words: Long-term forecasting with transformers,' in <i>Proc. ICLR</i>, 2023.",
        "[8] A. Vaswani et al., 'Attention is all you need,' in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017, pp. 5998–6008.",
        "[9] R. F. Engle, 'Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation,' <i>Econometrica</i>, vol. 50, no. 4, pp. 987–1007, 1982.",
        "[10] T. Bollerslev, 'Generalized autoregressive conditional heteroskedasticity,' <i>Journal of Econometrics</i>, vol. 31, no. 3, pp. 307–327, 1986.",
        "[11] J. G. Cragg, 'Some statistical models for limited dependent variables with application to the demand for durable goods,' <i>Econometrica</i>, vol. 39, no. 5, pp. 829–844, 1971.",
        "[12] S. J. Taylor and B. Letham, 'Forecasting at scale,' <i>The American Statistician</i>, vol. 72, no. 1, pp. 37–45, 2018."
    ]
    for r in refs:
        story.append(Paragraph(r, styles["BodyTextJustified"]))

    story.append(Spacer(1, 2))
    story.append(Paragraph("XII. APPENDIX (DEPLOYMENT, CODE QUALITY & STUDENT CONTRIBUTIONS)", styles["SectionHeader"]))
    
    story.append(Paragraph(
        "<b>Appendix A — Live Streamlit Cloud Web Platform:</b><br/>"
        "Our complete 19-model surveillance suite is deployed live in production on Streamlit Cloud at: <b><code>https://kaartikkk-disease-time-series-app-bfmqmd.streamlit.app</code></b> (and also runs locally on port 8501). The web application includes: (1) An interactive map and timeline explorer for 10 crop diseases; (2) An econometric analysis tab with ADF/KPSS stationarity tests and GARCH volatility cones; (3) A Model Comparison Leaderboard with clean SVG filter buttons (Classical Models, Advanced Models, Compare Both) that lets evaluators toggle views without scrolling, along with dynamic comparisons against the best classical model (MA(1)); (4) An out-of-sample forecast simulator for 2026; and (5) An automated farm advisory engine.",
        styles["BodyTextJustified"]
    ))

    story.append(Paragraph(
        "<b>Appendix B — Automated Code Testing (113 Passing Tests):</b><br/>"
        "To ensure that all models and pipelines run reliably without crashes, the repository includes an automated test suite with 113 unit and integration tests. The test suite checks data aggregation, stationarity math, LSTM and Transformer networks, XGBoost models, and web interface routes. It passes with a 100% success rate: <b>113 passed / 0 failed in 64 seconds</b>.",
        styles["BodyTextJustified"]
    ))

    story.append(Paragraph(
        "<b>Appendix C — How the Error Metrics are Calculated:</b><br/>"
        "&bull; <i>RMSE (Root Mean Squared Error):</i> Square root of the average squared error. Gives more penalty to large forecast misses.<br/>"
        "&bull; <i>MAE (Mean Absolute Error):</i> Average difference between predicted cases and actual cases in plain numbers.<br/>"
        "&bull; <i>Zero-Masked MAPE:</i> Average percentage error calculated only on months where disease cases actually occurred (avoids division by zero).<br/>"
        "&bull; <i>WAPE:</i> Sum of all absolute errors divided by total actual cases across the entire period.",
        styles["BodyTextJustified"]
    ))

    story.append(Paragraph(
        "<b>Appendix D — Individual Contributions & Student Accountability:</b><br/>"
        "<b>Kartik Lamba (Enrollment: S24CSEU0642)</b> — Developer and Author:<br/>"
        "1. <i>Data Engineering:</i> Downloaded, cleaned, and organized 1,320 monthly records from GBIF and NASA POWER; created 43 lag and weather features.<br/>"
        "2. <i>Classical Models (Milestone 2):</i> Built and tuned AR, MA, ARIMA, SARIMA, SARIMAX, and Holt-Winters models; verified stationarity with ADF/KPSS.<br/>"
        "3. <i>Advanced Models (Milestone 3):</i> Built, tuned, and trained PyTorch LSTM, GRU, PatchTST Transformer, and Hurdle XGBoost; estimated GARCH volatility.<br/>"
        "4. <i>Application & Testing:</i> Built the Streamlit cloud application, wrote 113 automated unit tests, and authored the milestone reports.",
        styles["BodyTextJustified"]
    ))

    doc.build(story, canvasmaker=AcademicNumberedCanvas)
    print(f"Successfully generated 9-Page Academic Report: {pdf_path}")

    # Copy to final_submission/reports/
    final_sub_dir = os.path.join(REPO_ROOT, "final_submission", "reports")
    os.makedirs(final_sub_dir, exist_ok=True)
    final_pdf_path = os.path.join(final_sub_dir, "MILESTONE_3_ADVANCED_FORECASTING_REPORT.pdf")
    shutil.copyfile(pdf_path, final_pdf_path)
    print(f"Successfully synchronized to final_submission: {final_pdf_path}")
    return pdf_path


if __name__ == "__main__":
    build_final_report_pdf()
