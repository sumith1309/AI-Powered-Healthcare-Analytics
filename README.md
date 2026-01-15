# 🏥 AI-Powered Healthcare Analytics Platform

<div align="center">

![Healthcare AI](https://img.shields.io/badge/Healthcare-AI%20Analytics-0EA5E9?style=for-the-badge&logo=heart&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ECharts](https://img.shields.io/badge/ECharts-5.4.3-AA344D?style=for-the-badge&logo=apache-echarts&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)

**A comprehensive analytics platform for comparing AI model performance in healthcare settings**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

This project provides a complete analytics solution for evaluating and comparing AI model performance across multiple healthcare departments. It includes:

- **Python Analysis Engine**: Comprehensive data generation, statistical analysis, and visualization
- **Interactive Web Dashboard**: Modern ECharts-powered visualizations with real-time interactivity
- **Executive One-Pager**: Printable/downloadable summary report for stakeholders

### 🎯 Key Objectives

1. Compare **AI_v1** vs **AI_v2** model performance across healthcare KPIs
2. Analyze department-specific improvements (Radiology, Pathology, Cardiology, Operations)
3. Generate 6-month accuracy forecasts using linear regression
4. Provide actionable insights for AI deployment decisions

---

## ✨ Features

### 📊 Data Analysis (Python)
- Multi-department, multi-month synthetic healthcare dataset
- KPI tracking: Accuracy, Turnaround Time, Cost Efficiency, Patient Satisfaction
- Statistical comparison between Baseline, AI_v1, and AI_v2 models
- Linear regression forecasting with R² reliability metrics
- Premium visualizations with Matplotlib and Seaborn

### 🌐 Interactive Dashboard (HTML/JavaScript)
- **Landing Page**: Modern hero section with animated gradient orbs
- **Model Comparison Tab**: 
  - KPI cards with real-time metrics
  - Bar charts comparing AI models
  - Radar chart for multi-dimensional analysis
  - Heatmap showing department × KPI performance
  - Trend line charts over 12 months
- **Forecast Tab**:
  - Individual department forecast charts
  - Comprehensive 18-month visualization (12 historical + 6 forecast)
  - Trend analysis discussion
- **One-Pager Tab**: Embedded executive summary

### 📄 Executive One-Pager
- Clean, print-optimized layout
- Interactive ECharts visualizations
- **Download Options**:
  - 🖨️ Print Report (browser print dialog)
  - 📥 Download PDF (high-quality export)
  - 📝 Download Summary (formatted text file)

---

## 🖼️ Demo

### Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────┐
│  🏥 MedTech AI Analytics                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ Accuracy    │ │ Turnaround  │ │ Cost        │ │ Satisfact.│ │
│  │ +15.5%      │ │ +32.1%      │ │ +23.2%      │ │ +22.8%    │ │
│  │ AI_v2 wins  │ │ AI_v2 wins  │ │ AI_v2 wins  │ │ AI_v2 wins│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
│                                                                 │
│  ┌─────────────────────────────┐ ┌─────────────────────────────┐│
│  │  KPI Comparison Bar Chart   │ │    🏆 Winner: AI_v2         ││
│  │  ████ AI_v1  ████ AI_v2     │ │    +44.4% Better Overall    ││
│  └─────────────────────────────┘ └─────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Clone the Repository

```bash
git clone https://github.com/sumith1309/AI-Powered-Healthcare-Analytics.git
cd AI-Powered-Healthcare-Analytics
```

### Install Python Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy
```

Or create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📖 Usage

### 1. Run Python Analysis

Generate visualizations and analysis reports:

```bash
python ai_model_comparison_analysis.py
```

This will:
- Generate synthetic healthcare data
- Create 7 visualization PNG files
- Print detailed statistical analysis to console
- Show comparison metrics and forecasts

### 2. Open Interactive Dashboard

Simply open the HTML file in your browser:

```bash
# On Windows
start index.html

# On macOS
open index.html

# On Linux
xdg-open index.html
```

### 3. View Executive One-Pager

Open `one-pager.html` directly or access via the dashboard's "One-Pager Summary" tab.

---

## 📁 Project Structure

```
AI-Powered-Healthcare-Analytics/
│
├── 📄 README.md                          # Project documentation
├── 🐍 ai_model_comparison_analysis.py    # Python analysis engine
├── 🌐 index.html                         # Interactive dashboard
├── 📄 one-pager.html                     # Executive summary report
│
└── 📊 Generated Visualizations (after running Python):
    ├── 01_ai_model_comparison_kpis.png
    ├── 02_performance_heatmap.png
    ├── 03_accuracy_trend_timeline.png
    ├── 04_radar_comparison.png
    ├── 05_executive_summary_dashboard.png
    ├── 06_accuracy_forecast_6months.png
    └── 07_comprehensive_forecast.png
```

---

## 📊 Key Metrics & Results

### AI Model Comparison Summary

| Metric | AI_v1 | AI_v2 | Improvement |
|--------|-------|-------|-------------|
| **Accuracy** | +9.8% | +15.5% | 58% better |
| **Turnaround** | +22.0% | +32.1% | 46% better |
| **Cost Savings** | +17.7% | +23.2% | 31% better |
| **Satisfaction** | +15.4% | +22.8% | 48% better |
| **Average** | +16.2% | +23.4% | **44.4% better** |

### 6-Month Forecast (by June 2025)

| Department | Predicted Accuracy | Trend |
|------------|-------------------|-------|
| Radiology | 0.883 | Slight decline (ceiling effect) |
| Pathology | 0.912 | Stable |
| Cardiology | 0.931 | Positive growth |
| Operations | 0.985 | Strongest growth (+0.68%/month) |

---

## 🛠️ Technologies Used

### Backend / Analysis
- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical graphics
- **SciPy** - Linear regression for forecasting

### Frontend / Dashboard
- **HTML5** - Structure
- **CSS3** - Styling with CSS Grid, Flexbox, animations
- **JavaScript (ES6+)** - Interactivity
- **ECharts 5.4.3** - Interactive charts and visualizations
- **Google Fonts** - Plus Jakarta Sans, JetBrains Mono

### Export / Download
- **html2canvas** - Screenshot capture for PDF
- **jsPDF** - PDF generation
- **Browser Print API** - Native printing

---

## 📈 Visualization Gallery

### 1. KPI Comparison Chart
Bar chart comparing AI_v1 vs AI_v2 across all metrics with gradient colors.

### 2. Performance Heatmap
Color-coded matrix showing improvement percentages by department × KPI.

### 3. Accuracy Trend Timeline
12-month line chart tracking accuracy evolution for each department.

### 4. Radar Comparison
Multi-axis radar chart for holistic model comparison.

### 5. Executive Dashboard
Combined visualization summarizing all key findings.

### 6. Accuracy Forecast
6-month prediction with confidence indicators and trend lines.

### 7. Comprehensive Forecast
18-month view combining historical data with projections.

---

## 🔮 Future Enhancements

- [ ] Real-time data integration via REST API
- [ ] Machine learning model for advanced forecasting
- [ ] Multi-language support
- [ ] Dark mode theme toggle
- [ ] Export to Excel/CSV
- [ ] User authentication and role-based access
- [ ] Database integration (PostgreSQL/MongoDB)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sumith**
- GitHub: [@sumith1309](https://github.com/sumith1309)

---

## 🙏 Acknowledgments

- [ECharts](https://echarts.apache.org/) for powerful visualization library
- [Google Fonts](https://fonts.google.com/) for beautiful typography
- Healthcare industry professionals for domain insights

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made by sumith for Healthcare Analytics

</div>
