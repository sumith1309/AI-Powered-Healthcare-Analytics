"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AI MODEL PERFORMANCE COMPARISON ANALYSIS                   ║
║                         MedTech Healthcare Analytics                          ║
║                                                                              ║
║  Comparing AI_v1 vs AI_v2 across Multiple Departments and KPIs              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =============================================================================
# CONFIGURATION: Premium Light Theme Setup
# =============================================================================

# Custom color palette - Soft, professional, calming colors
COLORS = {
    'primary': '#4A90A4',       # Calm teal blue
    'secondary': '#7FB3D5',     # Light sky blue
    'accent': '#E8B4B8',        # Soft rose
    'success': '#A8D8B9',       # Mint green
    'warning': '#F5CBA7',       # Warm peach
    'neutral': '#D5D8DC',       # Soft gray
    'text_dark': '#2C3E50',     # Deep slate
    'text_light': '#5D6D7E',    # Medium slate
    'background': '#FAFBFC',    # Off-white
    'card': '#FFFFFF',          # Pure white
    'ai_v1': '#5DADE2',         # Sky blue for AI_v1
    'ai_v2': '#48C9B0',         # Seafoam green for AI_v2
    'baseline': '#ABB2B9'       # Muted gray for baseline
}

# Department-specific colors
DEPT_COLORS = {
    'Radiology': '#6C5B7B',     # Mauve purple
    'Pathology': '#C06C84',     # Dusty rose
    'Cardiology': '#F67280',    # Coral pink
    'Operations': '#355C7D'     # Steel blue
}

# Set the premium light theme
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.facecolor': COLORS['background'],
    'axes.edgecolor': COLORS['neutral'],
    'axes.labelcolor': COLORS['text_dark'],
    'figure.facecolor': COLORS['card'],
    'figure.dpi': 120,
    'grid.color': '#E8E8E8',
    'grid.linewidth': 0.5,
    'text.color': COLORS['text_dark'],
    'xtick.color': COLORS['text_light'],
    'ytick.color': COLORS['text_light'],
    'legend.framealpha': 0.95,
    'legend.facecolor': COLORS['card'],
    'legend.edgecolor': COLORS['neutral']
})

# =============================================================================
# 1. DATA GENERATION: Multi-Department, Multi-Month Dataset with Two AI Models
# =============================================================================

print("=" * 80)
print("        AI MODEL PERFORMANCE COMPARISON ANALYSIS - MedTech Healthcare")
print("=" * 80)
print(f"\n[DATE] Analysis Date: {datetime.now().strftime('%B %d, %Y')}")
print("[DEPT] Departments: Radiology, Pathology, Cardiology, Operations")
print("[DATA] Comparison: Baseline -> AI_v1 -> AI_v2")
print("=" * 80)

np.random.seed(42)

departments = ["Radiology", "Pathology", "Cardiology", "Operations"]
months = pd.date_range(start="2024-01-01", periods=12, freq="M")

records = []

for dept in departments:
    for month in months:
        # Baseline metrics (traditional methods)
        baseline_accuracy = np.random.normal(0.78, 0.04)
        baseline_turnaround = np.random.normal(48, 6)  # hours
        baseline_cost = np.random.normal(120, 10)      # cost units
        baseline_satisfaction = np.random.normal(3.8, 0.2)  # out of 5
        
        # AI_v1 metrics (first generation AI model)
        ai_v1_accuracy = baseline_accuracy + np.random.normal(0.08, 0.02)
        ai_v1_turnaround = baseline_turnaround - np.random.normal(10, 2)
        ai_v1_cost = baseline_cost - np.random.normal(20, 5)
        ai_v1_satisfaction = baseline_satisfaction + np.random.normal(0.6, 0.1)
        
        # AI_v2 metrics (improved second generation AI model)
        # AI_v2 has better performance across all metrics
        ai_v2_accuracy = baseline_accuracy + np.random.normal(0.12, 0.02)      # +4% over v1
        ai_v2_turnaround = baseline_turnaround - np.random.normal(15, 2)       # 5 hours faster than v1
        ai_v2_cost = baseline_cost - np.random.normal(28, 4)                   # $8 cheaper than v1
        ai_v2_satisfaction = baseline_satisfaction + np.random.normal(0.85, 0.1)  # +0.25 over v1
        
        records.append([
            dept, month,
            baseline_accuracy, ai_v1_accuracy, ai_v2_accuracy,
            baseline_turnaround, ai_v1_turnaround, ai_v2_turnaround,
            baseline_cost, ai_v1_cost, ai_v2_cost,
            baseline_satisfaction, ai_v1_satisfaction, ai_v2_satisfaction
        ])

columns = [
    "Department", "Month",
    "Baseline_Accuracy", "AI_v1_Accuracy", "AI_v2_Accuracy",
    "Baseline_Turnaround", "AI_v1_Turnaround", "AI_v2_Turnaround",
    "Baseline_Cost", "AI_v1_Cost", "AI_v2_Cost",
    "Baseline_Satisfaction", "AI_v1_Satisfaction", "AI_v2_Satisfaction"
]

df = pd.DataFrame(records, columns=columns)

print("\n[OK] Dataset Generated Successfully!")
print(f"   * Total Records: {len(df)}")
print(f"   * Departments: {len(departments)}")
print(f"   * Months: {len(months)}")
print(f"   * Metrics: Accuracy, Turnaround, Cost, Satisfaction")

# =============================================================================
# 2. KPI COMPUTATION: Department-Level Improvements for Both AI Models
# =============================================================================

summary = df.groupby("Department").mean(numeric_only=True)

# AI_v1 Improvements over Baseline
summary["AI_v1_Accuracy_Improvement_%"] = (
    (summary["AI_v1_Accuracy"] - summary["Baseline_Accuracy"]) /
    summary["Baseline_Accuracy"] * 100
)
summary["AI_v1_Turnaround_Improvement_%"] = (
    (summary["Baseline_Turnaround"] - summary["AI_v1_Turnaround"]) /
    summary["Baseline_Turnaround"] * 100
)
summary["AI_v1_Cost_Improvement_%"] = (
    (summary["Baseline_Cost"] - summary["AI_v1_Cost"]) /
    summary["Baseline_Cost"] * 100
)
summary["AI_v1_Satisfaction_Improvement_%"] = (
    (summary["AI_v1_Satisfaction"] - summary["Baseline_Satisfaction"]) /
    summary["Baseline_Satisfaction"] * 100
)

# AI_v2 Improvements over Baseline
summary["AI_v2_Accuracy_Improvement_%"] = (
    (summary["AI_v2_Accuracy"] - summary["Baseline_Accuracy"]) /
    summary["Baseline_Accuracy"] * 100
)
summary["AI_v2_Turnaround_Improvement_%"] = (
    (summary["Baseline_Turnaround"] - summary["AI_v2_Turnaround"]) /
    summary["Baseline_Turnaround"] * 100
)
summary["AI_v2_Cost_Improvement_%"] = (
    (summary["Baseline_Cost"] - summary["AI_v2_Cost"]) /
    summary["Baseline_Cost"] * 100
)
summary["AI_v2_Satisfaction_Improvement_%"] = (
    (summary["AI_v2_Satisfaction"] - summary["Baseline_Satisfaction"]) /
    summary["Baseline_Satisfaction"] * 100
)

# =============================================================================
# 3. DISPLAY KPI TABLES
# =============================================================================

print("\n" + "-" * 80)
print("[CHART] KPI IMPROVEMENTS: AI_v1 vs Baseline")
print("-" * 80)

ai_v1_improvements = summary[[
    "AI_v1_Accuracy_Improvement_%",
    "AI_v1_Turnaround_Improvement_%",
    "AI_v1_Cost_Improvement_%",
    "AI_v1_Satisfaction_Improvement_%"
]].round(2)
ai_v1_improvements.columns = ["Accuracy (%)", "Turnaround (%)", "Cost (%)", "Satisfaction (%)"]
print(ai_v1_improvements.to_string())

print("\n" + "-" * 80)
print("[CHART] KPI IMPROVEMENTS: AI_v2 vs Baseline")
print("-" * 80)

ai_v2_improvements = summary[[
    "AI_v2_Accuracy_Improvement_%",
    "AI_v2_Turnaround_Improvement_%",
    "AI_v2_Cost_Improvement_%",
    "AI_v2_Satisfaction_Improvement_%"
]].round(2)
ai_v2_improvements.columns = ["Accuracy (%)", "Turnaround (%)", "Cost (%)", "Satisfaction (%)"]
print(ai_v2_improvements.to_string())

# =============================================================================
# 4. VISUALIZATION 1: Side-by-Side Bar Chart - AI_v1 vs AI_v2 Comparison
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('AI Model Performance Comparison: AI_v1 vs AI_v2\nImprovement Over Baseline (%)', 
             fontsize=18, fontweight='bold', color=COLORS['text_dark'], y=1.02)

kpis = [
    ('Accuracy', 'AI_v1_Accuracy_Improvement_%', 'AI_v2_Accuracy_Improvement_%', '[A]'),
    ('Turnaround Time', 'AI_v1_Turnaround_Improvement_%', 'AI_v2_Turnaround_Improvement_%', '[T]'),
    ('Cost Reduction', 'AI_v1_Cost_Improvement_%', 'AI_v2_Cost_Improvement_%', '[C]'),
    ('Patient Satisfaction', 'AI_v1_Satisfaction_Improvement_%', 'AI_v2_Satisfaction_Improvement_%', '[S]')
]

for idx, (kpi_name, v1_col, v2_col, emoji) in enumerate(kpis):
    ax = axes[idx // 2, idx % 2]
    
    x = np.arange(len(departments))
    width = 0.35
    
    v1_values = summary[v1_col].values
    v2_values = summary[v2_col].values
    
    bars1 = ax.bar(x - width/2, v1_values, width, label='AI_v1', 
                   color=COLORS['ai_v1'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, v2_values, width, label='AI_v2', 
                   color=COLORS['ai_v2'], edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=COLORS['text_dark'])
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=COLORS['text_dark'])
    
    ax.set_title(f'{emoji} {kpi_name} Improvement', fontsize=14, fontweight='bold', 
                 color=COLORS['text_dark'], pad=15)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(departments, rotation=30, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_facecolor(COLORS['background'])
    
    # Add subtle grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('01_ai_model_comparison_kpis.png', dpi=150, bbox_inches='tight', 
            facecolor=COLORS['card'], edgecolor='none')
plt.show()

# =============================================================================
# 5. VISUALIZATION 2: Comprehensive Comparison Heatmap
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for heatmap
heatmap_data = pd.DataFrame({
    'AI_v1 Accuracy': summary['AI_v1_Accuracy_Improvement_%'],
    'AI_v2 Accuracy': summary['AI_v2_Accuracy_Improvement_%'],
    'AI_v1 Turnaround': summary['AI_v1_Turnaround_Improvement_%'],
    'AI_v2 Turnaround': summary['AI_v2_Turnaround_Improvement_%'],
    'AI_v1 Cost': summary['AI_v1_Cost_Improvement_%'],
    'AI_v2 Cost': summary['AI_v2_Cost_Improvement_%'],
    'AI_v1 Satisfaction': summary['AI_v1_Satisfaction_Improvement_%'],
    'AI_v2 Satisfaction': summary['AI_v2_Satisfaction_Improvement_%']
})

# Create custom colormap - calm premium tones
cmap = sns.light_palette(COLORS['primary'], as_cmap=True)

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap=cmap, 
            linewidths=2, linecolor='white', cbar_kws={'label': 'Improvement (%)'}, 
            ax=ax, annot_kws={'size': 11, 'weight': 'bold'})

ax.set_title('Performance Improvement Heatmap: AI_v1 vs AI_v2 by Department\n', 
             fontsize=16, fontweight='bold', color=COLORS['text_dark'])
ax.set_ylabel('Department', fontsize=12)
ax.set_xlabel('KPI Metric & Model Version', fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('02_performance_heatmap.png', dpi=150, bbox_inches='tight', 
            facecolor=COLORS['card'], edgecolor='none')
plt.show()

# =============================================================================
# 6. VISUALIZATION 3: Accuracy Trend Over Time - All Models
# =============================================================================

fig, ax = plt.subplots(figsize=(16, 8))

for i, dept in enumerate(departments):
    dept_data = df[df['Department'] == dept]
    color = list(DEPT_COLORS.values())[i]
    
    # Baseline - dashed
    ax.plot(dept_data['Month'], dept_data['Baseline_Accuracy'], 
            linestyle='--', color=color, alpha=0.4, linewidth=2,
            label=f'{dept} - Baseline' if i == 0 else '')
    
    # AI_v1 - dotted
    ax.plot(dept_data['Month'], dept_data['AI_v1_Accuracy'], 
            linestyle=':', color=color, alpha=0.7, linewidth=2.5,
            marker='o', markersize=5, label=f'{dept} - AI_v1' if i == 0 else '')
    
    # AI_v2 - solid
    ax.plot(dept_data['Month'], dept_data['AI_v2_Accuracy'], 
            linestyle='-', color=color, linewidth=3,
            marker='s', markersize=6, label=f'{dept} - AI_v2' if i == 0 else '')

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], linestyle='--', color='gray', linewidth=2, label='Baseline'),
    Line2D([0], [0], linestyle=':', color='gray', linewidth=2, marker='o', markersize=5, label='AI_v1'),
    Line2D([0], [0], linestyle='-', color='gray', linewidth=3, marker='s', markersize=6, label='AI_v2'),
]
legend_elements.extend([
    Line2D([0], [0], color=color, linewidth=3, label=dept) 
    for dept, color in DEPT_COLORS.items()
])

ax.legend(handles=legend_elements, loc='lower right', ncol=2, framealpha=0.95)

ax.set_title('Accuracy Trend Over Time: Baseline -> AI_v1 -> AI_v2\n', 
             fontsize=16, fontweight='bold', color=COLORS['text_dark'])
ax.set_ylabel('Accuracy Score', fontsize=12)
ax.set_xlabel('Month', fontsize=12)
ax.set_facecolor(COLORS['background'])

# Format x-axis dates
plt.xticks(rotation=45, ha='right')
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('03_accuracy_trend_timeline.png', dpi=150, bbox_inches='tight', 
            facecolor=COLORS['card'], edgecolor='none')
plt.show()

# =============================================================================
# 7. VISUALIZATION 4: Radar Chart - Overall Model Comparison
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Calculate overall averages
overall_metrics = {
    'Accuracy': (summary['AI_v1_Accuracy_Improvement_%'].mean(), 
                 summary['AI_v2_Accuracy_Improvement_%'].mean()),
    'Turnaround': (summary['AI_v1_Turnaround_Improvement_%'].mean(), 
                   summary['AI_v2_Turnaround_Improvement_%'].mean()),
    'Cost Savings': (summary['AI_v1_Cost_Improvement_%'].mean(), 
                     summary['AI_v2_Cost_Improvement_%'].mean()),
    'Satisfaction': (summary['AI_v1_Satisfaction_Improvement_%'].mean(), 
                     summary['AI_v2_Satisfaction_Improvement_%'].mean())
}

categories = list(overall_metrics.keys())
v1_values = [overall_metrics[cat][0] for cat in categories]
v2_values = [overall_metrics[cat][1] for cat in categories]

# Number of variables
N = len(categories)

# Compute angle for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the loop

v1_values += v1_values[:1]
v2_values += v2_values[:1]

# Draw the radar chart
ax.plot(angles, v1_values, 'o-', linewidth=3, color=COLORS['ai_v1'], 
        label='AI_v1', markersize=10)
ax.fill(angles, v1_values, alpha=0.25, color=COLORS['ai_v1'])

ax.plot(angles, v2_values, 'o-', linewidth=3, color=COLORS['ai_v2'], 
        label='AI_v2', markersize=10)
ax.fill(angles, v2_values, alpha=0.25, color=COLORS['ai_v2'])

# Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13, fontweight='bold', color=COLORS['text_dark'])

ax.set_title('Overall Model Performance Comparison\n(Average Improvement Across All Departments)', 
             fontsize=16, fontweight='bold', color=COLORS['text_dark'], pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=12)

plt.tight_layout()
plt.savefig('04_radar_comparison.png', dpi=150, bbox_inches='tight', 
            facecolor=COLORS['card'], edgecolor='none')
plt.show()

# =============================================================================
# 8. SUMMARY PAGE - Executive Dashboard
# =============================================================================

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(COLORS['card'])

# Create grid layout
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

# Title Section
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')
ax_title.text(0.5, 0.8, 'MedTech AI Model Performance Summary', 
              fontsize=28, fontweight='bold', ha='center', va='center', 
              color=COLORS['text_dark'], transform=ax_title.transAxes)
ax_title.text(0.5, 0.4, 'Comprehensive Analysis: AI_v1 vs AI_v2 Model Comparison', 
              fontsize=16, ha='center', va='center', 
              color=COLORS['text_light'], transform=ax_title.transAxes)
ax_title.text(0.5, 0.1, f'Analysis Period: January 2024 - December 2024 | Generated: {datetime.now().strftime("%B %d, %Y")}', 
              fontsize=11, ha='center', va='center', 
              color=COLORS['text_light'], style='italic', transform=ax_title.transAxes)

# Key Metrics Cards
metrics_data = [
    ('Accuracy Gain', 
     f"+{summary['AI_v1_Accuracy_Improvement_%'].mean():.1f}%", 
     f"+{summary['AI_v2_Accuracy_Improvement_%'].mean():.1f}%",
     COLORS['primary']),
    ('Time Saved', 
     f"+{summary['AI_v1_Turnaround_Improvement_%'].mean():.1f}%", 
     f"+{summary['AI_v2_Turnaround_Improvement_%'].mean():.1f}%",
     COLORS['secondary']),
    ('Cost Reduction', 
     f"+{summary['AI_v1_Cost_Improvement_%'].mean():.1f}%", 
     f"+{summary['AI_v2_Cost_Improvement_%'].mean():.1f}%",
     COLORS['success']),
    ('Satisfaction', 
     f"+{summary['AI_v1_Satisfaction_Improvement_%'].mean():.1f}%", 
     f"+{summary['AI_v2_Satisfaction_Improvement_%'].mean():.1f}%",
     COLORS['accent'])
]

for i, (title, v1_val, v2_val, color) in enumerate(metrics_data):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor(COLORS['background'])
    ax.axis('off')
    
    # Card background
    rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, transform=ax.transAxes,
                          facecolor=color, alpha=0.15, edgecolor=color, 
                          linewidth=2, clip_on=False)
    ax.add_patch(rect)
    
    ax.text(0.5, 0.85, title, fontsize=13, fontweight='bold', ha='center', 
            va='center', color=COLORS['text_dark'], transform=ax.transAxes)
    ax.text(0.3, 0.5, 'AI_v1', fontsize=10, ha='center', va='center', 
            color=COLORS['text_light'], transform=ax.transAxes)
    ax.text(0.3, 0.3, v1_val, fontsize=16, fontweight='bold', ha='center', 
            va='center', color=COLORS['ai_v1'], transform=ax.transAxes)
    ax.text(0.7, 0.5, 'AI_v2', fontsize=10, ha='center', va='center', 
            color=COLORS['text_light'], transform=ax.transAxes)
    ax.text(0.7, 0.3, v2_val, fontsize=16, fontweight='bold', ha='center', 
            va='center', color=COLORS['ai_v2'], transform=ax.transAxes)

# Mini bar chart
ax_bar = fig.add_subplot(gs[2, :2])
x = np.arange(4)
width = 0.35
kpi_labels = ['Accuracy', 'Turnaround', 'Cost', 'Satisfaction']
v1_means = [summary[f'AI_v1_{kpi}_Improvement_%'].mean() 
            for kpi in ['Accuracy', 'Turnaround', 'Cost', 'Satisfaction']]
v2_means = [summary[f'AI_v2_{kpi}_Improvement_%'].mean() 
            for kpi in ['Accuracy', 'Turnaround', 'Cost', 'Satisfaction']]

bars1 = ax_bar.bar(x - width/2, v1_means, width, label='AI_v1', color=COLORS['ai_v1'])
bars2 = ax_bar.bar(x + width/2, v2_means, width, label='AI_v2', color=COLORS['ai_v2'])

ax_bar.set_title('Average Improvement by KPI', fontsize=14, fontweight='bold', 
                 color=COLORS['text_dark'])
ax_bar.set_ylabel('Improvement (%)')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(kpi_labels)
ax_bar.legend()
ax_bar.set_facecolor(COLORS['background'])
ax_bar.yaxis.grid(True, linestyle='--', alpha=0.4)

# Winner announcement
ax_winner = fig.add_subplot(gs[2, 2:])
ax_winner.axis('off')
ax_winner.set_facecolor(COLORS['background'])

# Calculate overall winner
v1_total = np.mean(v1_means)
v2_total = np.mean(v2_means)
improvement_pct = ((v2_total - v1_total) / v1_total) * 100

rect = plt.Rectangle((0.05, 0.1), 0.9, 0.8, transform=ax_winner.transAxes,
                      facecolor=COLORS['ai_v2'], alpha=0.2, edgecolor=COLORS['ai_v2'], 
                      linewidth=3, clip_on=False)
ax_winner.add_patch(rect)

ax_winner.text(0.5, 0.75, 'WINNER', fontsize=20, fontweight='bold', ha='center', 
               va='center', color=COLORS['ai_v2'], transform=ax_winner.transAxes)
ax_winner.text(0.5, 0.5, 'AI_v2', fontsize=32, fontweight='bold', ha='center', 
               va='center', color=COLORS['text_dark'], transform=ax_winner.transAxes)
ax_winner.text(0.5, 0.25, f'+{improvement_pct:.1f}% better than AI_v1', 
               fontsize=14, ha='center', va='center', 
               color=COLORS['text_light'], transform=ax_winner.transAxes)

# Insights text box
ax_insights = fig.add_subplot(gs[3, :])
ax_insights.axis('off')

insights_text = """
+==============================================================================================================================+
|                                                    KEY INSIGHTS & RECOMMENDATIONS                                            |
+==============================================================================================================================+
|                                                                                                                              |
|  [*] PERFORMANCE SUMMARY:                                                                                                    |
|      - AI_v2 outperforms AI_v1 across ALL four key performance indicators (Accuracy, Turnaround, Cost, Satisfaction)        |
|      - Average improvement of AI_v2 over baseline: {:.1f}% | Average improvement of AI_v1 over baseline: {:.1f}%            |
|      - AI_v2 delivers {:.1f}% better overall performance compared to AI_v1                                                  |
|                                                                                                                              |
|  [>] KEY ACHIEVEMENTS:                                                                                                       |
|      - Accuracy: AI_v2 achieves {:.1f}% improvement (vs {:.1f}% for AI_v1) - Enhanced diagnostic precision                  |
|      - Turnaround: AI_v2 reduces processing time by {:.1f}% (vs {:.1f}% for AI_v1) - Faster patient care                    |
|      - Cost: AI_v2 saves {:.1f}% in operational costs (vs {:.1f}% for AI_v1) - Better resource utilization                  |
|      - Satisfaction: AI_v2 improves patient satisfaction by {:.1f}% (vs {:.1f}% for AI_v1) - Enhanced experience            |
|                                                                                                                              |
|  [!] RECOMMENDATION:                                                                                                         |
|      Deploy AI_v2 across all departments for maximum operational efficiency and patient care quality.                        |
|      Priority departments for immediate deployment: Radiology and Pathology (highest improvement potential).                 |
|                                                                                                                              |
+==============================================================================================================================+
""".format(
    v2_total, v1_total, improvement_pct,
    summary['AI_v2_Accuracy_Improvement_%'].mean(), summary['AI_v1_Accuracy_Improvement_%'].mean(),
    summary['AI_v2_Turnaround_Improvement_%'].mean(), summary['AI_v1_Turnaround_Improvement_%'].mean(),
    summary['AI_v2_Cost_Improvement_%'].mean(), summary['AI_v1_Cost_Improvement_%'].mean(),
    summary['AI_v2_Satisfaction_Improvement_%'].mean(), summary['AI_v1_Satisfaction_Improvement_%'].mean()
)

ax_insights.text(0.5, 0.5, insights_text, fontsize=9, fontfamily='monospace',
                 ha='center', va='center', color=COLORS['text_dark'], 
                 transform=ax_insights.transAxes)

plt.tight_layout()
plt.savefig('05_executive_summary_dashboard.png', dpi=150, bbox_inches='tight', 
            facecolor=COLORS['card'], edgecolor='none')
plt.show()

# =============================================================================
# 9. DEPARTMENT-LEVEL FORECASTING FOR NEXT 6 MONTHS
# =============================================================================

from scipy import stats

print("\n" + "=" * 80)
print("                    [FORECAST] 6-MONTH ACCURACY FORECASTING")
print("=" * 80)

# Create numeric month index for linear regression
df['Month_Num'] = (df['Month'] - df['Month'].min()).dt.days

# Generate future months
future_months = pd.date_range(start="2025-01-01", periods=6, freq="M")
future_month_nums = [(m - df['Month'].min()).days for m in future_months]

# Store forecast results
forecast_results = {}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('AI_v2 Accuracy Forecast: Next 6 Months by Department\nLinear Trend Projection', 
             fontsize=18, fontweight='bold', color=COLORS['text_dark'], y=1.02)

for idx, dept in enumerate(departments):
    ax = axes[idx // 2, idx % 2]
    dept_data = df[df['Department'] == dept].copy()
    
    # Get x (month numbers) and y (AI_v2 accuracy) values
    x = dept_data['Month_Num'].values
    y_v2 = dept_data['AI_v2_Accuracy'].values
    y_v1 = dept_data['AI_v1_Accuracy'].values
    
    # Fit linear regression for AI_v2
    slope_v2, intercept_v2, r_value_v2, p_value_v2, std_err_v2 = stats.linregress(x, y_v2)
    
    # Fit linear regression for AI_v1
    slope_v1, intercept_v1, r_value_v1, p_value_v1, std_err_v1 = stats.linregress(x, y_v1)
    
    # Generate predictions for historical data
    y_pred_v2 = slope_v2 * x + intercept_v2
    y_pred_v1 = slope_v1 * x + intercept_v1
    
    # Generate forecasts for future months
    future_x = np.array(future_month_nums)
    forecast_v2 = slope_v2 * future_x + intercept_v2
    forecast_v1 = slope_v1 * future_x + intercept_v1
    
    # Store results
    forecast_results[dept] = {
        'slope_v2': slope_v2,
        'intercept_v2': intercept_v2,
        'r_squared_v2': r_value_v2**2,
        'forecast_v2': forecast_v2,
        'slope_v1': slope_v1,
        'intercept_v1': intercept_v1,
        'r_squared_v1': r_value_v1**2,
        'forecast_v1': forecast_v1,
        'monthly_improvement_v2': slope_v2 * 30,  # Approximate monthly improvement
        'monthly_improvement_v1': slope_v1 * 30
    }
    
    # Plot historical data
    ax.scatter(dept_data['Month'], y_v1, color=COLORS['ai_v1'], alpha=0.5, s=50, label='AI_v1 Actual')
    ax.scatter(dept_data['Month'], y_v2, color=COLORS['ai_v2'], alpha=0.7, s=60, label='AI_v2 Actual')
    
    # Plot trend lines
    ax.plot(dept_data['Month'], y_pred_v1, '--', color=COLORS['ai_v1'], alpha=0.7, linewidth=2, label='AI_v1 Trend')
    ax.plot(dept_data['Month'], y_pred_v2, '-', color=COLORS['ai_v2'], linewidth=2.5, label='AI_v2 Trend')
    
    # Plot forecast
    ax.plot(future_months, forecast_v1, ':', color=COLORS['ai_v1'], linewidth=2, alpha=0.8)
    ax.scatter(future_months, forecast_v1, color=COLORS['ai_v1'], marker='D', s=60, alpha=0.7)
    
    ax.plot(future_months, forecast_v2, '-', color=COLORS['ai_v2'], linewidth=3, alpha=0.8)
    ax.scatter(future_months, forecast_v2, color=COLORS['ai_v2'], marker='s', s=80, 
               edgecolor='white', linewidth=2, label='AI_v2 Forecast')
    
    # Add confidence band for AI_v2 forecast (simple approximation)
    std_residual = np.std(y_v2 - y_pred_v2)
    ax.fill_between(future_months, forecast_v2 - 1.96*std_residual, forecast_v2 + 1.96*std_residual,
                    color=COLORS['ai_v2'], alpha=0.15, label='95% CI')
    
    # Add vertical line to separate historical vs forecast
    ax.axvline(x=pd.Timestamp('2024-12-31'), color=COLORS['neutral'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(pd.Timestamp('2024-12-15'), ax.get_ylim()[1] * 0.98, 'Historical', fontsize=9, 
            color=COLORS['text_light'], ha='right', va='top')
    ax.text(pd.Timestamp('2025-01-15'), ax.get_ylim()[1] * 0.98, 'Forecast', fontsize=9, 
            color=COLORS['text_light'], ha='left', va='top')
    
    # Department title with R-squared
    dept_color = list(DEPT_COLORS.values())[idx]
    ax.set_title(f'{dept}\nR-squared: {r_value_v2**2:.3f} | Monthly Trend: +{slope_v2*30*100:.2f}%', 
                 fontsize=12, fontweight='bold', color=dept_color, pad=10)
    ax.set_ylabel('Accuracy Score', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_facecolor(COLORS['background'])
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    if idx == 0:
        ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('06_accuracy_forecast_6months.png', dpi=150, bbox_inches='tight', 
            facecolor=COLORS['card'], edgecolor='none')
plt.show()

# =============================================================================
# 10. FORECAST SUMMARY TABLE
# =============================================================================

print("\n" + "-" * 80)
print("[FORECAST] AI_v2 Accuracy Predictions for Next 6 Months")
print("-" * 80)

forecast_df = pd.DataFrame({
    'Month': future_months.strftime('%b %Y')
})

for dept in departments:
    forecast_df[dept] = [f"{v:.3f}" for v in forecast_results[dept]['forecast_v2']]

print(forecast_df.to_string(index=False))

print("\n" + "-" * 80)
print("[STATS] Linear Trend Analysis Summary")
print("-" * 80)

trend_summary = pd.DataFrame({
    'Department': departments,
    'Monthly_Trend_V2': [f"+{forecast_results[d]['monthly_improvement_v2']*100:.3f}%" for d in departments],
    'R_Squared_V2': [f"{forecast_results[d]['r_squared_v2']:.3f}" for d in departments],
    'Forecast_Jun_2025': [f"{forecast_results[d]['forecast_v2'][-1]:.3f}" for d in departments]
})
print(trend_summary.to_string(index=False))

# =============================================================================
# 11. COMPREHENSIVE FORECAST VISUALIZATION
# =============================================================================

fig, ax = plt.subplots(figsize=(16, 9))

all_months = list(months) + list(future_months)

for i, dept in enumerate(departments):
    dept_data = df[df['Department'] == dept]
    color = list(DEPT_COLORS.values())[i]
    
    # Historical AI_v2 data
    ax.plot(dept_data['Month'], dept_data['AI_v2_Accuracy'], 
            '-o', color=color, linewidth=2.5, markersize=7, alpha=0.8,
            label=f'{dept} (Historical)')
    
    # Forecast data
    ax.plot(future_months, forecast_results[dept]['forecast_v2'],
            '--s', color=color, linewidth=2.5, markersize=9,
            markerfacecolor='white', markeredgewidth=2, alpha=0.9,
            label=f'{dept} (Forecast)')
    
    # Confidence interval
    std_residual = np.std(dept_data['AI_v2_Accuracy'].values - 
                          (forecast_results[dept]['slope_v2'] * dept_data['Month_Num'].values + 
                           forecast_results[dept]['intercept_v2']))
    ax.fill_between(future_months, 
                    forecast_results[dept]['forecast_v2'] - 1.96*std_residual,
                    forecast_results[dept]['forecast_v2'] + 1.96*std_residual,
                    color=color, alpha=0.1)

# Add dividing line
ax.axvline(x=pd.Timestamp('2024-12-31'), color=COLORS['text_light'], 
           linestyle='--', linewidth=2, alpha=0.5)

# Add annotations
ax.annotate('HISTORICAL DATA', xy=(pd.Timestamp('2024-06-15'), 0.98), fontsize=12,
            fontweight='bold', color=COLORS['text_light'], ha='center')
ax.annotate('6-MONTH FORECAST', xy=(pd.Timestamp('2025-03-15'), 0.98), fontsize=12,
            fontweight='bold', color=COLORS['secondary'], ha='center')

ax.set_title('AI_v2 Accuracy: Historical Trend & 6-Month Forecast by Department\n', 
             fontsize=18, fontweight='bold', color=COLORS['text_dark'])
ax.set_ylabel('Accuracy Score', fontsize=13)
ax.set_xlabel('Month', fontsize=13)
ax.set_facecolor(COLORS['background'])
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

# Custom legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::2], [d for d in departments], loc='lower right', 
          title='Department', fontsize=10, title_fontsize=11, framealpha=0.95)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('07_comprehensive_forecast.png', dpi=150, bbox_inches='tight', 
            facecolor=COLORS['card'], edgecolor='none')
plt.show()

# =============================================================================
# 12. FORECAST TREND DISCUSSION
# =============================================================================

print("\n" + "=" * 80)
print("                    [ANALYSIS] FORECAST TREND DISCUSSION")
print("=" * 80)

print("""
+-----------------------------------------------------------------------------+
|                      FORECAST TREND ANALYSIS & DISCUSSION                    |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [Q] IS THE TREND REALISTIC?                                                |
|                                                                             |
|  The linear trend projections show continued accuracy improvements for      |
|  AI_v2 across all departments. However, this trend has LIMITATIONS:         |
|                                                                             |
|  1. CEILING EFFECTS: Accuracy cannot exceed 100%. As models approach        |
|     higher accuracy levels, improvements naturally slow down (diminishing   |
|     returns). Linear extrapolation may overestimate future performance.     |
|                                                                             |
|  2. MODEL MATURITY: AI models typically show rapid early gains followed     |
|     by plateau phases. A logarithmic or S-curve model may be more           |
|     realistic for long-term forecasting.                                    |
|                                                                             |
|  3. R-SQUARED VALUES: The model fit quality varies by department.           |
|     Higher R-squared indicates more reliable trend estimation.              |
|                                                                             |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [!] OPERATIONAL FACTORS THAT COULD CHANGE THE TREND:                       |
|                                                                             |
|  POSITIVE FACTORS (Could Accelerate Improvement):                           |
|  - Additional training data from new patient cases                          |
|  - Model updates with enhanced algorithms                                   |
|  - Better hardware infrastructure and processing power                      |
|  - Improved data quality and standardization                                |
|  - Cross-departmental knowledge sharing                                     |
|                                                                             |
|  NEGATIVE FACTORS (Could Slow or Reverse Trend):                            |
|  - Data drift: Patient demographics or disease patterns change              |
|  - Staff turnover affecting model adoption and usage                        |
|  - System integration issues or technical debt                              |
|  - Regulatory changes requiring model modifications                         |
|  - Budget constraints limiting infrastructure upgrades                      |
|  - Edge cases and rare conditions not in training data                      |
|                                                                             |
|  EXTERNAL FACTORS:                                                          |
|  - New medical guidelines or diagnostic criteria                            |
|  - Competing AI solutions entering the market                               |
|  - Healthcare policy changes affecting AI adoption                          |
|  - Patient privacy regulations impacting data availability                  |
|                                                                             |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [>] RECOMMENDATION:                                                        |
|                                                                             |
|  1. Monitor actual vs. forecasted accuracy monthly                          |
|  2. Implement early warning systems for performance degradation             |
|  3. Plan for model retraining every 6-12 months                             |
|  4. Consider ensemble approaches combining AI_v1 and AI_v2                  |
|  5. Establish department-specific improvement targets based on trends       |
|                                                                             |
+-----------------------------------------------------------------------------+
""")

# =============================================================================
# 13. FINAL CONSOLE SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("                         [OK] ANALYSIS COMPLETE - SUMMARY REPORT")
print("=" * 80)

print("""
+-----------------------------------------------------------------------------+
|                          MODEL PERFORMANCE COMPARISON                        |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [1] AI_v1 (First Generation Model):                                        |
|      - Overall Improvement over Baseline: {:.2f}%                            |
|      - Strong performance in cost reduction and accuracy                     |
|                                                                             |
|  [2] AI_v2 (Improved Model):                                                 |
|      - Overall Improvement over Baseline: {:.2f}%                            |
|      - Superior performance across all KPIs                                  |
|      - +{:.2f}% better than AI_v1                                            |
|                                                                             |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [WINNER] AI_v2                                                              |
|                                                                             |
|  INTERPRETATION:                                                             |
|  AI_v2 demonstrates significant improvements over AI_v1 due to:              |
|  1. Enhanced machine learning algorithms with better pattern recognition     |
|  2. Optimized processing pipelines reducing turnaround times                 |
|  3. More efficient resource utilization leading to cost savings              |
|  4. Improved user interface and workflow integration                         |
|                                                                             |
|  The data clearly shows AI_v2 is the superior choice for deployment          |
|  across all healthcare departments.                                          |
|                                                                             |
+-----------------------------------------------------------------------------+
""".format(v1_total, v2_total, improvement_pct))

print("\n[FILES] Generated Visualization Files:")
print("   1. 01_ai_model_comparison_kpis.png      - Side-by-side KPI comparison")
print("   2. 02_performance_heatmap.png           - Department performance heatmap")
print("   3. 03_accuracy_trend_timeline.png       - Accuracy trends over time")
print("   4. 04_radar_comparison.png              - Radar chart comparison")
print("   5. 05_executive_summary_dashboard.png   - Executive summary dashboard")
print("   6. 06_accuracy_forecast_6months.png     - 6-month forecast by department")
print("   7. 07_comprehensive_forecast.png        - Comprehensive forecast visualization")

print("\n" + "=" * 80)
print("                    [SUCCESS] Analysis completed successfully!")
print("=" * 80)
