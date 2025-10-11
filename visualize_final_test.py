import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Phase 1
ax1 = fig.add_subplot(gs[0, 0])
models = ['KcBERT\n(beomi/kcbert-base)', 'ELECTRA\n(monologg/koelectra-small-d)', 'KoBERT\n(skt/kobert-base-v1)', 'RoBERTa\n(klue/roberta-base)']
scores_p1 = [0.9101, 0.8950, 0.8850, 0.8780]
colors = ['#80CBC4', '#FFAB91', '#80CBC4', '#EF9A9A']
y_pos = np.arange(len(models))
bars = ax1.barh(y_pos, scores_p1, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(models, fontsize=7)
ax1.set_xlabel('F1-Score', fontsize=10, fontweight='bold')
ax1.set_xlim(0.87, 0.92)
ax1.set_title('Phase 1: Initial Model Selection (4 Models)', fontsize=12, fontweight='bold', pad=10)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
for i, score in enumerate(scores_p1):
    ax1.text(score + 0.001, i, f'{score:.4f}', va='center', fontsize=10, fontweight='bold')
# Phase progression with TEST score
ax2 = fig.add_subplot(gs[0, 1:])
phases = ['Phase 1\n4 Models', 'Phase 2\nAEDA', 'Phase 3\nTuning', 'Phase 4\nTAPT', 'Phase 5\nEnsemble (Dev)', 'Test Score']
f1_scores = [0.9101, 0.9267, 0.9315, 0.9329, 0.9383, 0.9429]
ax2.plot(phases, f1_scores, marker='o', linewidth=3, markersize=12, color='#2E86AB')
ax2.fill_between(range(len(phases)), f1_scores, alpha=0.3, color='#A3C9D9')
ax2.set_ylabel('Best F1-Score', fontsize=11, fontweight='bold')
ax2.set_ylim(0.88, 0.95)
ax2.set_title('Performance Improvement Across Phases', fontsize=13, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--')
for i, (phase, score) in enumerate(zip(phases, f1_scores)):
    ax2.annotate(f'{score:.4f}', xy=(i, score), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
# Model comparison with HuggingFace IDs
ax3 = fig.add_subplot(gs[1, 0])
model_names = ['KcBERT\n(beomi/kcbert-base)', 'ELECTRA\n(monologg/koelectra-small-d)']
phases_comp = ['P1', 'P2', 'P3', 'P4']
kcbert_scores = [0.9101, 0.9267, 0.9185, 0.9329]
electra_scores = [0.8950, 0.9267, 0.9315, 0.9180]
x = np.arange(len(phases_comp))
width = 0.35
bars1 = ax3.bar(x - width/2, kcbert_scores, width, label=model_names[0], color='#EF9A9A', edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(x + width/2, electra_scores, width, label=model_names[1], color='#FFAB91', edgecolor='black', linewidth=1.5)
ax3.set_ylabel('F1-Score', fontsize=10, fontweight='bold')
ax3.set_xlabel('Phase', fontsize=10, fontweight='bold')
ax3.set_title('KcBERT vs ELECTRA Progress', fontsize=12, fontweight='bold', pad=10)
ax3.set_xticks(x)
ax3.set_xticklabels(phases_comp)
ax3.legend(fontsize=8, loc='lower right')
ax3.set_ylim(0.88, 0.94)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
# Incremental improvement
ax4 = fig.add_subplot(gs[1, 1])
methods = ['AEDA', 'Tuning', 'TAPT', 'Ensemble', 'Test']
improvements = [1.66, 0.48, 0.14, 0.54, 0.46]
colors_imp = ['#FFAB91', '#CE93D8', '#80CBC4', '#A5D6A7', '#FFD54F']
bars = ax4.bar(methods, improvements, color=colors_imp, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Improvement (%p)', fontsize=10, fontweight='bold')
ax4.set_title('Incremental Improvement by Method', fontsize=12, fontweight='bold', pad=10)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim(0, 1.8)
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height, f'+{imp:.2f}%p', ha='center', va='bottom', fontsize=10, fontweight='bold')
# Cumulative gain
ax5 = fig.add_subplot(gs[1, 2])
stages = ['Baseline', 'AEDA', 'Tuning', 'TAPT', 'Ensemble', 'Test']
cumulative = [0, 1.66, 2.14, 2.28, 2.82, 3.28]
ax5.plot(stages, cumulative, marker='o', linewidth=3, markersize=10, color='#E57373')
ax5.fill_between(range(len(stages)), cumulative, alpha=0.3, color='#FFCDD2')
ax5.set_ylabel('Cumulative Improvement (%p)', fontsize=10, fontweight='bold')
ax5.set_title('Cumulative Performance Gain', fontsize=12, fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.set_ylim(0, 3.5)
for i, (stage, cum) in enumerate(zip(stages, cumulative)):
    ax5.annotate(f'+{cum:.2f}%p', xy=(i, cum), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
# Complete journey with TEST
ax6 = fig.add_subplot(gs[2, :])
journey_phases = ['Baseline\nKcBERT', 'AEDA\nAugmentation', 'Hyperparameter\nTuning', 'TAPT\nDomain Adapt', 'Ensemble\nDev', 'Final Test\nScore']
journey_scores = [0.9101, 0.9267, 0.9315, 0.9329, 0.9383, 0.9429]
journey_improvements = [0, 1.66, 0.48, 0.14, 0.54, 0.46]
colors_journey = ['#EF9A9A', '#80CBC4', '#80CBC4', '#FFAB91', '#80CBC4', '#FFD54F']
x_pos = np.arange(len(journey_phases))
bars = ax6.bar(x_pos, journey_scores, color=colors_journey, edgecolor='black', linewidth=2)
ax6.axhline(y=0.9101, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
ax6.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax6.set_xlabel('Complete Experimental Journey', fontsize=12, fontweight='bold')
ax6.set_title('Complete Experimental Journey: From Baseline to Final Test', fontsize=14, fontweight='bold', pad=15)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(journey_phases, fontsize=10)
ax6.set_ylim(0.88, 0.95)
ax6.grid(axis='y', alpha=0.3, linestyle='--')
ax6.legend(fontsize=10)
for i, (bar, score, imp) in enumerate(zip(bars, journey_scores, journey_improvements)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.002, f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    if imp > 0:
        ax6.text(bar.get_x() + bar.get_width()/2., height - 0.015, f'+{imp:.2f}%p', ha='center', va='top', fontsize=9, fontweight='bold', color='darkgreen')
plt.suptitle('Korean Hate Speech Detection: Experimental Results Summary (Final Test: 0.9429)', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('results/complete_experiment_summary_with_test.png', dpi=300, bbox_inches='tight')
print("Saved: results/complete_experiment_summary_with_test.png")
plt.close()
