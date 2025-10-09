import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm

# 한글 폰트 설정 시도
try:
    # 시스템에 있는 폰트 찾기
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    # NanumGothic 또는 다른 한글 폰트 찾기
    korean_font = None
    for font in font_list:
        if 'Nanum' in font or 'NanumGothic' in font:
            korean_font = fm.FontProperties(fname=font).get_name()
            break
    
    if korean_font:
        plt.rcParams['font.family'] = korean_font
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('results/all_experiments.csv')

# Figure 생성
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Phase 1: 4개 모델 비교 (좌상단)
ax1 = fig.add_subplot(gs[0, 0])
phase1 = df[df['Phase'] == 1].sort_values('F1_Score', ascending=True)
colors1 = ['#FF6B6B', '#4ECDC4', '#FFA07A', '#95E1D3']
bars1 = ax1.barh(phase1['Model'], phase1['F1_Score'], color=colors1, edgecolor='black')
ax1.set_xlabel('F1-Score', fontsize=11)
ax1.set_title('Phase 1: Initial Model Selection (4 Models)', fontsize=12, fontweight='bold')
ax1.set_xlim(0.87, 0.92)
ax1.grid(True, alpha=0.3, axis='x')
for i, (bar, score) in enumerate(zip(bars1, phase1['F1_Score'])):
    ax1.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{score:.4f}', va='center', fontsize=9, fontweight='bold')

# 2. Phase별 최고 성능 추이 (상단 중앙+우측 합침)
ax2 = fig.add_subplot(gs[0, 1:])
phase_best = df.groupby('Phase')['F1_Score'].max().reset_index()
ax2.plot(phase_best['Phase'], phase_best['F1_Score'], 
         marker='o', linewidth=3, markersize=12, color='#2E86AB')
ax2.fill_between(phase_best['Phase'], phase_best['F1_Score'], 
                  alpha=0.3, color='#2E86AB')
ax2.set_xlabel('Phase', fontsize=12)
ax2.set_ylabel('Best F1-Score', fontsize=12)
ax2.set_title('Performance Improvement Across Phases', fontsize=13, fontweight='bold')
ax2.set_xticks(range(1, 6))
ax2.set_xticklabels(['Phase 1\n4 Models', 'Phase 2\nAEDA', 'Phase 3\nTuning', 
                     'Phase 4\nTAPT', 'Phase 5\nEnsemble'])
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.88, 0.945)
# 값 표시
for idx, row in phase_best.iterrows():
    ax2.annotate(f'{row["F1_Score"]:.4f}', 
                 xy=(row['Phase'], row['F1_Score']),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 3. KcBERT vs ELECTRA 성능 비교 (중간 좌측)
ax3 = fig.add_subplot(gs[1, 0])
comparison_data = df[df['Phase'].isin([1, 2, 3, 4])]
pivot_data = comparison_data.pivot_table(values='F1_Score', index='Phase', columns='Model')
x = np.arange(len(pivot_data))
width = 0.35
if 'KcBERT' in pivot_data.columns:
    ax3.bar(x - width/2, pivot_data['KcBERT'], width, label='KcBERT', 
            color='#FF6B6B', edgecolor='black')
if 'ELECTRA' in pivot_data.columns:
    ax3.bar(x + width/2, pivot_data['ELECTRA'], width, label='ELECTRA', 
            color='#4ECDC4', edgecolor='black')
ax3.set_xlabel('Phase', fontsize=11)
ax3.set_ylabel('F1-Score', fontsize=11)
ax3.set_title('KcBERT vs ELECTRA Progress', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(['P1', 'P2', 'P3', 'P4'])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0.88, 0.94)

# 4. 방법론별 개선도 (중간 중앙)
ax4 = fig.add_subplot(gs[1, 1])
methods_map = {
    'Baseline': 0.9101,
    'AEDA': 0.9267,
    'Tuning': 0.9315,
    'TAPT': 0.9329,
    'Ensemble': 0.9383
}
improvements = [0]
for i in range(1, len(methods_map)):
    prev_score = list(methods_map.values())[i-1]
    curr_score = list(methods_map.values())[i]
    improvements.append((curr_score - prev_score) * 100)

colors_method = ['#95E1D3', '#FFA07A', '#F38181', '#AA96DA', '#FCBAD3']
bars4 = ax4.bar(range(1, len(methods_map)), improvements[1:], 
                color=colors_method[1:], edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(1, len(methods_map)))
ax4.set_xticklabels(['AEDA', 'Tuning', 'TAPT', 'Ensemble'], rotation=15)
ax4.set_ylabel('Improvement (%p)', fontsize=11)
ax4.set_title('Incremental Improvement by Method', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
for bar, imp in zip(bars4, improvements[1:]):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'+{imp:.2f}%p', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 5. 누적 개선도 (중간 우측)
ax5 = fig.add_subplot(gs[1, 2])
cumulative = [0.9101]
for score in list(methods_map.values())[1:]:
    cumulative.append(score)
baseline = cumulative[0]
cumulative_improvement = [(s - baseline) * 100 for s in cumulative]
ax5.plot(range(len(cumulative_improvement)), cumulative_improvement,
         marker='s', linewidth=2.5, markersize=10, color='#E63946')
ax5.fill_between(range(len(cumulative_improvement)), cumulative_improvement,
                  alpha=0.3, color='#E63946')
ax5.set_xticks(range(len(methods_map)))
ax5.set_xticklabels(['Baseline', 'AEDA', 'Tuning', 'TAPT', 'Ensemble'], rotation=15)
ax5.set_ylabel('Cumulative Improvement (%p)', fontsize=11)
ax5.set_title('Cumulative Performance Gain', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
for i, imp in enumerate(cumulative_improvement):
    ax5.annotate(f'+{imp:.2f}%p', xy=(i, imp), xytext=(0, 5),
                textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

# 6. 최종 결과 바 차트 (하단 전체)
ax6 = fig.add_subplot(gs[2, :])
final_methods = ['Baseline\n(KcBERT)', 'AEDA\n(+Data Aug)', 
                 'Tuning\n(+Hyperparams)', 'TAPT\n(+Domain Adapt)', 
                 'Ensemble\n(KcBERT+ELECTRA)']
final_scores = [0.9101, 0.9267, 0.9315, 0.9329, 0.9383]
colors_final = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
bars6 = ax6.bar(range(len(final_methods)), final_scores, 
                color=colors_final, edgecolor='black', linewidth=2, width=0.6)
ax6.set_xticks(range(len(final_methods)))
ax6.set_xticklabels(final_methods, fontsize=11)
ax6.set_ylabel('F1-Score', fontsize=12)
ax6.set_title('Complete Experimental Journey: From Baseline to Ensemble', 
              fontsize=14, fontweight='bold', pad=20)
ax6.set_ylim(0.88, 0.95)
ax6.grid(True, alpha=0.3, axis='y')
ax6.axhline(y=0.9101, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Baseline')
ax6.legend(loc='lower right')

# 값과 개선도 표시
for i, (bar, score) in enumerate(zip(bars6, final_scores)):
    height = bar.get_height()
    # F1 점수
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{score:.4f}', ha='center', va='bottom', 
             fontsize=11, fontweight='bold')
    # 개선도
    if i > 0:
        improvement = (score - final_scores[0]) * 100
        ax6.text(bar.get_x() + bar.get_width()/2., height - 0.008,
                 f'+{improvement:.2f}%p', ha='center', va='top',
                 fontsize=9, color='darkgreen', fontweight='bold')

plt.suptitle('Korean Hate Speech Detection: Experimental Results Summary', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('results/complete_experiment_summary.png', dpi=300, bbox_inches='tight')
print("Success: results/complete_experiment_summary.png")
plt.close()

# 추가: 간단한 테이블 이미지
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
for phase in range(1, 6):
    phase_data = df[df['Phase'] == phase]
    for _, row in phase_data.iterrows():
        table_data.append([
            f"Phase {row['Phase']}", 
            row['Model'], 
            row['Method'],
            f"{row['F1_Score']:.4f}",
            row['Note']
        ])

table = ax.table(cellText=table_data, 
                colLabels=['Phase', 'Model', 'Method', 'F1-Score', 'Note'],
                cellLoc='left', loc='center',
                colWidths=[0.1, 0.15, 0.2, 0.12, 0.43])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# 헤더 스타일
for i in range(5):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Phase별 색상
phase_colors = {1: '#FFE5E5', 2: '#E5F5FF', 3: '#FFF5E5', 4: '#F0E5FF', 5: '#E5FFE5'}
for i, row in enumerate(table_data, 1):
    phase = int(row[0].split()[1])
    for j in range(5):
        table[(i, j)].set_facecolor(phase_colors[phase])

plt.title('Detailed Experimental Results Table', fontsize=14, fontweight='bold', pad=20)
plt.savefig('results/experiment_table.png', dpi=300, bbox_inches='tight')
print("Success: results/experiment_table.png")
plt.close()
