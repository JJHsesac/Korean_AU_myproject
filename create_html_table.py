import pandas as pd

df = pd.read_csv('results/all_experiments.csv')

html = """
<html>
<head>
<style>
body {
    font-family: 'Arial', sans-serif;
    margin: 20px;
    background: #f5f5f5;
}
h1 {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 30px;
}
table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
thead {
    background: #3498db;
    color: white;
}
th {
    padding: 15px;
    text-align: left;
    font-weight: bold;
    font-size: 14px;
}
td {
    padding: 12px 15px;
    border-bottom: 1px solid #ecf0f1;
}
tr:hover {
    background: #f8f9fa;
}
.phase-1 { background: #ffe5e5; }
.phase-2 { background: #e5f5ff; }
.phase-3 { background: #fff5e5; }
.phase-4 { background: #f0e5ff; }
.phase-5 { background: #e5ffe5; }
.highlight { 
    font-weight: bold;
    color: #e74c3c;
}
</style>
</head>
<body>
<h1>Korean Hate Speech Detection: Experimental Results</h1>
<table>
<thead>
<tr>
<th>Phase</th>
<th>Model</th>
<th>Method</th>
<th>F1-Score</th>
<th>Note</th>
</tr>
</thead>
<tbody>
"""

for _, row in df.iterrows():
    phase_class = f"phase-{row['Phase']}"
    highlight = 'class="highlight"' if row['F1_Score'] >= 0.935 else ''
    html += f"""
<tr class="{phase_class}">
<td>Phase {row['Phase']}</td>
<td>{row['Model']}</td>
<td>{row['Method']}</td>
<td {highlight}>{row['F1_Score']:.4f}</td>
<td>{row['Note']}</td>
</tr>
"""

html += """
</tbody>
</table>
</body>
</html>
"""

with open('results/experiment_results.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("✅ HTML 표 생성 완료: results/experiment_results.html")
