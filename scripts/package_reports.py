"""Curate executed outputs and export a standalone offline notebook report."""
from pathlib import Path
import shutil,html,json,importlib.metadata
import nbformat,mistune

root=Path(__file__).resolve().parents[1]
outputs=root/'sp500_outputs'
report=root/'reports'
report.mkdir(exist_ok=True)
if not (outputs/'run_summary.json').exists():
    raise FileNotFoundError('Run scripts/execute_notebook.py from the repository first.')
for p in outputs.iterdir():
    if p.suffix in ['.csv','.json','.md','.png']:
        target=(root/'data/processed'/p.name) if p.name=='sp500_forecast_2026.csv' else report/p.name
        shutil.copy2(p,target)
nb=nbformat.read(root/'notebooks/sp500_eda_ml_forecast_2026.ipynb',as_version=4)
markdown=mistune.create_markdown(escape=False,plugins=['table','url'])
parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S&amp;P 500 Research | Muhammad Ibrahim Shahrukh</title>',
       '<style>body{margin:0;background:#e2e8f0;color:#1e293b;font:16px/1.65 system-ui,sans-serif}main{max-width:1220px;margin:28px auto;background:white;padding:30px;border-radius:18px}section{margin:24px 0}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{padding:9px 12px;border-bottom:1px solid #e2e8f0;text-align:left}th{background:#ede9fe}pre{overflow:auto;background:#0f172a;color:#e2e8f0;padding:15px;border-radius:10px;font-size:12px}a{color:#0369a1}details{margin:12px 0}summary{cursor:pointer}.output{overflow-x:auto}h1,h2,h3{line-height:1.25}</style></head><body><main>']
for cell in nb.cells:
    if cell.cell_type=='markdown':parts.append('<section>'+markdown(cell.source)+'</section>')
    elif cell.cell_type=='code':
        parts.append('<details><summary>View Python code</summary><pre>'+html.escape(cell.source)+'</pre></details>')
        for out in cell.outputs:
            if out.output_type=='stream':parts.append('<pre>'+html.escape(out.text)+'</pre>')
            else:
                d=out.get('data',{})
                if 'text/html' in d:parts.append('<div class="output">'+d['text/html']+'</div>')
                elif 'image/png' in d:parts.append('<img alt="Research visualization" src="data:image/png;base64,'+d['image/png']+'">')
                elif 'text/markdown' in d:parts.append(markdown(d['text/markdown']))
                elif 'text/plain' in d:parts.append('<pre>'+html.escape(d['text/plain'])+'</pre>')
parts.append('</main></body></html>')
(report/'sp500_research_report.html').write_text('\n'.join(parts),encoding='utf-8')
print('Curated outputs and standalone HTML report written.')
