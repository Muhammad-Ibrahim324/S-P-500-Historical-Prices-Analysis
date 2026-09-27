"""Run every cell in a fresh kernel. Any error fails execution."""
from pathlib import Path
import json,time,sys,os
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
path=root/'notebooks/sp500_eda_ml_forecast_2026.ipynb'
nb=nbformat.read(path,as_version=4)
t0=time.perf_counter()
def progress(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print(f'Executing cell {cell_index+1}/{len(nb.cells)}',flush=True)
if '--in-process' in sys.argv:
    # Socket-free IPython execution for restricted environments. Start this script
    # as a fresh process; every actual notebook code cell runs in order.
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(root)
    shell=InteractiveShell.instance()
    count=0
    for i,cell in enumerate(nb.cells):
        if cell.cell_type!='code':continue
        progress(cell,i)
        count+=1
        with capture_output(stdout=True,stderr=True,display=True) as cap:
            result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec:raise result.error_before_exec
        if result.error_in_exec:raise result.error_in_exec
        cell.execution_count=count
        cell.outputs=[]
        if cap.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
        if cap.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
        for out in cap.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
else:
    client=NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(root)}},allow_errors=False,on_cell_start=progress)
    client.execute()
nbformat.write(nb,path)
cc=[c for c in nb.cells if c.cell_type=='code']
errors=[o for c in cc for o in c.get('outputs',[]) if o.output_type=='error']
assert not errors and all(c.execution_count is not None for c in cc)
pngs=sum('image/png' in o.get('data',{}) for c in cc for o in c.outputs)
assert pngs>=12
report={'status':'passed','execution_mode':'fresh-process IPython' if '--in-process' in sys.argv else 'Jupyter kernel',
    'code_cells_executed':len(cc),'error_outputs':len(errors),'embedded_png_outputs':pngs,'elapsed_seconds':round(time.perf_counter()-t0,2)}
(root/'reports/execution_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
