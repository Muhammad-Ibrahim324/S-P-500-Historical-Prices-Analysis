"""Build a self-contained Kaggle notebook, ready for fresh-kernel execution."""
from pathlib import Path
import textwrap
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
def section(n,title,body,color='#6d28d9'):
    md(f'<div style="background:{color};color:white;padding:18px 24px;border-radius:12px"><h2 style="color:white;margin:0">{n} / {title}</h2></div>\n\n{body}')

md('''
<div style="background:linear-gradient(120deg,#071426,#164e63,#4c1d95);color:white;padding:36px;border-radius:18px;border-bottom:6px solid #2dd4bf">
<p style="color:#5eead4;letter-spacing:3px">MARKET RESEARCH LAB · 1980—2026</p>
<h1 style="color:white;font-size:38px">S&amp;P 500: History, Risk &amp; 2026 Forecasts</h1>
<p style="font-size:19px;color:#e0e7ff">From clean market data to honest machine-learning evaluation</p>
<p style="color:#fbbf24">Author: Muhammad Ibrahim Shahrukh</p></div>

**Question:** Can machine learning improve on carrying the latest S&P 500 close forward over the remaining 2026 trading sessions?

This is the **S&P 500 price index (`^GSPC`)**, measured in **index points**, not a stock, an ETF, or a dividend-reinvested portfolio. The supplied snapshot ends **September 25, 2026**. Forecasts cover **September 28–December 31, 2026** and are research scenarios, not promises.

**Route:** provenance → audit → history → risk → seasonality → features → temporal validation → untouched test → forecast → limitations.

<div style="background:#ecfeff;border-left:6px solid #0891b2;padding:16px;color:#164e63"><b>Beginner guide.</b> Run cells in order. Expand the interactive Plotly panels to hover, zoom and toggle traces. Every panel includes an embedded Matplotlib preview for viewers that block JavaScript.</div>

The notebook uses the attached clean CSV. It makes no market-data download and requires no API key or internet once dependencies are installed.
''')
section('01','Set up the research environment','Fixed seeds, visible versions and consistent colors make the experiment reproducible.','#0369a1')
code('''
from pathlib import Path
import sys, json, time, importlib.metadata
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from IPython.display import display, HTML, Markdown, Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
    HAS_XGB=True
except ImportError:
    HAS_XGB=False
    print('XGBoost unavailable: candidate omitted. Install requirements.txt for the full comparison.')
SEED=42
np.random.seed(SEED)
DATA_PATH=None  # Optional: Path('/kaggle/input/your-dataset/sp500_daily_clean.csv')
OUTPUT_DIR=Path('/kaggle/working/sp500_outputs') if Path('/kaggle/working').exists() else Path('sp500_outputs')
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
COLORS=['#0891b2','#7c3aed','#f59e0b','#e11d48','#059669','#2563eb','#db2777','#475569']
plt.rcParams.update({'figure.facecolor':'#f8fafc','axes.facecolor':'#f8fafc','axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'font.size':10,
    'axes.prop_cycle':matplotlib.cycler(color=COLORS)})
px.defaults.template='plotly_white'
px.defaults.color_discrete_sequence=COLORS
VERSIONS={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','matplotlib','plotly']}
if HAS_XGB:VERSIONS['xgboost']=importlib.metadata.version('xgboost')
print('Python',sys.version.split()[0],'| Packages:',VERSIONS)
display(HTML('<script>'+get_plotlyjs()+'</script>'))
PANEL_COUNT=0
def panel(fig,static,name):
    global PANEL_COUNT
    PANEL_COUNT+=1
    fig.update_layout(font=dict(family='Arial',size=13,color='#1e293b'),paper_bgcolor='#f8fafc',
        plot_bgcolor='#f8fafc',margin=dict(l=55,r=35,t=80,b=55),height=540,title_font=dict(size=21,color='#0f172a'))
    static.tight_layout()
    static.savefig(OUTPUT_DIR/f'{name}.png',dpi=135,bbox_inches='tight')
    display(Image(filename=str(OUTPUT_DIR/f'{name}.png')))
    plt.close(static)
    html=fig.to_html(full_html=False,include_plotlyjs=False,config={'displaylogo':False,'responsive':True})
    display(HTML('<details style="border:1px solid #a5b4fc;border-radius:10px;padding:12px"><summary style="color:#6d28d9;cursor:pointer;font-weight:bold">Open interactive Plotly panel</summary>'+html+'</details>'))
    fig.write_html(OUTPUT_DIR/f'{name}.html',include_plotlyjs=True,full_html=True)
def table(frame,caption=''):
    display(HTML('<div style="border-left:5px solid #7c3aed;padding:10px;background:#f5f3ff;color:#312e81"><b>'+caption+'</b></div>'))
    display(frame)
''')
section('02','Load and audit the snapshot','The loader supports the repository, an adjacent CSV and Kaggle’s nested input folders. Paths are deduplicated before checking for ambiguous inputs.','#0369a1')
code('''
if DATA_PATH is None:
    candidates=[Path('sp500_daily_clean.csv'),Path('data/processed/sp500_daily_clean.csv'),Path('../data/processed/sp500_daily_clean.csv')]
    if Path('/kaggle/input').exists():candidates+=list(Path('/kaggle/input').rglob('sp500_daily_clean.csv'))
    candidates=sorted({p.resolve() for p in candidates if p.is_file()})
    if len(candidates)!=1:raise ValueError(f'Expected one clean CSV, found {len(candidates)}. Set DATA_PATH explicitly: {candidates}')
    DATA_PATH=candidates[0]
df=pd.read_csv(DATA_PATH,parse_dates=['date'])
assert df.columns.tolist()==['date','ticker','open','high','low','close','volume']
assert df.ticker.eq('^GSPC').all() and df.date.is_monotonic_increasing and df.date.is_unique
assert not df.isna().any().any()
assert np.isfinite(df[['open','high','low','close','volume']]).all().all()
assert (df[['open','high','low','close']]>0).all().all()
assert (df.high>=df[['open','low','close']].max(axis=1)).all()
assert (df.low<=df[['open','high','close']].min(axis=1)).all()
assert (df.volume>=0).all() and (df.volume%1==0).all()
assert (df.date.dt.dayofweek<5).all()
assert df.date.max()==pd.Timestamp('2026-09-25'),'Update the protocol before using a different snapshot.'
df=df.set_index('date')
quality=pd.DataFrame({'check':['Rows','Missing cells','Duplicate dates','Zero volume observations','Weekend rows'],
    'result':[len(df),int(df.isna().sum().sum()),int(df.index.duplicated().sum()),int(df.volume.eq(0).sum()),int((df.index.dayofweek>4).sum())]})
table(quality,'Data quality dashboard')
display(df.head())
display(df[['open','high','low','close','volume']].describe().round(2))
print('Retained zero-volume dates:',df.index[df.volume.eq(0)].strftime('%Y-%m-%d').tolist())
''')
md('''
| Column | Short description | Meaning |
|---|---|---|
| `date` | Trading session date | ISO date; no intraday timestamp |
| `ticker` | Yahoo Finance index symbol | Always `^GSPC` |
| `open` | Opening index level | Index points |
| `high` | Highest daily index level | Index points |
| `low` | Lowest daily index level | Index points |
| `close` | Closing index level | Index points; forecast target |
| `volume` | Yahoo reported volume | Provider field; not index shares |

**Cleaning:** remove the two metadata rows (`Ticker`, `Date`), flatten headers, validate types and OHLC ordering, sort dates, add the source ticker and round OHLC to two decimals. Maximum rounding change: **0.0002343751 points**. All 11,779 observations are preserved, including the zero-volume record on May 24, 2023. No interpolation, winsorization, outlier deletion or new Yahoo download was performed.

**Verification limits:** structural checks do not independently verify every source quote. Original extraction time, yfinance version, timezone options and adjustment arguments were not provided. The absence of `Adj Close` does not establish how the original data were adjusted. No dividend-inclusive return claims are made.

Trading dates are observations, not a complete calendar. A full historical exchange-session completeness check was not performed. The repository retains the raw input and input/output SHA-256 hashes.
''')
section('03','Price, trend and drawdown','A log scale compares proportional changes across decades. Drawdown measures the decline from the previous closing high.','#0f766e')
code('''
close=df.close
ret=close.pct_change(fill_method=None)
logret=np.log(close).diff()
drawdown=close/close.cummax()-1
vol21=logret.rolling(21).std()*np.sqrt(252)
years=(df.index[-1]-df.index[0]).days/365.25
price_cagr=(close.iloc[-1]/close.iloc[0])**(1/years)-1
ytd=close.iloc[-1]/close.loc[:'2025-12-31'].iloc[-1]-1
cards=[('OBSERVATIONS',f'{len(df):,}'),('LATEST CLOSE',f'{close.iloc[-1]:,.2f} pts'),('2026 YTD',f'{ytd:.1%}'),('MAX DRAWDOWN',f'{drawdown.min():.1%}')]
display(HTML('<div style="display:flex;flex-wrap:wrap;gap:12px">'+''.join(f'<div style="background:{COLORS[i]};color:white;padding:18px;border-radius:12px;min-width:150px"><small>{k}</small><h2 style="color:white;margin:8px 0">{v}</h2></div>' for i,(k,v) in enumerate(cards))+'</div>'))
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.12,row_heights=[.7,.3],subplot_titles=['Closing index level (log scale)','Drawdown from previous high'])
fig.add_trace(go.Scatter(x=df.index,y=close,name='Close',line_color=COLORS[0]),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=close.rolling(200).mean(),name='200-session mean',line_color=COLORS[1]),row=1,col=1)
fig.add_trace(go.Scatter(x=df.index,y=drawdown*100,name='Drawdown %',fill='tozeroy',line_color=COLORS[3]),row=2,col=1)
fig.update_yaxes(type='log',title_text='Index points',row=1,col=1);fig.update_yaxes(title_text='Percent',row=2,col=1)
fig.update_layout(title='46 years of market history')
sf,axs=plt.subplots(2,1,figsize=(12,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
axs[0].plot(df.index,close,label='Close');axs[0].plot(df.index,close.rolling(200).mean(),label='200-session mean');axs[0].set_yscale('log');axs[0].set_ylabel('Index points · log');axs[0].legend()
axs[1].fill_between(df.index,drawdown*100,0,color=COLORS[3],alpha=.7);axs[1].set_ylabel('Drawdown %');sf.suptitle('S&P 500 | Long-run history and downside risk')
panel(fig,sf,'01_history_drawdown')
display(Markdown(f'**Snapshot:** price-only CAGR is **{price_cagr:.2%}**, maximum daily-close drawdown is **{drawdown.min():.2%}**, and latest 21-session annualized volatility is **{vol21.iloc[-1]:.2%}**. CAGR excludes dividends, inflation, fees and taxes.'))
''')
section('04','Recent market structure','Explore the latest 126 sessions, trend measures and reported volume. Volume is supplied by Yahoo; the index itself is not directly traded.','#0f766e')
code('''
recent=df.tail(126)
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.75,.25],vertical_spacing=.08)
fig.add_trace(go.Candlestick(x=recent.index,open=recent.open,high=recent.high,low=recent.low,close=recent.close,name='OHLC',increasing_line_color='#059669',decreasing_line_color='#e11d48'),row=1,col=1)
for w,c in [(20,COLORS[1]),(50,COLORS[2])]:fig.add_trace(go.Scatter(x=recent.index,y=close.rolling(w).mean().loc[recent.index],name=f'{w}-session mean',line_color=c),row=1,col=1)
fig.add_trace(go.Bar(x=recent.index,y=recent.volume/1e9,name='Reported volume (bn)',marker_color=COLORS[0]),row=2,col=1)
fig.update_layout(title='Recent price and reported activity',xaxis_rangeslider_visible=False)
sf,axs=plt.subplots(2,1,figsize=(12,6),sharex=True,gridspec_kw={'height_ratios':[3,1]})
axs[0].fill_between(recent.index,recent.low,recent.high,color=COLORS[0],alpha=.2,label='Daily low–high');axs[0].plot(recent.index,recent.close,label='Close')
for w in [20,50]:axs[0].plot(recent.index,close.rolling(w).mean().loc[recent.index],label=f'{w}-session mean')
axs[0].legend(ncol=3);axs[0].set_ylabel('Index points');axs[1].bar(recent.index,recent.volume/1e9,color=COLORS[0]);axs[1].set_ylabel('Volume · bn');sf.suptitle('Recent structure | Static OHLC range preview')
panel(fig,sf,'02_recent_market')
''')
section('05','Returns, tails and volatility','Returns make eras more comparable than raw price levels. Large moves are retained because they are part of market risk.','#be123c')
code('''
r=ret.dropna()*100
fig=make_subplots(rows=2,cols=2,subplot_titles=['Daily returns (%)','Return distribution','21-session annualized volatility (%)','Largest absolute daily returns (%)'])
fig.add_trace(go.Scatter(x=r.index,y=r,mode='lines',name='Daily return',line_color=COLORS[0]),row=1,col=1)
fig.add_trace(go.Histogram(x=r,nbinsx=120,name='Frequency',marker_color=COLORS[1]),row=1,col=2)
fig.add_trace(go.Scatter(x=vol21.index,y=vol21*100,name='Volatility',line_color=COLORS[3]),row=2,col=1)
extreme=r.loc[r.abs().nlargest(10).index].sort_values()
fig.add_trace(go.Bar(x=extreme.values,y=extreme.index.strftime('%Y-%m-%d'),orientation='h',marker_color=[COLORS[3] if v<0 else COLORS[4] for v in extreme]),row=2,col=2)
fig.update_layout(title='The risk dashboard',showlegend=False)
sf,axs=plt.subplots(2,2,figsize=(13,7))
axs[0,0].plot(r.index,r,lw=.6);axs[0,0].set_title('Daily returns (%)')
axs[0,1].hist(r,bins=120,color=COLORS[1]);axs[0,1].set_title('Return distribution · full tails')
axs[1,0].plot(vol21.index,vol21*100,color=COLORS[3]);axs[1,0].set_title('Annualized 21-session volatility (%)')
axs[1,1].barh(extreme.index.strftime('%Y-%m-%d'),extreme,color=[COLORS[3] if v<0 else COLORS[4] for v in extreme]);axs[1,1].set_title('Largest absolute daily moves (%)')
panel(fig,sf,'03_risk_dashboard')
table(pd.DataFrame({'daily_mean_pct':[r.mean()],'daily_std_pct':[r.std()],'skew':[r.skew()],
    'excess_kurtosis':[r.kurt()],'positive_sessions_pct':[(r>0).mean()*100],'worst_day_pct':[r.min()],'best_day_pct':[r.max()]}).round(3),'Return distribution summary')
''')
section('06','Calendar patterns and regime shifts','Monthly returns compound daily returns. January 1980 lacks a previous month-end close, so it is omitted. September 2026 and the 2026 annual return are partial. Descriptive patterns are not proof of a tradable edge.','#b45309')
code('''
monthly=(1+ret).resample('ME').prod(min_count=1)-1
monthly.iloc[0]=np.nan
heat=pd.DataFrame({'year':monthly.index.year,'month':monthly.index.month,'return':monthly.values*100}).pivot(index='year',columns='month',values='return').tail(16)
fig=px.imshow(heat,aspect='auto',color_continuous_scale='RdYlGn',color_continuous_midpoint=0,labels={'x':'Month','y':'Year','color':'Return %'},title='Monthly returns | Latest 16 calendar years')
sf,ax=plt.subplots(figsize=(12,7));im=ax.imshow(heat,aspect='auto',cmap='RdYlGn',vmin=-15,vmax=15)
ax.set_xticks(range(12),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']);ax.set_yticks(range(len(heat)),heat.index)
for i in range(len(heat)):
    for j in range(12):
        if pd.notna(heat.iloc[i,j]):ax.text(j,i,f'{heat.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8)
sf.colorbar(im,ax=ax,label='Monthly return %');ax.set_title('Monthly returns (%) · September 2026 is partial')
panel(fig,sf,'04_monthly_heatmap')
annual=(1+ret).resample('YE').prod(min_count=1)-1
annual.iloc[0]=np.nan
weekday=pd.DataFrame({'return_pct':r,'weekday':r.index.dayofweek}).groupby('weekday').return_pct.agg(['mean','std','count'])
weekday['se']=weekday['std']/np.sqrt(weekday['count'])
acf=pd.DataFrame({'lag':range(1,31),'returns':[ret.autocorr(k) for k in range(1,31)],'squared_returns':[(ret**2).autocorr(k) for k in range(1,31)]})
fig=make_subplots(rows=1,cols=3,subplot_titles=['Annual returns (%)','Weekday mean ± 1.96 SE (%)','Serial dependence'])
fig.add_trace(go.Bar(x=annual.index.year,y=annual*100,name='Annual returns',marker_color=COLORS[0]),row=1,col=1)
fig.add_trace(go.Bar(x=['Mon','Tue','Wed','Thu','Fri'],y=weekday['mean'],error_y=dict(array=1.96*weekday.se),name='Weekday mean',marker_color=COLORS[2]),row=1,col=2)
for c in ['returns','squared_returns']:fig.add_trace(go.Scatter(x=acf.lag,y=acf[c],name=c),row=1,col=3)
fig.update_layout(title='Seasonality and persistence | Descriptive, not causal')
sf,axs=plt.subplots(1,3,figsize=(14,4))
axs[0].bar(annual.index.year,annual*100);axs[0].set_title('Annual return % · 2026 YTD')
axs[1].bar(['Mon','Tue','Wed','Thu','Fri'],weekday['mean'],yerr=1.96*weekday.se,color=COLORS[2]);axs[1].set_title('Weekday mean ± 1.96 SE (%)')
for c in ['returns','squared_returns']:axs[2].plot(acf.lag,acf[c],label=c)
axs[2].legend(fontsize=8);axs[2].set_title('Return autocorrelation')
panel(fig,sf,'05_calendar_dependence')
display(Markdown('Weekday error bars use independent-observation standard errors and do not correct for serial dependence or multiple comparisons. Squared-return correlation can reveal volatility clustering even if raw-return correlation is weak.'))
''')
section('07','Relationships and caveats','Raw price correlations can be misleading. Inspect returns, range, volatility and relative volume instead. Zero volume is excluded from ratios but preserved in the clean dataset.','#b45309')
code('''
v=df.volume.replace(0,np.nan)
relationships=pd.DataFrame({'return':ret,'abs_return':ret.abs(),'range_pct':(df.high-df.low)/close,
    'volatility_21':vol21,'relative_volume':v/v.rolling(21,min_periods=15).median()})
corr=relationships.corr(method='spearman')
fig=px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r',title='Spearman relationships | Association is not causation')
sf,ax=plt.subplots(figsize=(8,6));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
ax.set_xticks(range(len(corr)),corr.columns,rotation=35,ha='right');ax.set_yticks(range(len(corr)),corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center')
sf.colorbar(im,ax=ax);ax.set_title('Spearman correlations · pairwise available observations')
panel(fig,sf,'06_relationships')
table(relationships.describe(percentiles=[.01,.5,.99]).round(4),'Transformed-variable profile')
''')
section('08','Forecast design and leakage prevention','Predict cumulative future log returns directly for every horizon, then convert them to index levels. No invented future OHLC data are required.')
md('''
**Timing.** Forecasts are issued after the close at time *t*. Today's return and OHLC values are available. All features use observations through *t*. Target *h* is `log(close[t+h] / close[t])`.

**Temporal evaluation.** Six sequential, non-overlapping validation paths each contain 67 sessions, matching the future horizon. The latest 67 observed sessions form the untouched final test. At each origin, a training row is eligible only when **all 67 target closes have already occurred**. This purges overlapping future labels. Keep at most the latest 2,520 eligible training origins (about ten trading years).

**Predeclared selection.** Rank by mean validation path log-RMSE with equal weight per origin. Freeze the winner before computing test metrics. The test is a final diagnostic, not a tuning set. Refit the locked winner on all eligible data for the future forecast. “Best” means best among the candidates under this evaluation, not universally best.

**Candidates:** no-change, 252-session log drift, 1,260-session log drift, Ridge (`alpha=10,100,1000`), Random Forest, Extra Trees and XGBoost when installed. Scaling is fitted separately within each Ridge training fold. Price-derived features are used; historical volume definitions and the zero record make volume less suitable for this experiment.

**Calendar:** future dates exclude NYSE's published 2026 holidays. Early-close sessions remain. Unscheduled closures cannot be anticipated; historical backtests use observed dates.
''')
code('''
HOLIDAYS_2026=pd.to_datetime(['2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25','2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25'])
future_dates=pd.bdate_range(df.index[-1].to_pydatetime()+timedelta(days=1),'2026-12-31').difference(HOLIDAYS_2026)
H=len(future_dates)
assert H==67
TRAIN_WINDOW=2520
X=pd.DataFrame(index=df.index)
for lag in [0,1,2,3,4,5,10,20]:X[f'log_return_lag_{lag}']=logret.shift(lag)
for w in [5,10,21,63,126,252]:
    X[f'momentum_{w}']=np.log(close/close.shift(w))
    X[f'volatility_{w}']=logret.rolling(w).std()
    X[f'distance_ma_{w}']=close/close.rolling(w).mean()-1
X['intraday_range']=(df.high-df.low)/close
X['intraday_return']=np.log(close/df.open)
X['overnight_return']=np.log(df.open/close.shift(1))
X['drawdown_252']=close/close.rolling(252).max()-1
X=X.replace([np.inf,-np.inf],np.nan)
Y=pd.DataFrame({h:np.log(close.shift(-h)/close) for h in range(1,H+1)})
valid_features=X.notna().all(axis=1).to_numpy()
origins=[len(df)-1-H*k for k in range(7,0,-1)]
validation_origins,test_origin=origins[:-1],origins[-1]
def train_at(origin):
    positions=np.arange(len(df))
    eligible=positions[(positions+H<=origin)&valid_features&Y.notna().all(axis=1).to_numpy()][-TRAIN_WINDOW:]
    assert len(eligible)>500 and eligible[-1]+H<=origin
    return X.iloc[eligible],Y.iloc[eligible],eligible
split_rows=[]
for i,o in enumerate(origins):
    tx,ty,ix=train_at(o)
    split_rows.append({'split':f'validation_{i+1}' if i<6 else 'test','origin':df.index[o].date(),
        'train_feature_start':df.index[ix[0]].date(),'train_feature_end':df.index[ix[-1]].date(),
        'latest_train_label':df.index[ix[-1]+H].date(),'forecast_start':df.index[o+1].date(),
        'forecast_end':df.index[o+H].date(),'training_rows':len(ix)})
splits=pd.DataFrame(split_rows)
table(splits,'Every latest training label is on or before its forecast origin')
print('Forecast sessions:',H,'| Features:',X.shape[1],'| First future date:',future_dates[0].date())
''')
section('09','Train and compare','Each candidate predicts a complete 67-session path from one historical origin. All ML candidates see the same eligible rows. This is the main computation step.')
code('''
def make_model(name):
    if name.startswith('Ridge_'):return make_pipeline(StandardScaler(),Ridge(alpha=float(name.split('_')[1])))
    if name=='RandomForest':return RandomForestRegressor(n_estimators=120,max_depth=7,min_samples_leaf=25,max_features=.8,random_state=SEED,n_jobs=2)
    if name=='ExtraTrees':return ExtraTreesRegressor(n_estimators=160,max_depth=8,min_samples_leaf=20,max_features=.8,random_state=SEED,n_jobs=2)
    if name=='XGBoost':return XGBRegressor(n_estimators=140,max_depth=3,learning_rate=.035,subsample=.85,colsample_bytree=.9,reg_lambda=20,objective='reg:squarederror',tree_method='hist',multi_strategy='multi_output_tree',random_state=SEED,n_jobs=2)
    raise ValueError(name)
BASELINES=['NoChange','Drift252','Drift1260']
CANDIDATES=BASELINES+['Ridge_10','Ridge_100','Ridge_1000','RandomForest','ExtraTrees']+(['XGBoost'] if HAS_XGB else [])
def predict_origin(name,origin,return_model=False):
    model=None
    if name=='NoChange':pred=np.zeros(H)
    elif name.startswith('Drift'):pred=np.arange(1,H+1)*logret.iloc[:origin+1].tail(int(name[5:])).mean()
    else:
        tx,ty,_=train_at(origin)
        model=make_model(name);model.fit(tx,ty)
        pred=np.asarray(model.predict(X.iloc[[origin]])).reshape(-1)
    assert len(pred)==H and np.isfinite(pred).all()
    return (pred,model) if return_model else pred
def evaluate(actual,predicted,origin_level):
    actual,predicted=np.asarray(actual),np.asarray(predicted)
    return {'mae_points':float(mean_absolute_error(actual,predicted)),
        'rmse_points':float(np.sqrt(mean_squared_error(actual,predicted))),
        'mape_pct':float(np.mean(np.abs(predicted/actual-1))*100),
        'log_rmse_pct':float(np.sqrt(np.mean(np.log(predicted/actual)**2))*100),'r2':float(r2_score(actual,predicted)),
        'endpoint_error_pct':float((predicted[-1]/actual[-1]-1)*100),
        'endpoint_direction_correct':int(np.sign(predicted[-1]/origin_level-1)==np.sign(actual[-1]/origin_level-1))}
t0=time.perf_counter();validation_rows=[];validation_paths=[]
for fold,origin in enumerate(validation_origins,1):
    actual=close.iloc[origin+1:origin+H+1].to_numpy()
    for name in CANDIDATES:
        prediction=close.iloc[origin]*np.exp(predict_origin(name,origin))
        validation_rows.append({'fold':fold,'model':name,'origin':str(df.index[origin].date()),**evaluate(actual,prediction,close.iloc[origin])})
        validation_paths.extend({'fold':fold,'model':name,'horizon':h+1,'date':str(df.index[origin+1+h].date()),'actual':float(actual[h]),'prediction':float(prediction[h])} for h in range(H))
    print(f'Validation origin {fold}/6: {df.index[origin].date()} | {time.perf_counter()-t0:.1f}s',flush=True)
validation_metrics=pd.DataFrame(validation_rows)
leaderboard=validation_metrics.groupby('model').agg(mean_log_rmse_pct=('log_rmse_pct','mean'),std_log_rmse_pct=('log_rmse_pct','std'),mean_mae_points=('mae_points','mean'),mean_rmse_points=('rmse_points','mean'),mean_mape_pct=('mape_pct','mean'),endpoint_direction_accuracy=('endpoint_direction_correct','mean')).sort_values(['mean_log_rmse_pct','model'])
baseline_score=leaderboard.loc['NoChange','mean_log_rmse_pct']
leaderboard['skill_vs_nochange_pct']=100*(1-leaderboard.mean_log_rmse_pct/baseline_score)
WINNER=leaderboard.index[0]  # Frozen before the test is evaluated.
ML_WINNER=next(n for n in leaderboard.index if n not in BASELINES)
table(leaderboard.round(4),'Validation leaderboard · lower log-RMSE is better')
print('Locked winner:',WINNER,'| Best ML candidate:',ML_WINNER)
''')
code('''
lb=leaderboard.reset_index()
fig=px.bar(lb,x='model',y='mean_log_rmse_pct',error_y='std_log_rmse_pct',color='model',title='Validation | Mean log-RMSE ± fold standard deviation')
fig.update_layout(showlegend=False,yaxis_title='Log-RMSE × 100')
sf,ax=plt.subplots(figsize=(12,4.5));ax.bar(lb.model,lb.mean_log_rmse_pct,yerr=lb.std_log_rmse_pct,color=[COLORS[i%len(COLORS)] for i in range(len(lb))],capsize=3)
ax.tick_params(axis='x',rotation=25);ax.set_ylabel('Log-RMSE × 100');ax.set_title('Six historical 67-session forecasts · error bars show fold variation')
panel(fig,sf,'07_model_comparison')
fold_view=validation_metrics.pivot(index='fold',columns='model',values='log_rmse_pct')
fig=px.line(fold_view,markers=True,title='Does performance persist across periods?',labels={'value':'Log-RMSE × 100','fold':'Validation origin'})
sf,ax=plt.subplots(figsize=(12,4.5))
for name in fold_view:ax.plot(fold_view.index,fold_view[name],marker='o',label=name)
ax.legend(ncol=3,fontsize=8);ax.set_xlabel('Validation origin');ax.set_ylabel('Log-RMSE × 100');ax.set_title('Performance stability across validation periods')
panel(fig,sf,'08_fold_stability')
''')
md('''
**Reading metrics:** MAE and RMSE are in index points. MAPE is proportional error. Log-RMSE measures multiplicative error and determines selection. Positive skill means improvement over no-change. R² can be negative on a market path; it does not determine the winner. Direction accuracy uses only six endpoint trials, not daily trades. Fold standard deviations are not confidence intervals.

Fixed, modest hyperparameters keep the comparison practical. Exhaustive tuning is not automatically better for one noisy index series. Screening candidates can still overfit six folds, making the untouched test essential.
''')
section('10','Untouched test and uncertainty check','Evaluate the locked winner on the last 67 observations. No closes inside the test path update the forecast. This matches the deployment task.','#be123c')
code('''
test_actual=close.iloc[test_origin+1:test_origin+H+1].to_numpy()
test_dates=df.index[test_origin+1:test_origin+H+1]
test_models=list(dict.fromkeys([WINNER,'NoChange',ML_WINNER]))
test_predictions={n:close.iloc[test_origin]*np.exp(predict_origin(n,test_origin)) for n in test_models}
test_metrics=pd.DataFrame([{'model':n,**evaluate(test_actual,p,close.iloc[test_origin])} for n,p in test_predictions.items()]).set_index('model')
test_metrics['rmse_skill_vs_nochange_pct']=100*(1-test_metrics.rmse_points/test_metrics.loc['NoChange','rmse_points'])
table(test_metrics.round(4),'Untouched test · the selection is not changed after seeing this table')
def bootstrap_bands(origin,center,draws=5000,block=5):
    past=logret.iloc[:origin+1].dropna().tail(1260).to_numpy()
    shocks=past-past.mean()
    rng=np.random.default_rng(SEED+origin)
    starts=rng.integers(0,len(shocks)-block+1,size=(draws,int(np.ceil(H/block))))
    ix=(starts[:,:,None]+np.arange(block)).reshape(draws,-1)[:,:H]
    log_noise=np.cumsum(shocks[ix],axis=1)
    quantiles=np.quantile(log_noise,[.025,.1,.9,.975],axis=0)
    return np.asarray(center)[None,:]*np.exp(quantiles)
test_bands=bootstrap_bands(test_origin,test_predictions[WINNER])
coverage80=np.mean((test_actual>=test_bands[1])&(test_actual<=test_bands[2]))
coverage95=np.mean((test_actual>=test_bands[0])&(test_actual<=test_bands[3]))
print(f'Test-path coverage: 80% simulation band = {coverage80:.1%}; 95% simulation band = {coverage95:.1%}')
print('Coverage is a fraction of correlated horizons in one path, not an independently calibrated confidence level.')
fig=go.Figure()
fig.add_trace(go.Scatter(x=test_dates,y=test_bands[0],line=dict(width=0),showlegend=False))
fig.add_trace(go.Scatter(x=test_dates,y=test_bands[3],fill='tonexty',fillcolor='rgba(124,58,237,.12)',line=dict(width=0),name='95% bootstrap scenario band'))
fig.add_trace(go.Scatter(x=test_dates,y=test_actual,name='Observed',line=dict(color='#0f172a',width=3)))
for n,p in test_predictions.items():fig.add_trace(go.Scatter(x=test_dates,y=p,name=n))
fig.update_layout(title='Untouched test | One forecast origin, no peeking',yaxis_title='Index points')
sf,ax=plt.subplots(figsize=(12,5));ax.fill_between(test_dates,test_bands[0],test_bands[3],color=COLORS[1],alpha=.12,label='95% bootstrap band');ax.plot(test_dates,test_actual,color='#0f172a',lw=2.5,label='Observed')
for n,p in test_predictions.items():ax.plot(test_dates,p,label=n)
ax.legend(fontsize=8);ax.set_ylabel('Index points');ax.set_title('Untouched 67-session test | Origin '+str(df.index[test_origin].date()))
panel(fig,sf,'09_untouched_test')
residual=test_predictions[WINNER]-test_actual
fig=make_subplots(rows=1,cols=2,subplot_titles=['Signed error (points)','Absolute error (points)'])
fig.add_trace(go.Bar(x=np.arange(1,H+1),y=residual,marker_color=COLORS[3]),row=1,col=1)
fig.add_trace(go.Scatter(x=np.arange(1,H+1),y=np.abs(residual),name='Selected model'),row=1,col=2)
fig.add_trace(go.Scatter(x=np.arange(1,H+1),y=np.abs(test_predictions['NoChange']-test_actual),name='No-change'),row=1,col=2)
fig.update_layout(title='Error diagnostics | Long horizons can fail differently')
sf,axs=plt.subplots(1,2,figsize=(12,4));axs[0].bar(np.arange(1,H+1),residual,color=COLORS[3]);axs[0].set_title('Signed error · index points')
axs[1].plot(np.arange(1,H+1),np.abs(residual),label=WINNER);axs[1].plot(np.arange(1,H+1),np.abs(test_predictions['NoChange']-test_actual),label='NoChange');axs[1].legend();axs[1].set_title('Absolute error · index points')
for ax in axs:ax.set_xlabel('Forecast horizon (sessions)')
panel(fig,sf,'10_error_diagnostics')
''')
md('''
**Bands:** resample contiguous five-session blocks of the last 1,260 observed log returns after subtracting their mean. Simulate 5,000 future shock paths around the selected model. The 10th/90th and 2.5th/97.5th percentiles form **80% and 95% bootstrap scenario bands**, not guaranteed or calibrated prediction intervals.

This assumes recent shocks remain relevant. It omits parameter uncertainty, future surprises, volatility regime changes and model bias. Coverage above is reported without using it to tune the bands. A smooth model path does not mean real prices will move smoothly.
''')
section('11','Feature interpretation','Extra Trees importance explains one fitted candidate, even if another model wins. It does not establish causality or out-of-sample usefulness.')
code('''
_,interpret_model=predict_origin('ExtraTrees',test_origin,return_model=True)
importance=pd.Series(interpret_model.feature_importances_,index=X.columns).sort_values(ascending=False)
top=importance.head(15).sort_values()
fig=px.bar(x=top.values,y=top.index,orientation='h',labels={'x':'Impurity-based importance','y':'Feature'},title='Extra Trees | Training-based feature importance')
sf,ax=plt.subplots(figsize=(11,6));ax.barh(top.index,top.values,color=COLORS[1]);ax.set_xlabel('Impurity-based importance');ax.set_title('Extra Trees · correlated features share credit')
panel(fig,sf,'11_feature_importance')
display(Markdown('Importance is computed from a model trained at the test origin, without test outcomes. Impurity importance can favor some feature types and redistribute credit across correlated indicators.'))
''')
section('12','Remaining 2026 forecast','Refit the locked validation winner using all eligible data. Forecasts remain separate from historical observations.','#0f766e')
code('''
final_log,final_model=predict_origin(WINNER,len(df)-1,return_model=True)
central=close.iloc[-1]*np.exp(final_log)
bands=bootstrap_bands(len(df)-1,central)
forecast=pd.DataFrame({'date':future_dates,'ticker':'^GSPC','forecast_origin':df.index[-1],
    'horizon_sessions':np.arange(1,H+1),'model':WINNER,'forecast_close':central,
    'scenario_lower_80':bands[1],'scenario_upper_80':bands[2],'scenario_lower_95':bands[0],
    'scenario_upper_95':bands[3],'no_change_close':close.iloc[-1]})
assert forecast.date.is_unique and (forecast.date>df.index[-1]).all()
assert not forecast.date.isin(HOLIDAYS_2026).any() and (forecast.date.dt.dayofweek<5).all()
assert np.isfinite(forecast.select_dtypes('number')).all().all()
assert (forecast.scenario_lower_95<=forecast.scenario_lower_80).all()
assert (forecast.scenario_lower_80<=forecast.scenario_upper_80).all()
assert (forecast.scenario_upper_80<=forecast.scenario_upper_95).all()
fig=go.Figure()
fig.add_trace(go.Scatter(x=df.index[-126:],y=close.iloc[-126:],name='Observed',line=dict(color='#0f172a',width=3)))
for lo,hi,label,fill in [(bands[0],bands[3],'95% bootstrap scenario','rgba(124,58,237,.12)'),(bands[1],bands[2],'80% bootstrap scenario','rgba(8,145,178,.23)')]:
    fig.add_trace(go.Scatter(x=future_dates,y=lo,line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=future_dates,y=hi,fill='tonexty',fillcolor=fill,line=dict(width=0),name=label))
fig.add_trace(go.Scatter(x=future_dates,y=central,name=f'Central path: {WINNER}',line=dict(color=COLORS[1],width=3)))
fig.add_trace(go.Scatter(x=future_dates,y=np.repeat(close.iloc[-1],H),name='No-change benchmark',line=dict(color=COLORS[2],dash='dash')))
fig.update_layout(title='S&P 500 | Remaining 2026 forecast and scenario bands',yaxis_title='Index points')
sf,ax=plt.subplots(figsize=(12,6));ax.plot(df.index[-126:],close.iloc[-126:],color='#0f172a',lw=2,label='Observed')
ax.fill_between(future_dates,bands[0],bands[3],alpha=.13,color=COLORS[1],label='95% bootstrap scenario');ax.fill_between(future_dates,bands[1],bands[2],alpha=.23,color=COLORS[0],label='80% bootstrap scenario')
ax.plot(future_dates,central,color=COLORS[1],lw=2,label=WINNER);ax.plot(future_dates,np.repeat(close.iloc[-1],H),color=COLORS[2],ls='--',label='No-change');ax.axvline(df.index[-1],ls=':',color='#64748b');ax.legend(fontsize=8);ax.set_ylabel('Index points');ax.set_title('Forecast from September 25, 2026 · scenario bands are not guarantees')
panel(fig,sf,'12_forecast_2026')
month_ends=forecast.groupby(forecast.date.dt.to_period('M')).tail(1).copy()
month_ends['change_from_last_close_pct']=(month_ends.forecast_close/close.iloc[-1]-1)*100
table(month_ends[['date','forecast_close','scenario_lower_80','scenario_upper_80','scenario_lower_95','scenario_upper_95','change_from_last_close_pct']].round(2),'Last forecast session of each month · September covers three sessions')
''')
section('13','Findings and next experiments','The narrative is generated from the executed tables to keep conclusions and numerical evidence consistent.','#0369a1')
code("""
end=forecast.iloc[-1]
test_skill=test_metrics.loc[WINNER,'rmse_skill_vs_nochange_pct']
outcome='beat' if test_skill>0 else ('matched' if abs(test_skill)<1e-9 else 'trailed')
summary=f'''
## Research findings

- **Data:** {len(df):,} observations from {df.index[0].date()} through {df.index[-1].date()}; last close **{close.iloc[-1]:,.2f} index points**.
- **Observed context:** 2026 YTD price return **{ytd:.2%}**; latest 21-session annualized volatility **{vol21.iloc[-1]:.2%}**. These are snapshot statistics, not current-news explanations.
- **Selection:** **{WINNER}** achieved mean validation log-RMSE **{leaderboard.loc[WINNER,'mean_log_rmse_pct']:.3f}%**, versus **{baseline_score:.3f}%** for no-change. Best ML candidate: **{ML_WINNER}**.
- **Untouched test:** the winner {outcome} no-change with **{test_skill:.2f}% RMSE skill**. MAE **{test_metrics.loc[WINNER,'mae_points']:,.2f} points**, RMSE **{test_metrics.loc[WINNER,'rmse_points']:,.2f} points**, MAPE **{test_metrics.loc[WINNER,'mape_pct']:.2f}%**. Validation success need not persist.
- **Year-end central scenario:** **{end.forecast_close:,.2f} points** on **{end.date.date()}**, **{(end.forecast_close/close.iloc[-1]-1)*100:+.2f}%** versus the last close.
- **Year-end 80% bootstrap range:** **{end.scenario_lower_80:,.2f}–{end.scenario_upper_80:,.2f} points**; **95% range:** **{end.scenario_lower_95:,.2f}–{end.scenario_upper_95:,.2f}**. Test-path coverage: **{coverage80:.1%} / {coverage95:.1%}**. These are simulations, not calibrated success probabilities.

## Interpretation and limitations

The central path extrapolates historical behavior. Policy, earnings, inflation, interest rates and geopolitical developments are not model inputs; this notebook cannot attribute its forecast to those drivers. Positive momentum does not ensure gains, and a downward curve does not predict a specific crash. The no-change comparison reveals how much complexity adds.

There are six validation origins and one test path. Forecast windows are distinct, but training windows overlap and markets are dependent. No statistically significant superiority is claimed. A different window or regime can change rankings. Direct multi-horizon predictions may be jagged because adjacent horizons have different targets. Scenario bands omit parameter uncertainty and may understate risk.

## Next experiments

1. Add dated, revision-aware economic and earnings inputs, using only values available at each origin.
2. Evaluate more historical origins and nested temporal tuning before changing a production model.
3. Compare volatility models and interval calibration across many independent forecast periods.
4. Study ETF tracking, transaction costs and turnover separately. Low forecast error is not proof of a profitable strategy.

**Educational research only.** This is a price index, not a dividend-reinvested portfolio or directly investable security. No ranking, accuracy, return or future index level is guaranteed.
'''
display(Markdown(summary))
(OUTPUT_DIR/'findings.md').write_text(summary,encoding='utf-8')
forecast.to_csv(OUTPUT_DIR/'sp500_forecast_2026.csv',index=False,float_format='%.6f',date_format='%Y-%m-%d')
validation_metrics.to_csv(OUTPUT_DIR/'validation_metrics.csv',index=False)
pd.DataFrame(validation_paths).to_csv(OUTPUT_DIR/'validation_predictions.csv',index=False)
leaderboard.to_csv(OUTPUT_DIR/'model_leaderboard.csv')
test_metrics.to_csv(OUTPUT_DIR/'test_metrics.csv')
pd.DataFrame({'date':test_dates,'actual_close':test_actual,**test_predictions,'scenario_lower_80':test_bands[1],
    'scenario_upper_80':test_bands[2],'scenario_lower_95':test_bands[0],'scenario_upper_95':test_bands[3]}).to_csv(OUTPUT_DIR/'test_predictions.csv',index=False)
splits.to_csv(OUTPUT_DIR/'temporal_splits.csv',index=False)
importance.rename('importance').to_csv(OUTPUT_DIR/'feature_importance.csv')
run={'author':'Muhammad Ibrahim Shahrukh','data_end':str(df.index[-1].date()),'rows':len(df),'selected_model':WINNER,
    'best_ml_model':ML_WINNER,'candidates':CANDIDATES,'forecast_sessions':H,'year_end_forecast':float(end.forecast_close),
    'seed':SEED,'features':list(X.columns),'packages':VERSIONS,'test_coverage_80':float(coverage80),
    'test_coverage_95':float(coverage95),'chart_panels':PANEL_COUNT,'selection_rule':'mean six-fold path log-RMSE',
    'test_rmse_skill_pct':float(test_skill),'scenario_band_method':'5000 centered 5-session moving-block bootstrap paths; last 1260 returns'}
(OUTPUT_DIR/'run_summary.json').write_text(json.dumps(run,indent=2))
print(f'Complete: {PANEL_COUNT} paired visual panels. Outputs saved to {OUTPUT_DIR.resolve()}')
""")
section('14','Sources and provenance','Prepared by Muhammad Ibrahim Shahrukh · September 27, 2026. Source documentation checked on that date.','#334155')
md('''
1. **Yahoo Finance**, [S&P 500 (`^GSPC`) historical data](https://finance.yahoo.com/quote/%5EGSPC/history/). Provider identified by the user. This analysis processes the supplied snapshot without re-downloading or authenticating every quote.
2. **Ran Aroussi and contributors**, [yfinance](https://ranaroussi.github.io/yfinance/) and [`download` reference](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html). Original collection was reported as using yfinance. Retrieval time, version and arguments are unknown. Start dates are inclusive; end dates are exclusive; OHLC adjustment is configurable.
3. **S&P Dow Jones Indices**, [S&P 500 overview](https://www.spglobal.com/spdji/en/indices/equity/sp-500/). Index definition and price/total-return context.
4. **NYSE**, [Holidays and trading hours](https://www.nyse.com/trade/hours-calendars). Source for published future 2026 sessions, including early closes.
5. **scikit-learn**, [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Background on chronological evaluation. This notebook uses custom purged forecast-origin splits for 67-session targets.
6. **XGBoost**, [Multiple outputs](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html). Experimental native multi-output trees require a compatible version.

**Suggested citation:** Shahrukh, Muhammad Ibrahim (2026). *S&P 500 Historical Data & 2026 ML Forecasts*. User-supplied Yahoo Finance / yfinance `^GSPC` snapshot, January 2, 1980–September 25, 2026; curated September 27, 2026. Add the final Kaggle URL and dataset version when published.

**Data rights:** yfinance's software license does not license Yahoo market data. Its documentation refers users to Yahoo terms and personal-use restrictions. Redistribution permission is not established by this package. Select a Kaggle license only after confirming the applicable rights; no CC0 claim is made here.

**Reproducibility:** preserve the CSV snapshot, source/output hashes, seed, versions, splits and exported metrics. A later Yahoo refresh can revise history. The repository contains the unmodified input and deterministic cleaning script. This notebook does not make network requests.
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'authors':[{'name':'Muhammad Ibrahim Shahrukh'}]})
dest=ROOT/'notebooks/sp500_eda_ml_forecast_2026.ipynb'
nbf.write(nb,dest)
print(f'Wrote {dest}: {len(cells)} cells; {sum(c.cell_type=="code" for c in cells)} code cells')
