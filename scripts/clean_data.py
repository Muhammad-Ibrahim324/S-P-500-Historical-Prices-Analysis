"""Reproduce the clean OHLCV dataset from the supplied yfinance CSV."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

def clean(source, destination, report):
    raw = pd.read_csv(source)
    assert raw.columns.tolist() == ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
    meta = raw['Price'].isin(['Ticker', 'Date'])
    tickers = raw.loc[raw.Price.eq('Ticker'), raw.columns[1:]].values.ravel()
    assert len(tickers) and set(tickers) == {'^GSPC'}
    df = raw.loc[~meta].rename(columns={'Price':'date', 'Open':'open', 'High':'high',
        'Low':'low','Close':'close','Volume':'volume'}).copy()
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='raise')
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='raise')
    assert df.notna().all().all(), 'Missing values require explicit review.'
    assert not df.date.duplicated().any(), 'Duplicate dates require explicit review.'
    assert np.isfinite(df[['open','high','low','close','volume']]).all().all()
    assert (df[['open','high','low','close']] > 0).all().all()
    assert (df.high >= df[['open','low','close']].max(axis=1)).all()
    assert (df.low <= df[['open','high','close']].min(axis=1)).all()
    assert (df.volume >= 0).all() and (df.volume % 1 == 0).all()
    assert (df.date.dt.dayofweek < 5).all()
    prices = ['open','high','low','close']
    max_rounding = float((df[prices] - df[prices].round(2)).abs().max().max())
    df[prices] = df[prices].round(2)
    df['volume'] = df.volume.astype('int64')
    df['ticker'] = '^GSPC'
    df = df[['date','ticker','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, date_format='%Y-%m-%d', float_format='%.2f')
    audit = {'source_filename':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'clean_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'input_rows_excluding_column_header':len(raw),'metadata_rows_removed':int(meta.sum()),
        'output_rows':len(df),'date_start':str(df.date.min().date()),'date_end':str(df.date.max().date()),
        'missing_values':int(df.isna().sum().sum()),'duplicate_dates':int(df.date.duplicated().sum()),
        'zero_volume_dates':df.loc[df.volume.eq(0),'date'].dt.strftime('%Y-%m-%d').tolist(),
        'ohlc_max_rounding_change_points':max_rounding,'ohlc_decimal_places':2,
        'imputed_rows':0,'removed_price_observations':0,'adjustment_setting':'unknown in original export',
        'download_timestamp':'not supplied','prepared_date':'2026-09-27'}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(audit, indent=2)+'\n')
    return audit

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=root/'data/raw/sp500_yfinance_original.csv')
    p.add_argument('--output',type=Path,default=root/'data/processed/sp500_daily_clean.csv')
    p.add_argument('--report',type=Path,default=root/'reports/data_quality.json')
    a=p.parse_args()
    print(json.dumps(clean(a.source,a.output,a.report),indent=2))
