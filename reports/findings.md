
## Research findings

- **Data:** 11,779 observations from 1980-01-02 through 2026-09-25; last close **7,743.41 index points**.
- **Observed context:** 2026 YTD price return **13.12%**; latest 21-session annualized volatility **10.80%**. These are snapshot statistics, not current-news explanations.
- **Selection:** **Ridge_1000** achieved mean validation log-RMSE **4.552%**, versus **5.136%** for no-change. Best ML candidate: **Ridge_1000**.
- **Untouched test:** the winner beat no-change with **29.97% RMSE skill**. MAE **101.02 points**, RMSE **125.28 points**, MAPE **1.32%**. Validation success need not persist.
- **Year-end central scenario:** **7,945.51 points** on **2026-12-31**, **+2.61%** versus the last close.
- **Year-end 80% bootstrap range:** **7,114.58–8,828.03 points**; **95% range:** **6,654.29–9,316.18**. Test-path coverage: **98.5% / 100.0%**. These are simulations, not calibrated success probabilities.

## Interpretation and limitations

The central path extrapolates historical behavior. Policy, earnings, inflation, interest rates and geopolitical developments are not model inputs; this notebook cannot attribute its forecast to those drivers. Positive momentum does not ensure gains, and a downward curve does not predict a specific crash. The no-change comparison reveals how much complexity adds.

There are six validation origins and one test path. Forecast windows are distinct, but training windows overlap and markets are dependent. No statistically significant superiority is claimed. A different window or regime can change rankings. Direct multi-horizon predictions may be jagged because adjacent horizons have different targets. Scenario bands omit parameter uncertainty and may understate risk.

## Next experiments

1. Add dated, revision-aware economic and earnings inputs, using only values available at each origin.
2. Evaluate more historical origins and nested temporal tuning before changing a production model.
3. Compare volatility models and interval calibration across many independent forecast periods.
4. Study ETF tracking, transaction costs and turnover separately. Low forecast error is not proof of a profitable strategy.

**Educational research only.** This is a price index, not a dividend-reinvested portfolio or directly investable security. No ranking, accuracy, return or future index level is guaranteed.
