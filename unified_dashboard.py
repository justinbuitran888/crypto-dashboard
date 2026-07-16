"""
Unified Signal Dashboard -- scans every setup studied in this project:
    SHORT signals: Bounce Short, First Red Day, Double Top, Head-and-Shoulders Top
    LONG signals:  Double Bottom, Head-and-Shoulders Bottom, Cup & Handle

Generates ONE modern, self-contained HTML dashboard (dashboard.html) with
live entry/stop/target for every signal firing today, plus the REAL win
rate / avg R stats from backtesting each setup on your own crypto data
(not copied from any book -- computed fresh, same rigor used throughout
this project).

HOW TO RUN
    pip install pandas numpy ccxt requests
    python unified_dashboard.py

WHERE TO SEE IT
    Open the generated dashboard.html on your Mac, or AirDrop / local-WiFi
    host it to view on your phone (same instructions as before).

UPDATE HISTORICAL_STATS BELOW whenever you re-run the individual backtest
scripts (backtest_bounce_short_t