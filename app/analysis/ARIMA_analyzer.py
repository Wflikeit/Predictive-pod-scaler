# ─────────────────────────────────────────────────────────────
# 0. Instalacja najnowszych wersji (tylko jeśli to nowy runtime)
# ─────────────────────────────────────────────────────────────
# !pip install -U pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima

# (W Colab dodaj przed komórką "!" – lokalnie w terminalu bez tego.)

# ─────────────────────────────────────────────────────────────
# 1. Importy
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# ─────────────────────────────────────────────────────────────
# 2. Przygotowanie danych
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots()

# Podzielmy na train/test ostatnie 24 miesiące jako walidację
data_set = (np.sin(np.linspace(0,20*np.pi,20*10, True))+1) * (1+ ((np.random.random(20*10)-0.5)*0.3))
train = data_set[:-10*10] #df.iloc[:-24]
test  = data_set[-10*10:] #df.iloc[-24:]

ax.plot(data_set)
plt.show()

# Szukanie najlepszego (p,d,q)(P,D,Q)m wg AIC
# Chcemy uchwycić sezonowość – m=12 (miesięczna)
model_auto = auto_arima(
    train,
    seasonal=True, m=2*10,
    trace=True,          # loguj próby
    suppress_warnings=True,
    stepwise=True,
    information_criterion="aic",
    error_action="ignore"  # pomiń nie-podażne konfiguracje
)

print(model_auto.summary())

order, sorder = model_auto.order, model_auto.seasonal_order
print("Wybrane parametry:", order, sorder)

sarima = SARIMAX(
    train,
    order=order,
    seasonal_order=sorder,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()

print(sarima.summary())

# Prognozujemy dokładnie tyle okresów, ile mamy w teście
n_periods = len(test)
forecast = sarima.get_forecast(steps=n_periods)
pred_mean = forecast.predicted_mean
pred_ci   = forecast.conf_int()

mape = mean_absolute_percentage_error(test, pred_mean) * 100
print(f"MAPE: {mape:.2f}%")

# ─ Plot ─
plt.figure(figsize=(14,5))
plt.plot(train, label="train")
plt.plot(test,  label="ground truth (test)", color="black")
plt.plot(pred_mean, label="forecast", color="royalblue")
# plt.fill_between(
#     pred_ci.index,
#     pred_ci.iloc[:,0],
#     pred_ci.iloc[:,1],
#     color="royalblue",
#     alpha=0.20,
#     label="95% CI"
# )
plt.title(f"SARIMA forecast, MAPE={mape:.2f}%")
plt.legend()
plt.show()

import joblib

joblib.dump(sarima, "sarima_airpassengers.joblib")
# ...
loaded = joblib.load("sarima_airpassengers.joblib")
print("Predict from loaded:", loaded.forecast(3))
