# ─────────────────────────────────────────────────────────────
# 0. Instalacja najnowszych wersji (tylko jeśli to nowy runtime)
# ─────────────────────────────────────────────────────────────
# !pip install -U pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima
from typing import Sequence

# (W Colab dodaj przed komórką "!" – lokalnie w terminalu bez tego.)

# ─────────────────────────────────────────────────────────────
# 1. Importy
# ─────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error

from app.domain.trendResult import TrendResult
from app.domain.ueSessionInfo import UeSessionInfo
from app.domain.trend_analyzer import TrendAnalyzer


class ARIMAAnalyzer(TrendAnalyzer):
    def __init__(self, period, minimal_reaction_time=1, sample_data: Sequence[UeSessionInfo] | None = None):
        self.period = period
        self.minimal_reaction_time = minimal_reaction_time
        self.model = None
        self.train(sample_data)

    def train(self, history: Sequence[UeSessionInfo] | None) -> None:
        history = np.array([x['session_count'] for x in history])
        self.model = auto_arima(
            history if len(history) > self.period * 2 else
            np.concatenate(
                np.random.randint(
                    low=history.min(),
                    high=history.max(),
                    size=(self.period * 2 - len(history))
                ),
                history
            ),
            seasonal=True, m=self.period * 2,
            trace=True,
            suppress_warnings=True,
            stepwise=True,
            information_criterion="aic",
            error_action="ignore"
        )

    def evaluate(self, history: Sequence[UeSessionInfo]) -> TrendResult:
        forecast = SARIMAX(
            np.array([x['session_count'] for x in history]),
            order=self.model.order,
            seasonal_order=self.model.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit().get_forecast(steps=self.period * 2)

        prediction = {
            'min': forecast.conf_int(),
            'mean': forecast.predicted_mean,
            'max': forecast.conf_int()
        }

        return TrendResult(delta=max(prediction['max']) - history[-1].session_count,
                           slope=(max(prediction['max']) - history[-1].session_count) /
                                 np.where(np.isclose(prediction['max'], max(prediction['max'])))[0][0],
                           current_sessions=prediction['max'])


if __name__ == "__main__":
    def gen_data(periods, samples_per_period, deviation):
        return ((np.sin(np.linspace(0, 2 * periods * np.pi, 2 * periods * samples_per_period, True)) + 1) * (
                1 + ((np.random.random(2 * periods * samples_per_period) - 0.5) * 2 * deviation)))


    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk")

    # ─────────────────────────────────────────────────────────────
    # 2. Przygotowanie danych
    # ─────────────────────────────────────────────────────────────

    fig, ax = plt.subplots()

    # Podzielmy na train/test ostatnie 24 miesiące jako walidację
    train1 = gen_data(10, 10, 0.1)
    train2 = gen_data(10, 10, 0.3)[:-5] * 2
    test = gen_data(10, 10, 0.3)[:-5] * 2

    # ax.plot(data_set)
    # plt.show()

    # Szukanie najlepszego (p,d,q)(P,D,Q)m wg AIC
    # Chcemy uchwycić sezonowość – m=12 (miesięczna)
    model_auto = auto_arima(
        train1,
        seasonal=True, m=2 * 10,
        trace=True,  # loguj próby
        suppress_warnings=True,
        stepwise=True,
        information_criterion="aic",
        error_action="ignore"  # pomiń nie-podażne konfiguracje
    )

    print(model_auto.summary())

    order, sorder = model_auto.order, model_auto.seasonal_order
    print("Wybrane parametry:", order, sorder)

    sarima = SARIMAX(
        train2,
        order=order,
        seasonal_order=sorder,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()

    print(sarima.summary())

    n_periods = len(test)
    forecast = sarima.get_forecast(steps=n_periods)
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int()

    mape = mean_absolute_percentage_error(test, pred_mean) * 100
    print(f"MAPE: {mape:.2f}%")

    # ─ Plot ─
    plt.figure(figsize=(14, 5))
    plt.plot(train1, label="data for training")
    plt.plot(train2, label="data for evaluation", color="gray")
    plt.plot(test, label="data to predict", color="black")
    # plt.plot(pred_mean, label="forecast", color="royalblue")
    plt.fill_between(
        np.arange(0, pred_ci.shape[0], step=1),
        pred_ci[:, 0],
        pred_ci[:, 1],
        color="royalblue",
        alpha=0.40,
        label="95% CI"
    )
    plt.title(f"SARIMA forecast, MAPE={mape:.2f}%")
    plt.legend()
    plt.show()

    import joblib

    joblib.dump(sarima, "sarima_airpassengers.joblib")
    # ...
    loaded = joblib.load("sarima_airpassengers.joblib")
    print("Predict from loaded:", loaded.forecast(3))
