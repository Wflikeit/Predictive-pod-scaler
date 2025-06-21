import logging

logger = logging.getLogger(__name__)

from typing import Sequence

import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.domain.trendResult import TrendResult
from app.domain.trend_analyzer import TrendAnalyzer
from app.domain.ueSessionInfo import UeSessionInfo


class ARIMAAnalyzer(TrendAnalyzer):
    def __init__(
            self,
            period: int,
            minimal_reaction_time: int = 1,
    ):
        self.period = period
        self.minimal_reaction_time = minimal_reaction_time
        self.model = None

    def train(self, history: Sequence[UeSessionInfo]) -> None:
        y = np.array([x.session_count for x in history])
        try:
            self.model = auto_arima(
                y,
                seasonal=True,
                m=self.period * 2,
                trace=False,
                suppress_warnings=True,
                stepwise=True,
                information_criterion="aic",
                error_action="ignore"
            )
        except ValueError as e:
            logger.warning(f"seasonal auto_arima failed ({e}); retrying non-seasonal")
            self.model = auto_arima(
                y,
                seasonal=False,
                trace=False,
                suppress_warnings=True,
                stepwise=True,
                information_criterion="aic",
                error_action="ignore"
            )


    def evaluate(self, history: Sequence[UeSessionInfo],
                 part_of_period: float = 0.5) -> TrendResult:
        if self.model is None:
            raise RuntimeError("Model ARIMA is not trained")

        y = np.array([x.session_count for x in history])

        sarima = SARIMAX(
            y, order=self.model.order, seasonal_order=self.model.seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)

        h = max(1, int(round(self.period * part_of_period)))
        fc = sarima.get_forecast(steps=h)
        mean = fc.predicted_mean[-1]
        low, up = fc.conf_int(alpha=0.10)[-1]  # 90 % CI

        predicted = mean + 0.5 * (up - mean)

        predicted = max(predicted, y[-1])

        delta = predicted - y[-1]
        slope = delta / h

        return TrendResult(
            delta=delta,
            slope=slope,
            current_sessions=int(round(predicted))
        )

# if __name__ == "__main__":
#     def gen_data(periods, samples_per_period, deviation):
#         return ((np.sin(np.linspace(0, 2 * periods * np.pi, 2 * periods * samples_per_period, True)) + 1) * (
#                 1 + ((np.random.random(2 * periods * samples_per_period) - 0.5) * 2 * deviation)))
#
#     plt.style.use("seaborn-v0_8-whitegrid")
#     sns.set_context("talk")
#
#     # ─────────────────────────────────────────────────────────────
#     # 2. Przygotowanie danych
#     # ─────────────────────────────────────────────────────────────
#
#     fig, ax = plt.subplots()
#
#     # Podzielmy na train/test ostatnie 24 miesiące jako walidację
#     train1 = gen_data(10, 10, 0.1)
#     train2 = gen_data(10, 10, 0.3)
#     test = gen_data(10, 10, 0.3)
#
#     # ax.plot(data_set)
#     # plt.show()
#
#     # Szukanie najlepszego (p,d,q)(P,D,Q)m wg AIC
#     # Chcemy uchwycić sezonowość – m=12 (miesięczna)
#     model_auto = auto_arima(
#         train1,
#         seasonal=True, m=2 * 10,
#         trace=True,  # loguj próby
#         suppress_warnings=True,
#         stepwise=True,
#         information_criterion="aic",
#         error_action="ignore"  # pomiń nie-podażne konfiguracje
#     )
#
#     print(model_auto.summary())
#
#     order, sorder = model_auto.order, model_auto.seasonal_order
#     print("Wybrane parametry:", order, sorder)
#
#     sarima = SARIMAX(
#         train2,
#         order=order,
#         seasonal_order=sorder,
#         enforce_stationarity=False,
#         enforce_invertibility=False
#     ).fit()
#
#     print(sarima.summary())
#
#     n_periods = len(test)
#     forecast = sarima.get_forecast(steps=n_periods)
#     pred_mean = forecast.predicted_mean
#     pred_ci = forecast.conf_int()
#
#     mape = mean_absolute_percentage_error(test, pred_mean) * 100
#     print(f"MAPE: {mape:.2f}%")
#
#     # ─ Plot ─
#     plt.figure(figsize=(14, 5))
#     plt.plot(train1, label="data for training")
#     plt.plot(train2, label="data for evaluation", color="gray")
#     plt.plot(test, label="data to predict", color="black")
#     # plt.plot(pred_mean, label="forecast", color="royalblue")
#     plt.fill_between(
#         np.arange(0, pred_ci.shape[0], step=1),
#         pred_ci[:, 0],
#         pred_ci[:, 1],
#         color="royalblue",
#         alpha=0.40,
#         label="95% CI"
#     )
#     plt.title(f"SARIMA forecast, MAPE={mape:.2f}%")
#     plt.legend()
#     plt.show()
#
#     import joblib
#
#     joblib.dump(sarima, "sarima_airpassengers.joblib")
#     # ...
#     loaded = joblib.load("sarima_airpassengers.joblib")
#     print("Predict from loaded:", loaded.forecast(3))
