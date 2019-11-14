import pandas as np
import numpy as np
import os
from preparing_univariate_weather_ts_analysis import load_dataset

FILENAME = "./data/daily-min-temperatures.csv"

df = load_dataset(FILENAME)


def fit_model(endog_data, exog_data=None, model_type=None):

    """ AR method models the next step in the sequence as a linear function of the observations at prior time steps.

    MA) method models the next step in the sequence as a linear function of the residual errors from a mean process at prior time steps.
    The method is suitable for univariate time series without trend and seasonal components. 

    ARMA combines the two. Suitable for univariate TS without trend or seasonality components.

    ARIMA combines both Autoregression (AR) and Moving Average (MA) models as well as a differencing 
    pre-processing step of the sequence to make the sequence stationary, called integration (I). Good for data sets with trend but without seasonality. 

    SARIMA combines the ARIMA model with the ability to perform the same autoregression, differencing, and moving average modeling at the seasonal level.
    The notation for the model involves specifying the order for the AR(p), I(d), and MA(q) models as parameters to an ARIMA function and AR(P), I(D), MA(Q) 
    and m parameters at the seasonal level. e.g. SARIMA(p, d, q)(P, D, Q)m where “m” is the number of time steps in each season (the seasonal period). 
    Suitable for univariate with trend or seasonality.

    SARIMAX includes the modeling of exogenous variables. Exogenous variables are also called covariates and can be thought of as parallel input sequences 
    that have observations at the same time steps as the original series.  The original time series can be referred to as endogenous data to contrast it 
    from exogenous sequences. The observations for exogenous variables are included in the model directly at each time step and are not modeled in the 
    same way as the primary endogenous sequence (e.g. as an AR, MA, etc. process). The SARIMAX method can be used to model with the inclusion of exogenous 
    variables in the other subvariants of linear modeling (ARX, MAX, ARMAX, ARMIAX). The method is suitable for univariate time series without or 
    without seasonal componenets and exogenous variables. 
    
    VARMA is ARMA but as a multivariate time series. VARMA can also be used to develop VMA models. Suitable for multi-TS without trend and without seasonality. 
    VARMAX is VARMA with exogenous regressors/variables. Exogenous variables are also called covariates and can be thought of as parallel input sequences that 
    have observations at the same time steps as the original series. Again, VARMAX is suitable for multivariate TS without seasonality/trend and with exogenous variables.

    The Simple Exponential Smoothing (SES) method models the next time step as an exponentially weighted linear function of observations at prior time steps. 
    Suitable for univariate time series without trend and seasonal components.
    """
    from statsmodels.tsa.arima_model import ARIMA

    p, d, q = 0, 0, 0

    if model_type == "AR":
        p = 1
        model = ARIMA(endog_data, order=(p, d, q))
        model_fit = model.fit(disp=False)

    if model_type == "MA":
        q = 1
        model = ARIMA(endog_data, order=(p, d, q))
        model_fit = model.fit(disp=False)

    if model_type == "ARMA":
        p, q = 1, 1
        model = ARIMA(endog_data, order=(p, d, q))
        model_fit = model.fit(disp=False)

    if model_type == "ARIMA":
        p, d, q = 1, 1, 1
        model = ARIMA(endog_data, order=(p, d, q))
        model_fit = model.fit(disp=False)

    if model_type == "SARIMA":
        p, d, q, P, D, Q = 1, 1, 1, 1, 1, 1
        m = 3  # for a quarterly seasonal period
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            endog_data,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        model_fit = model.fit(disp=False)

    if model_type in ["VAR", "VMA", "VARMA", "VARMAX"]:
        """ note that this is a multivariate time series so your data is a 2- or more-dimensional vector """
        from statsmodels.tsa.statespace.varmax import VARMAX

        p = 1
        q = 1
        if model == "VARMAX":
            model = VARMAX(endog_data, exog=exog_data, order=(p, q))
        else:
            model = VARMAX(endog_data, order=(p, q))

        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()

    if model_type == "SES":
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        model = SimpleExpSmoothing(endog_data)
        model_fit = model.fit()
        yhat = model_fit.predict(len(endog_data), len(endog_data))

    if model_type not in ["VAR", "VMA", "VARMA", "VARMAX", "SES"]:
        yhat = model_fit.predict(len(endog_data), len(endog_data), typ="levels")

    # return prediction for print
    return yhat


print(
    "The next value in the temperature time series is: {}".format(
        fit_model(df["temperature"], model_type="SARIMA")
    )
)
print("\n")
print("Temperature at the last available index: {}".format(df.iloc[3639, :]))
