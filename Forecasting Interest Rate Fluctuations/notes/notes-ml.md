On différencie pour que les données soient "stationary".
Pourquoi est-ce important?

Stationarity is important because non-stationary time series data can lead to unreliable and spurious regression results. Specifically:

Spurious Relationships: Non-stationary data may exhibit trends or patterns over time that can cause two unrelated variables to appear correlated.
Invalid Statistical Inferences: Standard statistical tests assume stationarity. Using non-stationary data violates these assumptions, leading to invalid confidence intervals and hypothesis tests.
Predictive Stability: Stationary data ensures that the underlying properties of the time series do not change over time, making models built on past data more reliable for future predictions.
Differencing the data to achieve stationarity stabilizes the mean and variance, allowing for valid regression analysis and meaningful interpretation of the results.



# Références
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4399406#:~:text=Yes!,communicate%20on%20monetary%20policy%20decisions.





# Cross-validation for time series
https://robjhyndman.com/hyndsight/tscv/




What exogenous and endogenous variables could I pass to my VAR model if I want to forecast yield for 10-year US treasury bonds?
ChatGPT

Possible endogenous variables usually include those jointly determined with the 10-year yield—i.e., other interest rates or financial variables that move together in response to market conditions. Exogenous variables are typically macroeconomic or policy indicators that are assumed to affect yields but are not themselves driven (or only minimally driven) by day-to-day yield fluctuations. Below is a schematic breakdown:

Endogenous variables (go in the VAR system): Yields on other maturities (e.g., 2-year, 5-year, 30-year) Short-term interest rates (e.g., fed funds rate, 3-month T-bill) Measures of market activity (e.g., stock indexes, credit spreads, VIX) Exchange rates (if relevant to bond demand and capital flows) Inflation expectations or breakeven inflation rates

Exogenous variables (passed as exog to the VAR model): Macroeconomic announcements (e.g., GDP growth, unemployment rate, CPI releases) Monetary policy events (e.g., FOMC meeting dates, policy rate announcements) Fiscal policy variables (e.g., government spending or debt issuance data) Leading indicators or sentiment indexes (e.g., Conference Board Leading Economic Index) Global shocks or events you consider outside the system (e.g., oil price shocks if deemed exogenous)

In practice, you decide which variables are endogenous vs. exogenous based on whether you believe they are jointly determined with the 10-year yield or come from outside and affect the yield unidirectionally.