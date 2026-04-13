import QuantLib as ql
import pandas as pd
import itertools


def american_put_pde(S0, r, K, v, T):
    """
    Solve for the American put option price using QuantLib's finite difference PDE solver.

    Parameters
    ----------
    S0 : float  - current stock price
    r  : float  - continuous risk-free rate (e.g. 0.05)
    K  : float  - strike price
    v  : float  - volatility (e.g. 0.2)
    T  : float  - time to maturity in years

    Returns
    -------
    float : American put option price
    """
    # Calendar / day-count setup
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()

    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    maturity_date = today + ql.Period(int(T * 365), ql.Days)

    # Option payoff and exercise
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    exercise = ql.AmericanExercise(today, maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    # Market data handles
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, day_count)
    )
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, v, day_count)
    )

    # Black-Scholes-Merton process
    bsm_process = ql.BlackScholesMertonProcess(
        spot_handle, dividend_ts, flat_ts, vol_ts
    )

    # Finite difference engine (Crank-Nicolson via FdBlackScholesVanillaEngine)
    time_steps = 1000
    grid_points = 1000
    engine = ql.FdBlackScholesVanillaEngine(bsm_process, time_steps, grid_points)
    option.setPricingEngine(engine)

    return option.NPV()


def price_grid(S0, r, v, T_list, K_list):
    """
    Compute American put prices for all (T, K) combinations.

    Parameters
    ----------
    S0     : float        - current stock price (fixed)
    r      : float        - risk-free rate (fixed)
    v      : float        - volatility (fixed)
    T_list : list[float]  - maturities to sweep
    K_list : list[float]  - strikes to sweep

    Returns
    -------
    pd.DataFrame with columns [T, K, price]
    """

    rows = []
    for T, K in itertools.product(T_list, K_list):
        p = american_put_pde(S0=S0, r=r, K=K, v=v, T=T)
        rows.append({"S0": S0, "K": K, "T": T, "r": r, "v": v, "price": p})

    return pd.DataFrame(rows, columns=["S0", "K", "T", "r", "v", "price"])


if __name__ == "__main__":

    T_sched = [
        0.01125
    ]
    K_sched = [110,105,100,95]

    df = price_grid(100,0.05,0.2,T_sched,K_sched)

    df.to_csv("DiscretizationTestSet2.csv", index=False)
