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
    time_steps = 500
    grid_points = 500
    engine = ql.FdBlackScholesVanillaEngine(bsm_process, time_steps, grid_points)
    option.setPricingEngine(engine)

    return option.NPV()


def price_grid(S0, r_list, v_list, T_list, K_list):
    """
    Compute American put prices for all (T, K, r, v) combinations.

    Parameters
    ----------
    S0     : float        - current stock price (fixed)
    r_list : list[float]  - risk-free rates to sweep
    v_list : list[float]  - volatilities to sweep
    T_list : list[float]  - maturities to sweep
    K_list : list[float]  - strikes to sweep

    Yields
    ------
    dict : with keys [S0, K, T, r, v, price]
    """

    for T, K, r, v in itertools.product(T_list, K_list, r_list, v_list):
        p = american_put_pde(S0=S0, r=r, K=K, v=v, T=T)
        yield {"S0": S0, "K": K, "T": T, "r": r, "v": v, "price": p}


if __name__ == "__main__":

    K = [72, 76, 80, 84, 88, 92, 96, 98, 99, 101, 102, 104, 108, 112, 116, 120, 124, 128, 132]  # 17 values
    #[ 70  74  78  82  86  90  94  97 100 103 106 110 114 118 122 126 130]

    r = [0.01, 0.025, 0.04, 0.055, 0.07, 0.10] 

    t = [0.08, 0.11, 0.16, 0.23, 0.33, 0.47, 0.68, 0.97, 1.40, 2.00]  

    v = [0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.80]

    filename = "ML_PROJ/trainingSet.csv"
    batch_size = 1000
    first_batch = True
    rows = []

    for row in price_grid(100, r, v, t, K):
        rows.append(row)
        if len(rows) >= batch_size:
            df = pd.DataFrame(rows, columns=["S0", "K", "T", "r", "v", "price"])
            df.to_csv(filename, mode='a', header=first_batch, index=False)
            first_batch = False
            rows = []

    # Save remaining rows
    if rows:
        df = pd.DataFrame(rows, columns=["S0", "K", "T", "r", "v", "price"])
        df.to_csv(filename, mode='a', header=first_batch, index=False)
