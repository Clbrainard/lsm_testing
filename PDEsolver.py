import QuantLib as ql


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


if __name__ == "__main__":
    price = american_put_pde(S0=100, 
                             r=0.05, 
                             K=95, v=0.2, T=1.0)
    print(f"American Put Price (PDE): {price:.6f}")


    #    {100.0, 100.0, 1.0, 0.05, 0.2, 6.089770},  // ATM
    #    {100.0, 105.0, 1.0, 0.05, 0.2, 8.739389},  // ITM mild  (K > So for put)
    #    {100.0, 110.0, 1.0, 0.05, 0.2, 11.971846},  // ITM deep  (K > So for put)
    #    {100.0,  95.0, 1.0, 0.05, 0.2, 4.012655}   // OTM       (K < So for put)


