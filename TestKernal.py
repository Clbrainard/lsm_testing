import QuantLib as ql
import pandas as pd
import os
import itertools

def price_american_put_lsm(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    num_paths,
    steps_per_year,
    poly_degree,
    poly_type,
    seed = 42
) -> float:
    """
    Price an American put using QuantLib's MCAmericanEngine (LSM).

    Parameters
    ----------
    S             : Initial asset price
    K             : Strike price
    T             : Time to expiry in years
    r             : Risk-free rate
    sigma         : Volatility
    num_paths     : Number of simulated paths
    steps_per_year: Time steps per year
    poly_degree   : Polynomial degree for LSM regression
    poly_type     : Basis function family:
                      0 = Monomial
                      1 = Laguerre
                      2 = Hermite
                      3 = Hyperbolic
                      4 = Legendre
                      5 = Chebyshev
                      6 = Chebyshev2nd
    seed          : Random seed for reproducibility

    Returns
    -------
    float : American put price
    """
    day_count  = ql.Actual365Fixed()
    today      = ql.Date(1, 1, 2000)
    expiry     = today + int(round(T * 365))
    ql.Settings.instance().evaluationDate = today

    spot    = ql.QuoteHandle(ql.SimpleQuote(S))
    rate    = ql.YieldTermStructureHandle(ql.FlatForward(today, r,   day_count))
    div     = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
    vol     = ql.BlackVolTermStructureHandle(
                  ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count))

    process    = ql.BlackScholesMertonProcess(spot, div, rate, vol)
    time_steps = int(round(T * steps_per_year))

    option = ql.VanillaOption(
                 ql.PlainVanillaPayoff(ql.Option.Put, K),
                 ql.AmericanExercise(today, expiry))

    engine = ql.MCAmericanEngine(
        process,
        "pseudorandom",
        timeSteps            = time_steps,
        polynomOrder         = poly_degree,
        polynomType          = poly_type,
        seed                 = seed,
        requiredSamples      = num_paths,
        nCalibrationSamples  = 2048,   # increase this
    )
    option.setPricingEngine(engine)

    return option.NPV()

def RunTest(
        stock_price,
        r,
        strike,
        years_to_exp,
        Moneyness,
        vol,
        real_price,
        deg_schedule,
        basis_schedule,
        path_schedule,
        step_schedule,
        path,
        seed
        ):

    headers = [
        "Stock_Price",
        "Risk_Free_Rate",
        "Strike_Price",
        "Years_to_exp",
        "Moneyness",
        "Volatility",
        "Actual Price",
        "Pred_Price",
        "Error",
        "PercentError",
        "seed",
        "paths",
        "steps",
        "basis",
        "degree"
    ]

    results = pd.DataFrame(columns=headers)

    grid = itertools.product(deg_schedule, basis_schedule, path_schedule, step_schedule)

    # 2. Iterate and run
    for deg, basis, paths, steps in grid:
        print(f"Testing: deg={deg}, basis={basis}, paths={paths}, steps={steps}")
        try:
            pred = price_american_put_lsm(
                                            stock_price,
                                            strike,
                                            years_to_exp,
                                            r,
                                            vol,
                                            paths,
                                            steps,
                                            deg,
                                            basis,
                                            seed
                                        )
            error = abs(pred-real_price)
            percentError = (error/real_price)*100
            results.loc[len(results)] = [
                stock_price,
                r,
                strike,
                years_to_exp,
                Moneyness,
                vol,
                real_price,
                pred,
                error,
                percentError,
                seed,
                paths,
                steps,
                basis,
                deg
            ]

        except Exception as e:
            print(f"  ERROR with deg={deg}, basis={basis}: {e}")
            raise
    
    results.to_csv(path, index=False, mode='a', header=False)

def test1():
    seed = 42
    file_path = 'results.csv'

    deg_schedule = [2]
    basis_schedule = [0]
    path_schedule = [100000000]
    step_schedule = [365]

    testSet = pd.read_csv("TestSet.csv")

    for row in testSet.itertuples(index=False):
        RunTest(
            row.Stock_Price,
            row.Risk_Free_Rate,
            row.Strike,
            row.Years_to_exp,
            row.Moneyness,
            row.Volatility,
            row.Actual_Price,
            deg_schedule,
            basis_schedule,
            path_schedule,
            step_schedule,
            file_path,
            seed
        )

test1()
    
    