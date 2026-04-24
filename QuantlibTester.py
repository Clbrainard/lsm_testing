"""
benchmark.py

For a given P and list of N values, runs 100 MC American put pricing tests
per N value against a test set CSV, computes APE, and writes results to CSV.

Install:
    pip install QuantLib pandas
"""

import multiprocessing as mp
from itertools import cycle

import pandas as pd
import QuantLib as ql


# ---------------------------------------------------------------------------
# Config — adjust these
# ---------------------------------------------------------------------------

CSV_PATH    = "NTestSet.csv"
OUTPUT_PATH = "Conj1_results.csv"
P           = 10000         # number of MC paths
N_VALUES    = [10, 100, 1000, 10000, 100000]  # list of time step values to test
N_TESTS     = 100             # number of tests per N value
SEEDS       = list(range(N_TESTS)) # seed pool — cycles across tests [0,1,...,9,0,1,...]
Q_DEFAULT   = 0.0             # dividend yield if not present in CSV


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------

def price_american_put(P, N, T, v, r, q, S0, seed, K):
    day_counter = ql.Actual365Fixed()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    expiry = today + int(round(T * 365))

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S0)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_counter)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_counter)),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), v, day_counter)),
    )

    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Put, K),
        ql.AmericanExercise(today, expiry),
    )
    option.setPricingEngine(ql.MCAmericanEngine(
        process,
        "PseudoRandom",
        timeSteps=N,
        requiredSamples=P,
        seed=seed,
        polynomOrder=4,
        antitheticVariate=True,
        nCalibrationSamples=P // 4,
    ))
    return option.NPV()


def _worker(kwargs):
    result = kwargs.copy()
    predicted = price_american_put(
        P=kwargs["P"], N=kwargs["N"], T=kwargs["T"], v=kwargs["v"],
        r=kwargs["r"], q=kwargs["q"], S0=kwargs["S0"],
        seed=kwargs["seed"], K=kwargs["K"],
    )
    result["APE"] = abs(predicted - kwargs["actual_price"]) / kwargs["actual_price"] * 100
    return result


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    if "q" not in df.columns:
        df["q"] = Q_DEFAULT

    seed_cycle = cycle(SEEDS)
    all_jobs = []

    for N in N_VALUES:
        rows = [df.iloc[i % len(df)] for i in range(N_TESTS)]
        for row in rows:
            all_jobs.append({
                "P": P, "N": N,
                "S0": row["S0"], "K": row["K"], "T": row["T"],
                "r": row["r"],   "v": row["v"], "q": row["q"],
                "actual_price": row["price"],
                "seed": next(seed_cycle),
            })

    print(f"Pricing {len(all_jobs)} options across {len(N_VALUES)} N values "
          f"({N_TESTS} tests each) using {mp.cpu_count()} cores...")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(_worker, all_jobs)

    out_df = pd.DataFrame(results)[["P", "N", "S0", "K", "T", "r", "v", "APE"]]
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Results written to {OUTPUT_PATH}")

