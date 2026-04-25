"""
benchmark.py

For each N value, runs 100 tests for every option in the CSV.
Results are flushed to CSV after each N batch completes.

Install:
    pip install QuantLib pandas
"""

import multiprocessing as mp

import pandas as pd
import QuantLib as ql


# ---------------------------------------------------------------------------
# Config — adjust these
# ---------------------------------------------------------------------------

CSV_PATH    = "NTestSet.csv"
OUTPUT_PATH = "C1-results.csv"
P           = 10000
N_VALUES    = [5,50,500,5000,50000,100000]
N_TESTS     = 100
SEEDS       = list(range(100))
Q_DEFAULT   = 0.0

REFERENCE_DATE = ql.Date(1, 1, 2020)


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------

def price_american_put(P, N, T, v, r, q, S0, seed, K):
    day_counter = ql.Actual365Fixed()
    today = REFERENCE_DATE
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

    # MCAmericanEngine has two RNG streams — calibration and pricing.
    # Both must be explicitly seeded for fully deterministic results.
    option.setPricingEngine(ql.MCAmericanEngine(
        process,
        "PseudoRandom",
        timeSteps=N,
        requiredSamples=P,
        seed=seed,
        polynomOrder=2,
        antitheticVariate=True,
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

    n_options = len(df)
    print(f"{n_options} options in CSV")
    print(f"Jobs per N: {n_options} options x {N_TESTS} tests = {n_options * N_TESTS}")
    print(f"Total jobs: {n_options * N_TESTS * len(N_VALUES)}")
    print(f"Running on {mp.cpu_count()} cores...\n")

    write_header = True
    with mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        for N in N_VALUES:
            batch = [
                {
                    "P": P, "N": N,
                    "S0": row["S0"], "K": row["K"], "T": row["T"],
                    "r": row["r"],   "v": row["v"], "q": row["q"],
                    "actual_price": row["price"],
                    "seed": SEEDS[test_idx],
                }
                for _, row in df.iterrows()
                for test_idx in range(N_TESTS)
            ]

            results = pool.map(_worker, batch)
            out_df = pd.DataFrame(results)[["P", "N", "S0", "K", "T", "r", "v", "APE"]]
            out_df.to_csv(OUTPUT_PATH, mode="a", index=False, header=write_header)
            write_header = False
            print(f"N={N:>4} done — mean APE: {out_df['APE'].mean():.4f}%")

    print(f"\nAll results written to {OUTPUT_PATH}")