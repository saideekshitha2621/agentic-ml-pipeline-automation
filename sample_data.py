"""Generates a synthetic customer dataset for demoing/testing the pipeline."""
import numpy as np
import pandas as pd


def make_sample_dataset(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    segment = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25])
    age = np.where(
        segment == "A", rng.normal(25, 4, n),
        np.where(segment == "B", rng.normal(45, 6, n), rng.normal(60, 5, n)),
    ).clip(18, 85)
    income = np.where(
        segment == "A", rng.normal(35000, 8000, n),
        np.where(segment == "B", rng.normal(75000, 15000, n), rng.normal(95000, 20000, n)),
    ).clip(15000, None)
    spend_score = np.where(
        segment == "A", rng.normal(70, 15, n),
        np.where(segment == "B", rng.normal(40, 12, n), rng.normal(55, 18, n)),
    ).clip(0, 100)
    tenure_years = rng.exponential(3, n).clip(0, 25)
    region = rng.choice(["North", "South", "East", "West"], size=n)
    membership = rng.choice(["Basic", "Silver", "Gold", "Platinum"], size=n, p=[0.4, 0.3, 0.2, 0.1])

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{i:05d}" for i in range(n)],
            "age": age.round(0),
            "annual_income": income.round(2),
            "spend_score": spend_score.round(2),
            "tenure_years": tenure_years.round(2),
            "region": region,
            "membership_tier": membership,
        }
    )

    # inject missing values and duplicates so preprocessing has real work to do
    for col in ["age", "annual_income", "spend_score", "tenure_years"]:
        mask = rng.random(n) < 0.04
        df.loc[mask, col] = np.nan
    mask = rng.random(n) < 0.03
    df.loc[mask, "region"] = np.nan

    dupes = df.sample(frac=0.03, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


if __name__ == "__main__":
    df = make_sample_dataset()
    df.to_csv("sample_customers.csv", index=False)
    print(f"Wrote sample_customers.csv with {len(df)} rows")
