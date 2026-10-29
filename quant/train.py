"""Learn rolling cross-sectional factor payoffs using only matured delay-1 labels.

Hypothesis: learned momentum, reversal, volatility, liquidity and size payoffs
improve a fixed factor mix. Beta exposure and trading costs are rival explanations;
failure to improve the fixed baseline after costs falsifies the research idea.
Inputs are completed daily bars; signal s first earns return s+2. Fit only those
labels available at close t. A 252-session window needs 126 valid daily samples.
Daily equal-weighted cross-sectional sufficient statistics fit six coefficients
with fixed ridge 0.01 on the mean Gram matrix. Twenty-session signal smoothing
limits turnover. This source owns all model state; no cache survives a call.
"""

from quantlab.alpha.ops import np, pd

SETTINGS = {
    "book": "long_only", "delay": 1, "universe": "TOP500",
    "neutralization": "market", "truncation": 0.05, "decay": 0,
    "start": "2015-01-01", "end": "2025-09-11", "cost_bps": 5.0,
    "tax": True, "booksize": 1.0,
}


def alpha(d) -> pd.DataFrame:
    """Return causal scores aligned exactly with the supplied daily panel."""
    returns = d.returns
    listed = d.close.notna() & d.codes.str.fullmatch(r"\d{5}0")
    liquidity = d.turnover.rolling(20, min_periods=10).mean().where(listed)
    universe = liquidity.rank(axis=1, ascending=False, method="first") <= 500
    active = universe & (d.volume > 0)
    raw = (
        -returns.rolling(5).sum().shift(2),
        returns.rolling(21).sum(),
        returns.rolling(126).sum().shift(21),
        -returns.rolling(60).std(),
        d.turnover.rolling(5).mean() / d.turnover.rolling(60).mean().where(lambda x: x > 0),
        -np.log(d.cap.where(d.cap > 0)),
    )
    features = []
    for values in raw:
        ranked = values.where(universe & np.isfinite(values)).rank(axis=1, pct=True)
        features.append(ranked.sub(ranked.mean(axis=1), axis=0).to_numpy())
    del raw, ranked, values
    eligible = active.to_numpy() & np.logical_and.reduce([np.isfinite(x) for x in features])
    realized = returns.to_numpy()
    bars = listed.to_numpy()
    output = np.zeros(returns.shape)
    grams = np.zeros((252, 6, 6))
    crosses = np.zeros((252, 6))
    valid_days = np.zeros(252, dtype=bool)
    gram_sum, cross_sum = np.zeros((6, 6)), np.zeros(6)
    count = 0
    for t in range(len(returns)):
        slot = t % 252
        gram_sum -= grams[slot]
        cross_sum -= crosses[slot]
        count -= int(valid_days[slot])
        grams[slot].fill(0)
        crosses[slot].fill(0)
        valid_days[slot] = False
        s = t - 2
        if s >= 0:
            usable = eligible[s] & bars[t] & np.isfinite(realized[t])
            if usable.sum() >= 7:
                x = np.column_stack([feature[s, usable] for feature in features])
                y = realized[t, usable].astype(float)
                y = y - y.mean()
                grams[slot] = (x.T @ x) / len(y)
                crosses[slot] = (x.T @ y) / len(y)
                valid_days[slot] = True
                gram_sum += grams[slot]
                cross_sum += crosses[slot]
                count += 1
        if count >= 126:
            coefficients = np.linalg.solve(gram_sum / count + 0.01 * np.eye(6), cross_sum / count)
            usable = eligible[t]
            x = np.column_stack([feature[t, usable] for feature in features])
            output[t, usable] = x @ coefficients
    scores = pd.DataFrame(output, index=returns.index, columns=returns.columns)
    return scores.rolling(20, min_periods=1).mean().where(eligible, 0.0)
