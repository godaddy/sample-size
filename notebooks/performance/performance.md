# Improving the performance of FDR adjustment

[PR #35](https://github.com/godaddy/sample-size/pull/35) introduced support for
multiple metrics and cohorts accounting for FDR adjustment. However, we found
that it is not adequately performant. [PR #42](https://github.com/godaddy/sample-size/pull/42)
improved performance by refactoring code, but there are still parameters that
were chosen arbitrarily that impact performance. The goal of this document is to
choose good values for those parameters, specifically for `REPLICATIONS`,
`EPSILON`, and `MAX_RECURSION_DEPTH`.




```python
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
from itertools import product
from plotly import express as px

# Launch jupyter from the sample_size repo dreictory and set PYTHONPATH=`pwd`
from sample_size.sample_size_calculator import SampleSizeCalculator
```

```python
class MockMetricGenerator:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def generate_metrics(self, k: int = None, counts: int = None):
        assert k or counts

        if k:
            return [self.generate_metric() for _ in range(k)]

        metrics = []
        for kind in counts:
            metrics.append([self.generate_metric(kind) for _ in range(counts[kind])])

        return metrics

    def generate_metric(self, kind: str = None, **kwargs):
        TYPE_MAP = {
            "boolean": self._generate_boolean_metric,
            "numeric": self._generate_numeric_metric,
            "ratio": self._generate_ratio_metric,
        }

        kind = kind or self.rng.choice(["boolean", "numeric", "ratio"])

        return TYPE_MAP[kind](**kwargs)

    def _generate_boolean_metric(self, prob: float = None, mde: float = None) -> None:
        p = self.rng.random()

        return {
            "metric_type": "boolean",
            "metric_metadata": {"probability": prob or p, "mde": mde or p * 0.1},
        }

    def _generate_numeric_metric(
        self, variance: float = None, mde: float = None, σ2: float = 5000
    ) -> None:
        variance = variance or 5000 * self.rng.random()
        return {
            "metric_type": "numeric",
            "metric_metadata": {
                "variance": variance,
                "mde": mde or (self.rng.random()) * np.sqrt(variance),
            },
        }

    def _generate_ratio_metric(
        self,
        μ: float = 500,
        σ: float = 23,
    ) -> None:
        σ_num = σ * self.rng.uniform(0.1, 1)
        σ_denom = σ * self.rng.uniform(0.1, 1)

        num = μ * self.rng.uniform(0.1, 1)
        denom = μ * self.rng.uniform(0.1, 1)

        num_var = σ_num**2
        denom_var = σ_denom**2

        cov = np.sqrt(num_var * denom_var) * self.rng.uniform(0, 0.1)

        mde = self.rng.uniform(0, 0.01) * ((num) / (denom))

        return {
            "metric_type": "ratio",
            "metric_metadata": {
                "numerator_mean": num,
                "numerator_variance": num_var,
                "denominator_mean": denom,
                "denominator_variance": denom_var,
                "covariance": cov,
                "mde": mde,
            },
        }
```

Here's an example output from the code above

```python
metrics = MockMetricGenerator(6).generate_metrics(10)
pd.DataFrame(
    [{"metric_type": m["metric_type"], **m["metric_metadata"]} for m in metrics]
).round(3)
```

## Parameter Tuning

Historically the power calculator has used these values:

```python
REPLICATIONS = 500
EPSILON = 0.05
MAX_DEPTH = 20
```

These were selected fairly arbitrarily, and we can likely improve on them. They
also all depend on each other. For `EPSILON`, it's probably the case that if
we're going through the trouble of estimating sample sizes at all we would
probably prefer a better precision than $\pm 0.05$. However, as we increase the
precision we increase our time complexity by order $\mathcal{O}(log(\epsilon))$.
That also opens the question of `REPLICATIONS`, which determines the precision
of our empirical estimates by $\mathcal{O}(\frac{1}{\sqrt{n}})$ and (assuming
our estimates are good enough) increases the time complexity of the entire
search by $\mathcal{O}(n)$. `REPLICATIONS` places an essential limit on our
precision, regardless of what we set `EPSILON` to, and these values determine
whether we're likely to encounter `MAX_DEPTH` before we converge on an estimate.

A reasonable strategy might be to fix `EPSILON`, choose `REPLICATIONS` such that
our estimates are within an acceptable error threshold, and then choose
`MAX_DEPTH` based on application-specific needs (i.e. how much time we have to
wait).

Here are some simulation results where we try to quantify the variability of
empirical estimates of power as a function of `REPLICATIONS`, which we can use
to select a trade-off with time complexity in order to then choose `EPSILON`.
These empirical estimates are called for _each_ recursive call.

```python
def get_filename(*args, **kwargs):
    if not (args or kwargs):
        return f"./simulations/default.feather"

    return f"./simulations/{args}-{kwargs}.feather"


def save_to_disk(simulate):
    # TODO: 
    def save_simulation(*args, **kwargs):
        df = simulate(*args, **kwargs)
        filename = get_filename(*args, **kwargs)

        df.to_feather(filename)

        return filename

    return save_simulation


SEED = 1
RESOLUTION = 15
REPLICATION_LOG_BOUNDS = (4, 7)
SAMPLE_SIZE_LOG_BOUNDS = (3, 8)
N_REPEATS = 10
N_SIMULTANEOUS_HYPOTHESES = [2, 3, 4, 5]
DATA_SEEDS = range(3)


@save_to_disk
def simulate(
    seed=SEED,
    n_repeats=N_REPEATS,
    resolution=RESOLUTION,
    data_seeds=None,
    n_simultaneous_hypotheses=None,
    sample_size_log_bounds=None,
    replication_log_bounds=None,
):
    n_simultaneous_hypotheses = n_simultaneous_hypotheses or N_SIMULTANEOUS_HYPOTHESES
    sample_size_log_bounds = sample_size_log_bounds or SAMPLE_SIZE_LOG_BOUNDS
    replication_log_bounds = replication_log_bounds or REPLICATION_LOG_BOUNDS
    data_seeds = data_seeds or DATA_SEEDS

    replication_params = np.exp(np.linspace(*replication_log_bounds, RESOLUTION))

    sample_size_params = np.exp(np.linspace(*sample_size_log_bounds, RESOLUTION // 2))

    data_versions = [*product(data_seeds, n_simultaneous_hypotheses)]

    TOTAL_SIMULATIONS = (
        n_repeats
        * len(data_versions)
        * len(replication_params)
        * len(sample_size_params)
    )

    np.random.seed(seed)

    results = []

    with tqdm(total=TOTAL_SIMULATIONS, ncols=100, smoothing=0.005) as pbar:
        for data_ver, m in data_versions:
            metric_generator = MockMetricGenerator(data_ver)
            mocks = metric_generator.generate_metrics(m)
            calculator = SampleSizeCalculator()
            calculator.register_metrics(mocks)

            for _ in range(n_repeats):
                for sample_size in sample_size_params.astype(int):
                    for rep in replication_params.astype(int):
                        start = time.time()
                        power = calculator._expected_average_power(sample_size, rep)
                        duration = time.time() - start

                        results.append(
                            {
                                "power": power,
                                "rep": rep,
                                "sample_size": sample_size,
                                "duration": duration,
                                "data_ver": data_ver,
                                "m": m,
                                "equal_power_group": f"{m}_{data_ver}_{sample_size}",
                            }
                        )

                        pbar.update(1)

    df = pd.DataFrame(results)

    CATEGORICAL = ["rep", "sample_size", "data_ver", "m", "equal_power_group"]
    df[CATEGORICAL] = df[CATEGORICAL].astype("category")

    return df
```

```python
# filename = simulate()
df = pd.read_feather(f"./simulations/default.feather")

mb = "{:,.2f}".format(df.memory_usage(deep=True).sum() / 10**6)
print(f"Using {mb} Mb of memory")
```

As you can see, duration is calculated directly from how long the simulation
takes on my local machine. Presumably these durations will be proportional to
durations in any other setting with adequate memory


We have no "ground" truth for power here, so we'll assume that the average over
all of our simulation repeats and Monte Carlo replication values represents
something close to the truth. We can use that value as a basis for estimating
errors.

```python
error = lambda x: x - x.mean()


equal_powered_groups = df.groupby("equal_power_group")

df["error"] = equal_powered_groups["power"].transform(error)
```

```python
def color_palette(opacity=1, colors=px.colors.qualitative.Plotly):
    palette = []
    for color in colors:
        color = color[1:]
        rgb = []
        for i in (0, 2, 4):
            decimal = int(color[i : i + 2], 16)
            rgb.append(f"{decimal}")

        palette.append(f"rgba({','.join(rgb)},{opacity})")
    return palette
```

```python
jitter = 6 * np.random.random(size=len(df.index))

px.scatter(
    df,
    x=df["rep"].astype('int') + jitter,
    y="error",
    height=700,
    color="duration",
    color_continuous_scale=color_palette(opacity=0.05, colors=["#ff0000", "#0000ff"]),
    title="Deviation of power estimate with respect to grand mean",
    labels={
        "x": "replications (with jitter)",
        "y": "error",
    },
)
```

As you can see, as we increase the number of replications we see a diminishing
return in the typical magnitude of error, but a relatively linear increase in
duration.

```python
df["squared_error"] = df["error"] ** 2
df["error"] = np.abs(df["error"])

df.groupby("rep")[["error", "squared_error", "duration"]].agg(
    [("mean", np.mean), ("95th percentile", lambda x: np.quantile(np.abs(x), 0.95))]
)
```

The table above reveals that for a desired `EPSILON` of 1% choices greater than
~400 should be adequate. If we assume an initial search space where the initial
bounds upper and lower bounds yield power 1 and power 0 respectively, in the
best case we could be able to converge on the true power in roughly $\log_2
\frac{1}{\epsilon}$, which for an `EPSILON` of 1% corresponds to $\approx 7$
recursive calls. Extrapolating based on duration, for my machine, this would
yield a search time of around 0.675 seconds, which is in line with previous
benchmarking efforts.

I recommend we use the following parameter values

```
REPLICATIONS = 400
EPSILON = 0.01
MAX_DEPTH = 20
```


## Further work

We could improve the way `get_sample_size` handles failing to converge.
Sometimes even moderate `EPSILON` choices can lead to a situation where there is
_no_ integer value for sample-size that returns the value we want (e.g. sample
of 12345 yields power .798 and a sample of 12346 yields power .802). Right now
we throw an exception, but it might be more appropriate to add a warning and
return the value that we converged on (e.g. 12346). If we were able to
gracefully handle these cases, we could do away with `MAX_DEPTH` altogether.

We should also consider setting a random seed prior to performing every binary
search to avoid returning conflicting answers to our users