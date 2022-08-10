# Improving the performance of FDR adjustment

[PR #35](https://github.com/godaddy/sample-size/pull/35) introduced support for
multiple metrics and cohorts accounting for FDR adjustment. However, we found
that it is not adequately performant. [PR #The goal of this document is to
demonstrate changes that make the calculator practical for use in an API



```python
import os
import sys
import numpy as np
import cProfile
from timeit import timeit
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import time
from itertools import product
import plotly.express as px

# Set PYTHONPATH=`pwd` when launching jupyter
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
metrics = MockMetricGenerator(5).generate_metrics(10)
pd.DataFrame(
    [{"metric_type": m["metric_type"], **m["metric_metadata"]} for m in metrics]
).round(3)
```

## Benchmarking

We can start by [profiling](https://docs.python.org/3/library/profile.html) our
current implementation for an arbitrary number of metrics. We'll set the
parameters of interest (e.g. number of replications and epsilon) to their
defaults initially.

```python
REPLICATIONS = 500
EPSILON = 0.05
MAX_DEPTH = 20

metric_generator = MockMetricGenerator(1024)
mocks = metric_generator.generate_metrics(6)
```

```python
legacy = LegacyCalculator()
legacy.register_metrics(mocks)
```

```python
improved = ImprovedCalculator()
improved.register_metrics(mocks)
```

```python
# Takes a while to run!
profile_time = cProfile.run(
    "legacy.get_sample_size(REPLICATIONS, EPSILON, MAX_DEPTH)", sort="cumulative"
)
```

<!-- #region -->
### Improving code directly

#### Removing loops
The profiling reveals that the for-loop below accounts for around 90% of the
time spent calculating sample-size.

```python3
 power = []
 num_of_tests = len(self.metrics) * (self.variants - 1)
 for num_true_null in range(num_of_tests):
     num_true_alt = num_of_tests - num_true_null
     nulls = np.array([True] * num_true_null + [False] * num_true_alt)
     for _ in range(replication):
         p_values = []
         true_null = nulls[np.random.permutation(num_of_tests)]
         for v in range(self.variants - 1):
             p_values.extend(
                 [
                     m.generate_p_value(true_null[v * len(self.metrics) + i], sample_size,)
                     for i, m in enumerate(self.metrics)
                 ]
             )
         rejected = multipletests(p_values, alpha=self.alpha, method="fdr_bh")[0]
         power.append(sum(rejected[~true_null]) / num_true_alt)
 return float(np.mean(power))
```

Some things we might do this improve this would be to avoid looping through
`replications` and instead pass a `size` argument to our p-value generators. We
can also improve how we calculate power by avoiding double-averaging our
emperical results. Those changes look like this.

```python3
 rejected_count = 0
 true_alt_count = 0

 # a metric for each test we would conduct
 metrics = self.metrics * (self.variants - 1)

 def fdr_bh(a):
     return multipletests(a, alpha=self.alpha, method="fdr_bh")[0]

 for num_true_alt in range(1, len(metrics) + 1):
     true_alt_count += num_true_alt * replication

     true_alt = np.array([np.random.permutation(len(metrics)) < num_true_alt for _ in range(replication)]).T

     p_values = []
     for i, m in enumerate(metrics):
         p_values.append(m.generate_p_values(true_alt[i], sample_size))

     rejected = np.apply_along_axis(fdr_bh, 1, np.array(p_values).T)

     rejected_count += rejected.sum()

 return rejected_count / true_alt_count
```

#### Quickening convergence
One thing to note is that because we use Bonferroni correction to calculate an
an initial upper bound, our upper estimates are are usually _very_ large
(sometimes in the tens of billions). We can improve the rate at which we
converge on the true sample-size by switching from using the _arithmetic_ mean
to calculate our candidate sample-size to using our a _geometric_ mean (so our
candidate sample sizes are closer to the lower bound). Our original code also
had a bug where it didn't handle the case of our bounds converging _without
achieving the desired power_.

### Results of code improvement
<!-- #endregion -->

```python
profile_time = cProfile.run(
    "improved.get_sample_size(REPLICATIONS, EPSILON, MAX_DEPTH)", sort="cumulative"
)
```

```python
durations = []

REPEATS = 10

def estimate_durations():
    for m in range(2, 6):
        metric_generator = MockMetricGenerator(1024)
        mocks = metric_generator.generate_metrics(m)

        legacy = LegacyCalculator()
        legacy.register_metrics(mocks)

        improved = ImprovedCalculator()
        improved.register_metrics(mocks)

        durations.append({
            "duration (s)": timeit("legacy.get_sample_size(REPLICATIONS, EPSILON, MAX_DEPTH)", number=REPEATS, globals=globals()) / REPEATS,
            "type": "legacy",
            "m": m,
        })

        durations.append({
            "duration (s)": timeit("improved.get_sample_size(REPLICATIONS, EPSILON, MAX_DEPTH)", number=REPEATS, globals=globals()) / REPEATS,
            "type": "improved",
            "m": m,
        })

    pd.DataFrame(durations).to_feather("durations.csv")
```

```python
# estimate_durations()
pd.read_feather('durations.csv').pivot(index='m', columns=['type'], values=['duration (s)']).to_markdown()
```

<!-- #region -->
## Parameter Tuning

For the benchmarking above we used

```python3
REPLICATIONS = 500
EPSILON = 0.05
MAX_DEPTH = 20
```

These values were selected fairly arbitrarily, and we can likely improve on
them. They also all depend on each other. For `EPSILON`, it's probably the case
that if we're going through the trouble of estimating sample sizes at all we
would probably prefer a better precision than $\pm 0.05$. However, as we
increase the precision we increase our time complexity by order
$\mathcal{O}(log(\epsilon))$. That also opens the question of `REPLICATIONS`,
which determines the precision of our empirical estimates by
$\mathcal{O}(\frac{1}{\sqrt{n}})$ and (assuming our estimates are good enough)
increases the time complexity of the entire search by $\mathcal{O}(n)$.
`REPLICATIONS` places an essential limit on our precision, regardless of what we
set `EPSILON` to, and these values determine whether we're likely to encounter
`MAX_DEPTH` before we converge on an estimate.

Here are some simulation results where we try to quantify the variability of
empirical estimates of power as a function of `REPLICATIONS`, which we can use
to select a trade-off with time complexity in order to then choose `EPSILON`.
<!-- #endregion -->

```python
RESOLUTION = 8
REPLICATION_NUBMER = np.exp(np.linspace(4, 8, RESOLUTION)).astype(int)
SAMPLE_SIZES = np.exp(np.linspace(3, 8.5, RESOLUTION // 2)).astype(int)

N_REPEATS = 50

N_METRICS = [2, 3, 4]
DATA_VERSIONS = [*product(range(2), N_METRICS)]

results = []

TOTAL_SIMULATIONS = (
    N_REPEATS * len(DATA_VERSIONS) * len(REPLICATION_NUBMER) * len(SAMPLE_SIZES)
)

def run_simulation():
    with tqdm(total=TOTAL_SIMULATIONS, ncols=100, smoothing=0.01) as pbar:
        for data_ver, m in DATA_VERSIONS:
            metric_generator = MockMetricGenerator(data_ver)
            mocks = metric_generator.generate_metrics(m)
            improved = ImprovedCalculator()
            improved.register_metrics(mocks)

            for _ in range(N_REPEATS):
                for sample_size in SAMPLE_SIZES:
                    for rep in REPLICATION_NUBMER:
                        start = time.time()
                        power = improved._expected_average_power(sample_size, rep)
                        duration = time.time() - start

                        results.append(
                            {
                                "power": power,
                                "rep": rep,
                                "sample_size": sample_size,
                                "duration": duration,
                                "data_ver": data_ver,
                                "m": m,
                            }
                        )

                        pbar.update(1)

    pd.DataFrame(results).to_feather("empirical_power.csv")
```

```python
# run_simulation()
df = pd.read_feather('empirical_power.csv')
```

```python
squared_error = lambda x: (x - x.mean()) ** 2

equal_powered_groups = df.groupby(['sample_size', 'data_ver', 'm'])

df['squared_error'] = equal_powered_groups['power'].transform(squared_error)
```

```python
df_mean = df.groupby('rep').mean()

df_mean['sqrt_MSE'] = np.sqrt(df_mean['squared_error'])
df_mean[['sqrt_MSE', 'duration']]
```

As you can see, even for large choices of `REPLICATIONS`, the duration is low
enough to still be of practical value. For a desired `EPSILON` of 0.01, we would
want to pick a value greater than ~500.

I recommend we use 500 replications, as this seems to be a reasonable trade-off
between speed (about a tenth of a second per recursive call) and precision (a
little better than 0.01).


## Further work

We could improve the way `get_sample_size` handles failing to converge.
Sometimes even moderate `EPSILON` choices can lead to a situation where there is
_no_ integer value for sample-size that returns the value we want (e.g. sample
of 12345 yields power .798 and a sample of 12346 yields power .802). Right now
we throw an exception, but it might be more appropriate to add a warning and
return the value that we converged on (e.g. 12346).

We should also consider setting a random seed prior to performing every binary
search to avoid returning conflicting answers to our users
