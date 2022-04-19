# 1. Determine how to choose the assumed number of true null hypotheses
# We assume the frequency of each number of $H_0$ are ~uniform(0,M). Thus we define expected average power as arithmetic mean of expected average power for all possible true $H_0$
# 2. Provide an algorithm in Python to solve/search for required sample size based on the prosed method in ML-4990 (p-value generator, average power calculator, sample size calculator)
# 3. Validate the choice by comparing the power with power when using sample size without any adjustment
# 4. add ratio metrics
# 5. add number of variants as input

# import packages
from statsmodels.stats.power import TTestIndPower
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from scipy import stats
import pandas as pd
from itertools import product, combinations_with_replacement, combinations
from math import sqrt
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest

seed = 10


def p_value_generator(metric_type, baseline, delta, var, sample_size, rep, num_mean=None, num_var=None,
                      den_mean=None, den_var=None, cov=None):
    p_null, p_alt=np.zeros(rep), np.zeros(rep)
    if metric_type == 'Numeric':
            null_data = np.random.normal(baseline, np.sqrt(var), (sample_size, rep * 2))
            alt_data = np.random.normal(baseline + delta, np.sqrt(var), (sample_size, rep))

            # resample all negative values
            while np.sum(null_data < 0) > 0:
                null_data[null_data < 0] = np.random.normal(baseline, np.sqrt(var), len(null_data[null_data < 0]))

            while np.sum(alt_data < 0) > 0:
                alt_data[alt_data < 0] = np.random.normal(baseline + delta, np.sqrt(var),
                                                          len(alt_data[alt_data < 0]))
            for k in range(rep):
                p_null[k] = stats.ttest_ind(null_data[:, k], null_data[:, k + rep])[1]
                p_alt[k] = stats.ttest_ind(null_data[:, k], alt_data[:, k])[1]

    elif metric_type == 'Ratio':
            var = (num_var / den_mean ** 2
                    + den_var * num_mean ** 2 / den_mean ** 4
                    - 2 * cov * num_mean / den_mean ** 3)

            null_data=np.random.normal(num_mean/ den_mean, np.sqrt(var),
                             (sample_size, rep*2))
            alt_data=np.random.normal(num_mean/ den_mean + delta, np.sqrt(var),
                             (sample_size, rep))

            for l in range(rep):
                p_null[l] = ztest(null_data[:, l], null_data[:, l + rep])[1]
                p_alt[l] = ztest(null_data[:, l], alt_data[:, l])[1]

    elif metric_type == 'Boolean':
            null_data = np.round(np.random.normal(baseline * sample_size, np.sqrt(baseline*(1-baseline) * sample_size),
                                         rep * 2))
            alt_data = np.round(np.random.normal((baseline + delta) * sample_size,
                                                 np.sqrt((baseline+delta)*(1-baseline-delta) * sample_size),rep))
            for j in range(rep):
                p_null[j] = proportions_ztest([null_data[j], null_data[j + rep]], [sample_size, sample_size])[1]
                p_alt[j] = proportions_ztest([null_data[j], alt_data[j]], [sample_size, sample_size])[1]

    return p_null, p_alt


# will explicitly take number of variants and metrics as input
def expected_average_power(variants, metric_type, baseline, delta, var, sample_size, alpha, rep,num_mean=None,
                           num_var=None,den_mean=None,den_var=None, cov=None):
    m = len(metric_type)

    # p-values of true null and true alternatives
    pp_null = np.zeros((m*(variants-1), rep))
    pp_alt = np.zeros((m*(variants-1), rep))

    metric_type=np.repeat(metric_type,variants-1)
    baseline=np.repeat(baseline,variants-1)
    delta=np.repeat(delta,m*(variants-1))
    var=np.repeat(var,m*(variants-1))

    for i in range(m*(variants-1)):
        pp=p_value_generator(metric_type[i], baseline[i], delta[i], var[i], sample_size, rep, num_mean, num_var,
                             den_mean, den_var, cov)
        pp_null[i,:]=pp[0]
        pp_alt[i,:]=pp[1]


    # 0=true null, 1=true alternative, exclude when all H are true null
    true_H = [*product([0, 1], repeat=m*(variants-1))][1:]
    # sum of average power for each n combination and their counts
    result = np.zeros((m*(variants-1), 2))
    # 1=rejected, 0=fail to reject
    rejs = np.empty((rep, m*(variants-1)))

    for t in range(len(true_H)):
        null_index = np.argwhere(np.array(true_H[t]) == 0)
        alt_index = np.argwhere(np.array(true_H[t]) == 1)

        for r in range(rep):
            true_null_p = [float(pp_null[x, r]) for x in null_index]
            true_alt_p = [float(pp_alt[x, r]) for x in alt_index]
            # first len(true_null) hypotheses are true null
            pvalues = np.concatenate((np.array(true_null_p), np.array(true_alt_p)))
            rejs[r, :] = multipletests(pvalues, alpha=alpha, method='fdr_bh')[0]

        actual_pw = np.sum(rejs[:, len(null_index):], axis=1) / len(alt_index)

        # i_th row in result shows power when there are i true null decision metrics for all variants combined
        result[len(null_index), 0] += np.mean(actual_pw)
        result[len(null_index), 1] += 1

    # return each possible true null's expected power for reference as well as aggregated expected power
    return pd.DataFrame(result[:, 0] / result[:, 1], columns=['avg power']), np.mean(result[:, 0] / result[:, 1])


def required_sample_size(variants, metric_type, baseline, delta, var, power, alpha, rep, interval=10, num_mean=None,
                           num_var=None,den_mean=None,den_var=None, cov=None):
    m = len(metric_type)

    obj = TTestIndPower()
    # calculate required sample size based on minimum standardized effect size since it requires maximum sample size
    d = np.min(delta / np.sqrt(var))
    lower = obj.solve_power(effect_size=d, alpha=alpha, power=power,
                            ratio=1, alternative='two-sided')
    upper = obj.solve_power(effect_size=d, alpha=alpha / m, power=power,
                            ratio=1, alternative='two-sided')

    # print(f'We look for the minimum required sample size in range [{int(lower)},{int(upper)}]')
    for size in np.linspace(lower, upper, interval):
        exp_power = expected_average_power(variants, metric_type, baseline, delta, var, int(size), alpha, rep,
                                           num_mean, num_var,den_mean,den_var, cov)[1]
        if exp_power >= power:
            return [int(size), np.round(exp_power, 4), int(lower), int(upper)]
    return [int(size), np.round(exp_power, 4), int(lower), int(upper)]


expected_average_power(variants=3, metric_type=['Boolean','Boolean','Numeric'], baseline=[0.08,0.07,10], delta=[0.01,0.02,2], var=[0.08*0.92,0.07*0.93,1000],
                       sample_size=1000, alpha=0.05, rep=100,num_mean=None,
                           num_var=None,den_mean=None,den_var=None, cov=None)

required_sample_size(variants=3, metric_type=['Boolean','Boolean','Numeric'], baseline=[0.08,0.07,10],
                     delta=[0.01,0.02,2], var=[0.08*0.92,0.07*0.93,1000], power=0.8, alpha=0.05, rep=100)
