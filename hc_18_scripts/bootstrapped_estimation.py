import numpy as np
def delta_delta_hedges_g(groups, n_bootstraps=5000, ci=95):
    W_pl, M_pl, W_dr, M_dr = groups
    obs_g = lambda a, b: hedges_g(a, b, paired=False)
    obs = {
        'delta_M': obs_g(M_pl, M_dr),
        'delta_W': obs_g(W_pl, W_dr)
    }
    obs['delta_delta'] = obs['delta_M'] - obs['delta_W']

    boot_M = bootstrap_hedges_g(M_pl, M_dr, n_bootstraps=n_bootstraps)[0]
    boot_W = bootstrap_hedges_g(W_pl, W_dr, n_bootstraps=n_bootstraps)[0]
    boot_dd = boot_M - boot_W

    lower = np.percentile(boot_dd, (100-ci)/2)
    upper = np.percentile(boot_dd, 100-(100-ci)/2)

    return obs, boot_dd, (lower, upper)



def hedges_g(group1, group2, paired = False):
    """
    Calculate Hedges' g for paired data.
    """
    if paired:
        d = cohens_d_paired(group1, group2)
    else:
        d = cohens_d_unpaired(group1, group2)
        
    n1, n2 = len(group1), len(group2)

    # Apply Hedges' g bias correction
    bias_correction = 1 - (3 / ((4 * (n1 + n2)) - 9))
    g = d * bias_correction
    
    return g
import numpy as np

def bootstrap_hedges_g(group1, group2, n_bootstraps=1000, ci=95, paired=False):
    group1 = np.array(group1)
    group2 = np.array(group2)
    bootstrapped_g = []

    if paired:
        assert len(group1) == len(group2), "For paired comparison, both groups must be the same size."
        n = len(group1)

        for _ in range(n_bootstraps):
            indices = np.random.choice(n, size=n, replace=True)
            resample1 = group1[indices]
            resample2 = group2[indices]
            g = hedges_g(resample1, resample2, paired=True)
            bootstrapped_g.append(g)
    else:
        for _ in range(n_bootstraps):
            resample1 = np.random.choice(group1, size=len(group1), replace=True)
            resample2 = np.random.choice(group2, size=len(group2), replace=True)
            g = hedges_g(resample1, resample2, paired=False)
            bootstrapped_g.append(g)

    bootstrapped_g = np.array(bootstrapped_g)
    
    lower = np.percentile(bootstrapped_g, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_g, 100 - (100 - ci) / 2)
    
    return bootstrapped_g, (lower, upper)



def cohens_d_unpaired(group1, group2):
    # Calculate means
    M1, M2 = np.nanmean(group1), np.nanmean(group2)
    
    # Calculate standard deviations
    s1, s2 = np.nanstd(group1, ddof=1), np.nanstd(group2, ddof=1)
    
    # Sample sizes
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    s_p = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (M1 - M2) / s_p
    return d


def cohens_d_paired(data1, data2):
    # Compute the difference scores
    diff = data1 - data2
    
    # Mean and standard deviation of the differences
    M_d = np.nanmean(diff)
    s_d = np.nanstd(diff, ddof=1)
    
    # Cohen's d for paired data
    d = M_d / s_d
    return d


def pvalue_hedges_g_permutation(group1, group2, n_permutations=1000, ci=95, paired=False):
    """Computes Hedges' g with a permutation test by shuffling group1 and group2 values."""
    observed_g = hedges_g(group1, group2, paired)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    shuffled_g = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)  # Shuffle combined data
        perm_group1, perm_group2 = combined[:n1], combined[n1:]  # Split into new groups

        g = hedges_g(perm_group1, perm_group2, paired)
        shuffled_g.append(g)
    shuffled_g = np.array(shuffled_g)

    # Compute p-value: proportion of permuted g-values more extreme than observed g
    p_value = (np.nansum(np.abs(shuffled_g) >= np.abs(observed_g)) + 1) / (n_permutations+1)

    return observed_g, shuffled_g, p_value




def calculate_p_value(observed_statistic, shuffled_distribution, alternative='two-sided'):
    # https://www.davidzeleny.net/anadat-r/doku.php/en:monte_carlo#:~:text=The%20P-value%20is%20calculated,added%20into%20the%20null%20distribution.

    from collections import namedtuple
    Statistic = namedtuple('Statistic', ['p_value', 'distribution_size', 'extreme_counts'])
    distribution_size = shuffled_distribution.shape[0]
    
    if alternative == 'two-sided':
        # Calculate the proportion for two-sided test
        extreme_counts_greater = np.nansum(shuffled_distribution >= observed_statistic)
        greater_proportion = (extreme_counts_greater + 1) / (distribution_size + 1)
        
        extreme_counts_less = np.nansum(shuffled_distribution <= observed_statistic)
        less_proportion = (extreme_counts_less + 1) / (distribution_size + 1)
        
        # Two-tailed p-value is twice the minimum of both tails
        p_value = 2 * np.nanmin([greater_proportion, less_proportion])
        extreme_counts = np.nanmin([extreme_counts_greater, extreme_counts_less])
    
    elif alternative == 'greater':
        # One-sided (greater) test
        extreme_counts = np.nansum(shuffled_distribution >= observed_statistic)
        p_value = (extreme_counts + 1) / (distribution_size + 1)
    
    elif alternative == 'less':
        # One-sided (less) test
        extreme_counts = np.nansum(shuffled_distribution <= observed_statistic)
        p_value = (extreme_counts + 1) / (distribution_size + 1)
    else:
        
        raise(ValueError(f"Invalid value for 'alternative': {alternative}. Expected 'two-sided', 'greater', or 'less'."))

    # If p-value is zero, set to minimum resolution
    # if extreme_counts == 0:
        # p_value = 1 / (distribution_size + 1)

    statistic = Statistic(p_value=p_value, distribution_size=distribution_size, extreme_counts=extreme_counts)
    
    return statistic