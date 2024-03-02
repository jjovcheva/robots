# Analysing Robot Performance in Python
This analysis is performed on data from robotics sessions at the College of Charleston in South Carolina. The variables of interest are stress, attention, mental workload, and robot predictability. We are interested in the following relationships: 

__Single-factor__
- Robots: one robot sessions (abcd) vs two robot sessions (efgh)
- Speed: fast sessions (cdh) vs slow sessions (abg) vs 1 fast 1 slow (ef)
- Orientation: right focus (acf) vs left focus (bde)

__Two-factor__
- Speed v Robot Number: one robot slow (ab) vs two robots slow (g) and one robot fast (cd) vs two robots fast (h)
- Speed v Orientation: right robot slow (ae) vs left robot slow (bf) and right robot fast (cf) vs left robot fast (de)

The normality of the data for each variable was assessed using the Kolmogorov-Smirnov (KS) test. Suppose that a population is believed to have some cumulative frequency distribution function $F(x)$, i.e. for any value $x$, $F$ is the proportion of individuals with measurements $\le x$. The cumulative step-function of a random sample of $N$ observations, $S_N(x)$, is expected to resemble $F(x)$. If $S_N(x)$ deviates significantly from $F(x)$, there is reason to believe the hypothetical distribution is not the correct one. In this case, then, the null hypothesis is that the data come from a normal distribution.

Next, the two-sample KS test was used to compare the distributions for each of the conditions. Here, the KS statistic quantifies the distance between cumulative distribution functions (CDFs) of pairs of distributions. Again, the CDF represents the probability that a random variable takes on a value $\le x$. The KS statistic is then $$D_n=\mathrm{max}|F_1(x)-F_2(x)|$$ where $F_1, F_2$ are the CDFs of the distributions being compared. In this case, the null hypothesis posits that the two samples are drawn from the same distribution, while the alternative hypothesis suggests the distributions are significantly different.

Levene's test was used to test the equality of variances for the different samples. For a variable $Y$ of a sample with size $N$ divided into $k$ subgroups, the Levene test statistic is given by
```math
    W=\frac{(N-k)}{(k-1)} \frac{\sum_{i=1} N_i\left(Z_{i .}-Z_{. .}\right)^2}{\sum_{i=1} \sum_{j=1}\left(Z_{i j}-Z_{i .}\right)^2},
```
where $Z_{ij}=|Y_{ij}-\bar Y_{i.}|$.

The distributions are then analysed using the Kruskal-Wallis test. The test is non-parametric and suitable for skewed distributions. However, the distributions to be compared are assumed to have a similar shape. The distributions are plotted to check that the data meet this criterion. The null hypothesis is that the medians are equal. The test statistic is given by
```math
    H=(N-1) \frac{\sum_{i=1} n_i\left(\bar{r}_{i .}-\bar{r}\right)^2}{\sum_{i=1} \sum_{j=1}\left(r_{i j}-\bar{r}\right)^2},
```
where $N$ is the total number of observations, $g$ is the number of groups, $n_i$ is the number of observations in group $i$, $r_{ij}$ is the rank of observation $j$ in group $i$, $\bar r_{i .}=\sum_{j=1} r_{i j}/{n_i}$ is the average rank of all observations in group $i$, and $\bar r=\frac{1}{2}(N+1)$ is the average of all $r_{ij}$.

Samples with non-equal variance are analysed using an additional ad-hoc Mann-Whitney U test, also known as Wilcoxon rank-sum test (not to be mistaken for the Wilcoxon signed-rank test) with a Bonferroni correction to handle the problem of multiple comparisons. Significant comparisons are followed up with a calculation of the Cliff's Delta statistic, 
```math
    d=\frac1{m n}\sum_i\sum_j \delta(i,j) \ \text{where } \delta=\begin{cases} +1 & x_i > y_i \\ -1 & x_i < y_i \\ 0 & x_i=y_i \end{cases},
```
where the two distributions being compared have sizes $m, n$ and contain items $x_i, x_j$ respectively. The Cliff's Delta is a non-parametric effect size measurement that quantifies the difference between two groups of observations and, usefully, it captures the direction of the difference in sign in the Mann-Whitney U statistic. 





