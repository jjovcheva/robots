# Analysing Robot Performance in Python
This analysis is performed on data from robotics sessions at the College of Charleston in South Carolina. The variables of interest are stress, attention, mental workload, and robot predictability. We are interested in the following relationships: 
** Single-factor **
- Robots: one robot sessions (abcd) vs two robot sessions (efgh)
- Speed: fast sessions (cdh) vs slow sessions (abg) vs 1 fast 1 slow (ef)
- Orientation: right focus (acf) vs left focus (bde)

** Two-factor **
- Speed v Robot number: one robot slow (ab) vs two robots slow(g) and one robot fast (cd) vs two robots fast (h)
- Speed v Orientation: right robot slow (ae) vs left robot slow (bf) and right robot fast (cf) vs left robot fast (de)

The normality of the data for each variable is assessed using the Kolmogorov-Smirnov (KS) test. Suppose that a population is believed to have some cumulative frequency distribution function $F(x)$, i.e. for any value $x$, $F$ is the proportion of individuals with measurements $\le x$. The cumulative step-function of a random sample of $N$ observations, $S_N(x)$, is expected to resemble $F(x)$. If $S_N(x)$ deviates significantly from $F(x)$, there is reason to believe the hypothetical distribution is not the correct one. In this case, then, the null hypothesis is that the data come from a normal distribution. 

Next, the two-sample KS test is used to compare the distributions for each of the conditions. Here, the KS statistic quantifies the distance between cumulative distribution functions (CDFs) of pairs of distributions. Again, the CDF represents the probability that a random variable takes on a value $\le x$. The KS statistic is then $$D_n=\text{max}|F_1(x)-F_2(x)|,$$ where $F_1, F_2$ are the CDFs of the distributions being compared. In this case, the null hypothesis posits that the two samples are drawn from the same distribution, while the alternative hypothesis suggests the distributions are significantly different. 

Levene's test is used to test the equality of variances for the different conditions. For a variable $Y$ of a sample with size $N$ divided into $k$ subgroups, the Levene test statistic is given by $ W = \frac{N-k}{k-1}\frac{\sum_{i=1}^k N_i(Z_{i.} - Z_{..})^2}{\sum_{i=1}^k \sum_{j=1}^{N_i} N_i(Z_{ij} - Z_{i.})^2}, $ where $Z_{ij}=|Y_{ij}-\bar{Y_{i.}}|$. 





