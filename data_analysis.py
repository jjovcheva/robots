import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kstest, kruskal, chi2_contingency, levene, wilcoxon, ecdf, norm, mannwhitneyu, ks_2samp
import numpy as np
from itertools import combinations

# Set up plotting styles.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"],
    "font.size": 12
})

def load_data(file_path='data.xlsx'):
    '''
    Load data from an Excel file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str, optional
        The path to the Excel file. Default is 'data.xlsx'.

    Returns
    -------
    data : DataFrame
        The loaded data.
    numeric_cols : list
        List of numeric column names.
    one_var_groups : dict
        Dictionary defining groups for single-factor analysis.
    two_var_groups : dict
        Dictionary defining groups for two-factor analysis.
    categories : dict
        Dictionary containing category names and their values.
    '''
    data = pd.read_excel(file_path)

    # Rename columns for clarity.
    columns = {
        'Emotion': 'Stress',
        'Unnamed: 2': 'Attention',
        'Unnamed: 3': 'Excitement',
        'Unnamed: 4': 'Robot Predictability',
        'Mental Workload': 'Demand',
        'Unnamed: 6': 'Rush',
        'Unnamed: 7': 'Difficulty',
        'Unnamed: 8': 'Negative Affect',
        'Unnamed: 9': 'Success',
        'Unnamed: 10': 'Total Workload'
    }
    data.rename(columns=columns, inplace=True)

    # Drop the first row.
    data = data.drop(data.index[0])

    # Select only the relevant columns.
    selected_columns = ['Session Type', 'Stress', 'Attention', 'Total Workload', 'Robot Predictability']
    data = data[selected_columns]

    # Convert data types to numeric where appropriate.
    numeric_cols = ['Stress', 'Attention', 'Robot Predictability', 'Total Workload']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Create additional categorical variables.
    data['Robots'] = data['Session Type'].apply(lambda x: 'One Robot' if x[0] in 'abcd' else 'Two Robots')
    data['Speed'] = data['Session Type'].apply(lambda x: 'Fast' if x[0] in 'cdh' else ('Slow' if x[0] in 'abg' else 'Mixed'))
    data['Orientation'] = data['Session Type'].apply(lambda x: 'Right Focus' if x[0] in 'acf' else 'Left Focus')

    # Define session groups for single-factor analysis.
    one_var_groups = {
        "Robot Number": (['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']),
        "Speed": (['c', 'd', 'h'], ['a', 'b', 'g'], ['e', 'f']),
        "Orientation": (['a', 'c', 'f'], ['b', 'd', 'e'])
    }

    # Define two-factor session groups.
    two_var_groups = {
        "Speed v Robot Number - Fast": (['c', 'd'], ['h']),
        "Speed v Robot Number - Slow": (['a', 'b'], ['g']),
        "Speed v Orientation - Fast": (['a', 'e'], ['b', 'f']),
        "Speed v Orientation - Slow": (['c', 'f'], ['d', 'e'])
    }
    
    # Define categories for visualization.
    categories = {
        'Robots': ['One Robot', 'Two Robots'],
        'Speed': ['Fast', 'Slow', 'Mixed'],
        'Orientation': ['Right Focus', 'Left Focus']
    }

    data.to_csv('my_data.txt', sep='\t', index=False) 

    return data, numeric_cols, one_var_groups, two_var_groups, categories

data, numeric_cols, one_var_groups, two_var_groups, categories = load_data()

def analyse_dist(data):
    '''
    Analyse the distribution of variables.

    Parameters
    ----------
    data : DataFrame
        The input DataFrame.

    Returns
    -------
    None
    '''
    # Calculate basic statistics.
    res = {}
    for column in numeric_cols:
        mean = np.mean(data[column])
        variance = np.var(data[column])
        std_dev = np.sqrt(variance)
        res[column] = {"mean": mean, "variance": variance, "std dev": std_dev}

    file_path = './stats/basic_stats.txt'

    # Save the results to a text file.
    with open(file_path, 'w+') as file:
        for column, result in res.items():
            file.write(f"{column}\n")
            file.write(f"  Mean: {result['mean']}\n")
            file.write(f"  Variance: {result['variance']}\n")
            file.write(f"  Standard Deviation: {result['std dev']}\n\n")

    # Create histograms.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(data['Stress'], ax=axes[0, 0])
    axes[0, 0].set_title('Stress Levels')
    sns.histplot(data['Attention'], ax=axes[0, 1])
    axes[0, 1].set_title('Attention Levels')
    sns.histplot(data['Robot Predictability'], ax=axes[1, 0])
    axes[1, 0].set_title('Robot Predictability')
    sns.histplot(data['Total Workload'], ax=axes[1, 1])
    axes[1, 1].set_title('Total Workload')
    
    plt.tight_layout()
    plt.savefig('./plots/histograms', dpi=800)
    
def calc_ks_stat(data):
    '''
    Calculate the Kolmogorov-Smirnov (KS) test statistics for each variable.

    Parameters
    ----------
    data : DataFrame
        The input DataFrame.

    Returns
    -------
    None
    '''
    res = {}
    for column in numeric_cols:
        # Perform the Kolmogorov-Smirnov test.
        stat, p_value = kstest(data[column], 'norm')
        # Store result.
        res[column] = {"ks-stat": stat, "p_value": p_value}

    file_path = './stats/ks_test.txt'

    # Save the result.
    with open(file_path, 'w+') as file:
        # Iterate over the dictionary and write each result to the file.
        for column, result in res.items():
            file.write(f"{column}\n")
            file.write(f"  Kolmogorov-Smirnov statistic: {result['ks-stat']}\n")
            file.write(f"  p-value: {result['p_value']}\n\n")
            if result['p_value'] > 0.05:
                file.write(f" '{column}' normally distributed.\n\n")
            else:
                file.write(f" '{column}' not normally distributed.\n\n")

def ecdf(data):
    '''
    Compute the empirical cumulative distribution function (ECDF) for a 1D array.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame.

    Returns
    -------
    x : ndarray
        Sorted unique values of the input array.
    y : ndarray
        Empirical cumulative probabilities corresponding to the sorted values.
    '''
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y   
            
def plot_ks(data):
    '''
    Plot Kolmogorov-Smirnov (KS) test results for each variable.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame.

    Returns
    -------
    None
    '''
    # Set up 2x2 grid for plotting.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, column in enumerate(numeric_cols):
        # Compute ECDF for the selected column.
        x, y = ecdf(data[column].dropna())

        # Theoretical CDF (assuming normal distribution).
        x_theor = np.linspace(min(data[column].dropna()), max(data[column].dropna()), 100)
        theoretical_cdf = norm.cdf(x_theor, loc=np.mean(data[column]), scale=np.std(data[column]))

        # Perform Kolmogorov-Smirnov test.
        ks_statistic, p_value = kstest(data[column].dropna(), 'norm', args=(np.mean(data[column]), np.std(data[column])))

        # Plot results.
        ax = axes[i//2, i%2]
        ax.plot(x_theor, theoretical_cdf, label='Theoretical CDF', color='red')
        ax.step(x, y, label='ECDF', color='blue')
        ax.set_title(f'{column} (KS Statistic: {ks_statistic:.3f}, p-value: {p_value:.3f})')
        ax.set_xlabel(f'{column}')
        ax.set_ylabel('Cumulative Probability')
        ax.legend()

        # Highlight the maximum difference.
        diff = np.abs(y - norm.cdf(x, loc=np.mean(data[column]), scale=np.std(data[column])))
        max_diff_idx = np.argmax(diff)
        ax.fill_betweenx([0, y[max_diff_idx]], x[max_diff_idx], x_theor[np.searchsorted(x_theor, x[max_diff_idx])], color='gray', alpha=0.5)

    plt.tight_layout()
    plt.savefig('./plots/ks_tests.png', dpi=500)
    
def ks_one_var(data, cat_col, groups):    
    '''
    Perform Kolmogorov-Smirnov (KS) test for one variable across different groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    cat_col : str
        Name of the categorical column specifying groups.
    groups : list
        List of group identifiers.

    Returns
    -------
    res : dict
        Dictionary containing KS test results for each column and pairwise group comparisons.
        Keys are column names, values are lists of tuples (group1, group2, KS statistic, p-value).
    '''
    res = {}
    
    for col in numeric_cols:
        pairwise_results = []
        for group1, group2 in combinations(groups, 2):
            # Extract data for the two groups.
            data1 = data[data[cat_col] == group1][col]
            data2 = data[data[cat_col] == group2][col]

            # Perform Kolmogorov-Smirnov test.
            stat, p_value = ks_2samp(data1, data2)

            pairwise_results.append((group1, group2, stat, p_value))

        res[col] = pairwise_results
    
    return res

def calc_ks_one_var(data):
    '''
    Calculate and save KS test results for one variable across different groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    None
    '''
    # Set up dictionary for results.
    res = {}

    # Perform tests for each category.
    for category, groups in categories.items():
        category_res = ks_one_var(data, category, groups)
        res[category] = category_res
        
    res_text = ""
    for category, results in res.items():
        res_text += f"{category}:\n"
        for variable, var_res in results.items():
            res_text += f"  {variable}:\n"
            for group1, group2, stat, p_value in var_res:
                res_text += f"    {group1} vs {group2}:\n"
                res_text += f"      Kolmogorov-Smirnov statistic: {stat}\n"
                res_text += f"      p-value: {p_value}\n\n"

    # Save the results to a text file.
    file_path = './stats/ks_one_var.txt'
    with open(file_path, 'w+') as file:
        file.write(res_text)
                
def ks_two_var(data, group1, group2, var):    
    '''
    Perform Kolmogorov-Smirnov (KS) test for two variables across two groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    group1 : list
        List of identifiers for the first group.
    group2 : list
        List of identifiers for the second group.
    var : str
        Name of the variable for which the KS test is performed.

    Returns
    -------
    stat : float
        KS statistic.
    p_value : float
        Two-tailed p-value.
    '''
    data_1 = data[data['Session Type'].isin(group1)][var]
    data_2 = data[data['Session Type'].isin(group2)][var]
    stat, p_value = ks_2samp(data_1, data_2)
    
    return stat, p_value

def calc_ks_two_var(data):
    '''
    Calculate and save KS test results for two variables across two groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    None
    '''
    # Perform test and collect results for the comparisons.
    res = {condition: {variable: ks_two_var(data, groups[0], groups[1], variable)
                    for variable in numeric_cols}
                    for condition, groups in two_var_groups.items()}

    # Save results.
    file_path = './stats/ks_two_var.txt'
    with open(file_path, 'w+') as file:
        for condition, variables in res.items():
            file.write(f"{condition}:\n")
            for variable, result in variables.items():
                file.write(f"  {variable}:\n")
                file.write(f"    KS statistic: {result[0]}\n")
                file.write(f"    p-value: {result[1]}\n")
            file.write("\n")
              
def levene_one_var(data, groups, var):
    '''
    Perform Levene's test for one variable across different groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    groups : dict
        Dictionary defining groups for analysis.
    var : str
        Name of the variable for which the Levene's test is performed.

    Returns
    -------
    stat : float
        Levene's test statistic.
    p_value : float
        p-value.
    '''
    # Extract group data.
    group_data = [data[data['Session Type'].isin(group)][var] for group in groups]
    
    # Calculate Levene statistic and p-value.
    stat, p_value = levene(*group_data)
    return stat, p_value

def calc_levene_one_var(data):
    '''
    Calculate and save Levene's test results for one variable across different groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    None
    '''
    # Perform test and collect results.
    res = {variable: {condition: levene_one_var(data, groups, variable)
                    for condition, groups in one_var_groups.items()}
                    for variable in numeric_cols}

    file_path = './stats/levene_one_var.txt'

    # Save results to text file.
    with open(file_path, 'w+') as file:
        for variable, conditions in res.items():
            file.write(f"{variable}:\n")
            for condition, result in conditions.items():
                file.write(f"  {condition}:\n")
                file.write(f"    Levene statistic: {result[0]}\n")
                file.write(f"    p-value: {result[1]}\n")
            file.write("\n")

def levene_two_var(data, group1, group2, var):
    '''
    Perform Levene's test for two variables across two groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    group1 : list
        List of identifiers for the first group.
    group2 : list
        List of identifiers for the second group.
    var : str
        Name of the variable for which the Levene's test is performed.

    Returns
    -------
    stat : float
        Levene's test statistic.
    p_value : float
        p-value.
    '''
    data_1 = data[data['Session Type'].isin(group1)][var]
    data_2 = data[data['Session Type'].isin(group2)][var]

    stat, p_value = levene(data_1, data_2)
    return stat, p_value

def calc_levene_two_var(data):
    '''
    Calculate and save Levene's test results for two variables across two groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    None
    '''
    # Perform test and collect results for the comparisons.
    res = {condition: {variable: levene_two_var(data, groups[0], groups[1], variable)
                    for variable in numeric_cols}
                    for condition, groups in two_var_groups.items()}

    # Save results.
    file_path = './stats/levene_two_var.txt'
    with open(file_path, 'w+') as file:
        for condition, variables in res.items():
            file.write(f"{condition}:\n")
            for variable, result in variables.items():
                file.write(f"  {variable}:\n")
                file.write(f"    Levene statistic: {result[0]}\n")
                file.write(f"    p-value: {result[1]}\n")
            file.write("\n")

def kruskal_one_var(data, groups, var):
    '''
    Perform Kruskal-Wallis test for one variable across different groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    groups : dict
        Dictionary defining groups for analysis.
    var : str
        Name of the variable for which the Kruskal-Wallis test is performed.

    Returns
    -------
    stat : float
        Kruskal-Wallis test statistic.
    p_value : float
        p-value.
    '''
    # Extract data for each group.
    group_data = [data[data['Session Type'].isin(group_sessions)][var] for group_sessions in groups]
    # Perform Kruskal-Wallis test.
    stat, p_value = kruskal(*group_data)
        
    return stat, p_value

def calc_kruskal_one_var(data):
    '''
    Calculate and save Kruskal-Wallis test results for one variable across different groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    None
    '''
    # Perform test and collect results.
    res = {variable: {condition: kruskal_one_var(data, groups, variable)
                    for condition, groups in one_var_groups.items()}
                    for variable in numeric_cols}

    file_path = './stats/kruskal_one_var.txt'

    # Save results to text file.
    with open(file_path, 'w+') as file:
        for variable, conditions in res.items():
            file.write(f"{variable}:\n")
            for condition, result in conditions.items():
                file.write(f"  {condition}:\n")
                file.write(f"    Kruskal-Wallis statistic: {result[0]}\n")
                file.write(f"    p-value: {result[1]}\n")
            file.write("\n")
           
def kruskal_two_var(data, group1, group2, var):
    '''
    Perform Kruskal-Wallis test for two variables across two groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    group1 : list
        List of identifiers for the first group.
    group2 : list
        List of identifiers for the second group.
    var : str
        Name of the variable for which the Kruskal-Wallis test is performed.

    Returns
    -------
    stat : float
        Kruskal-Wallis test statistic.
    p_value : float
        p-value.
    '''
    data_1 = data[data['Session Type'].isin(group1)][var]
    data_2 = data[data['Session Type'].isin(group2)][var]

    stat, p_value = kruskal(data_1, data_2)
    return stat, p_value

def calc_kruskal_two_var(data):
    '''
    Calculate and save Kruskal-Wallis test results for two variables across two groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    None
    '''
    # Perform test and collect results for the comparisons.
    res = {condition: {variable: kruskal_two_var(data, groups[0], groups[1], variable)
                    for variable in numeric_cols}
                    for condition, groups in two_var_groups.items()}

    # Save results.
    file_path = './stats/kruskal_two_var.txt'
    with open(file_path, 'w+') as file:
        for condition, variables in res.items():
            file.write(f"{condition}:\n")
            for variable, result in variables.items():
                file.write(f"  {variable}:\n")
                file.write(f"    Kruskal-Wallis statistic: {result[0]}\n")
                file.write(f"    p-value: {result[1]}\n")
            file.write("\n")
     
def compare_dists(data, categories, cat_col):
    '''
    Compare distributions of numeric variables across categories.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    categories : list
        List of categories to compare distributions.
    cat_col : str
        Name of the categorical column.

    Returns
    -------
    None
    '''
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    for i, var in enumerate(numeric_cols):
        row = i // 2
        col = i % 2

        for cat in categories:
            data = data[data[cat_col] == cat][var]
            sns.histplot(data=data, kde=True, stat='density', 
                         alpha=0.5, label=cat, ax=axes[row, col])

        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('Density') 
        axes[row, col].set_title(f'{var} Distribution by {cat_col}')

    fig.tight_layout()
    plt.legend()
    plt.savefig(f'{cat_col}_dists', dpi=500) 
    
def plot_dists(data):
    '''
    Plot distributions of variables across different categories.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    None
    '''
    # Define categories to analyze.
    robot_categories = ['One Robot', 'Two Robots']
    speed_categories = ['Fast', 'Slow', 'Mixed']
    focus_categories = ['Right Focus', 'Left Focus']

    # Call the function to generate plots.
    compare_dists(data, 'Robots', robot_categories)
    compare_dists(data, 'Speed', speed_categories)
    compare_dists(data, 'Orientation', focus_categories) 
    
def mw_one_var(data, cat_col, groups):
    '''
    Perform Mann-Whitney U test for one variable across different groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    cat_col : str
        Name of the categorical column.
    groups : dict
        Dictionary defining groups for analysis.

    Returns
    -------
    res : dict
        Dictionary containing results of the Mann-Whitney U test for each variable.
        Keys are variable names, and values are lists of tuples, each containing:
        (group1, group2, statistic, p-value, significant).
    '''
    
    res = {}
    # Calculate number of comparisons for Bonferroni correction.
    num_comparisons = len(list(combinations(groups, 2)))
    bonferroni_alpha = 0.05 / num_comparisons

    for col in numeric_cols:
        pairwise_results = []
        for group1, group2 in combinations(groups, 2):
            # Extract data for the two groups.
            data1 = data[data[cat_col] == group1][col]
            data2 = data[data[cat_col] == group2][col]

            # Perform Mann-Whitney U test.
            stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')

            # Check against Bonferroni-corrected alpha.
            significant = p_value < bonferroni_alpha
            pairwise_results.append((group1, group2, stat, p_value, significant))

        res[col] = pairwise_results
   
    return res

def calc_mw_one_var(data, categories):
    '''
    Calculate and save Mann-Whitney U test results for one variable across different groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    categories : dict
        Dictionary defining categories and their corresponding groups.

    Returns
    -------
    None
    '''

    # Set up dictionary for results.
    res = {}

    # Perform tests for each category.
    for category, groups in categories.items():
        category_res = mw_one_var(data, category, groups)
        res[category] = category_res
                
    res_text = ""
    for category, results in res.items():
        res_text += f"{category}:\n"
        for variable, var_res in results.items():
            res_text += f"  {variable}:\n"
            for group1, group2, stat, p_value, significant in var_res:
                print()
                res_text += f"    {group1} vs {group2}:\n"
                res_text += f"      Mann-Whitney U statistic: {stat}\n"
                res_text += f"      p-value: {p_value}\n\n"

    # Save the results to a text file.
    file_path = './stats/mw_one_var.txt'
    with open(file_path, 'w+') as file:
        file.write(res_text)

def mw_two_var(data, groups):
    '''
    Perform Mann-Whitney U test for two variables across two groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    groups : dict
        Dictionary defining two groups for comparison.

    Returns
    -------
    res : dict
        Dictionary containing results of the Mann-Whitney U test for each group.
        Keys are group names, and values are dictionaries containing results for each variable.
        Within each variable dictionary, keys are variable names, and values are dictionaries
        containing U-statistic and p-value.
    '''
    # Create dictionary for results.
    res = {}
    # Iterate over each group.
    for group_name, (group1, group2) in groups.items():
        # Extracting data for the two groups
        data1 = data[data['Session Type'].isin(group1)]
        data2 = data[data['Session Type'].isin(group2)]

        group_res = {}
        for col in numeric_cols:
            # Perform Mann-Whitney U test.
            stat, p_value = mannwhitneyu(data1[col], data2[col], alternative='two-sided')
            group_res[col] = {'U-statistic': stat, 'p-value': p_value}

        res[group_name] = group_res

    return res

def calc_mw_two_var(data):
    '''
    Calculate and save Mann-Whitney U test results for two variables across two groups.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    None
    '''
    res = mw_two_var(data, two_var_groups)

    # Save the results to a text file.
    res_text = ""
    for group, results in res.items():
        res_text += f"{group}:\n"
        for variable, result in results.items():
            res_text += f"  {variable}:\n"
            res_text += f"    Mann-Whitney U statistic: {result['U-statistic']}\n"
            res_text += f"    p-value: {result['p-value']}\n"
        res_text += "\n"

    file_path = './stats/mw_two_var.txt'
    with open(file_path, 'w+') as file:
        file.write(res_text)
        
def violin_plots(data):
    '''
    Generate violin plots for selected variables across different categories.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    None
    '''
    fig, axes = plt.subplots(2, 2, figsize=(15, 10)) # Creates a 2x2 grid of plots

    # Stress subplot
    sns.violinplot(x=data['Robots'], y=data['Stress'], ax=axes[0, 0], palette='rocket_r')
    sns.violinplot(x=data['Speed'], y=data['Stress'], ax=axes[0, 0], palette='rocket_r')
    sns.violinplot(x=data['Orientation'], y=data['Stress'], ax=axes[0, 0], palette='rocket_r')
    axes[0, 0].set_title('Stress Median and Interquartile Range')
    axes[0, 0].set_xlabel('')

    # Attention subplot
    sns.violinplot(x=data['Robots'], y=data['Attention'], ax=axes[0, 1], palette='rocket_r')
    sns.violinplot(x=data['Speed'], y=data['Attention'], ax=axes[0, 1], palette='rocket_r')
    sns.violinplot(x=data['Orientation'], y=data['Attention'], ax=axes[0, 1], palette='rocket_r')
    axes[0, 1].set_title('Attention Median and Interquartile Range')
    axes[0, 1].set_xlabel('')

    # Robot Predictability subplot
    sns.violinplot(x=data['Robots'], y=data['Robot Predictability'], ax=axes[1, 0], palette='rocket_r')
    sns.violinplot(x=data['Speed'], y=data['Robot Predictability'], ax=axes[1, 0], palette='rocket_r')
    sns.violinplot(x=data['Orientation'], y=data['Robot Predictability'], ax=axes[1, 0], palette='rocket_r')
    axes[1, 0].set_title('Robot Predictability Median and Interquartile Range')
    axes[1, 0].set_xlabel('')

    # Total Workload subplot
    sns.violinplot(x=data['Robots'], y=data['Total Workload'], ax=axes[1, 1], palette='rocket_r')
    sns.violinplot(x=data['Speed'], y=data['Total Workload'], ax=axes[1, 1], palette='rocket_r')
    sns.violinplot(x=data['Orientation'], y=data['Total Workload'], ax=axes[1, 1], palette='rocket_r')
    axes[1, 1].set_title('Total Workload Median and Interquartile Range')
    axes[1, 1].set_xlabel('')

    plt.tight_layout() # Adjusts the plots to prevent overlapping
    plt.savefig('./plots/violinplots', dpi=500)
    
def one_var_corrs(data):
    '''
    Compute and save correlations within categories.

    This function calculates correlations between variables within different categories,
    such as Robots, Speed, and Orientation, and saves the results to a text file.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.

    Returns
    -------
    robots : DataFrame
        Correlation matrix grouped by Robots.
    speed : DataFrame
        Correlation matrix grouped by Speed.
    orientation : DataFrame
        Correlation matrix grouped by Orientation.
    '''
    # Group data for correlation analysis.
    robots = data.groupby('Robots')[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()
    speed = data.groupby('Speed')[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()
    orientation = data.groupby('Orientation')[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()

    # Convert to strings.
    robots_str = robots.to_string()
    speed_str = speed.to_string()
    orientation_str = orientation.to_string()

    # Combine strings.
    comb_corr = "Correlations - Grouped by Robots:\n" + robots_str + "\n\n" + \
                "Correlations - Grouped by Speed:\n" + speed_str + "\n\n" + \
                "Correlations - Grouped by Orientation:\n" + orientation_str

    # Save to a text file.
    file_path = './stats/1_var_inner_corrs.txt'
    with open(file_path, 'w+') as file:
        file.write(comb_corr)
        
    return robots, speed, orientation
        
def two_var_corrs(data):
    '''
    Compute and save correlations between two variables.

    This function calculates correlations between variables based on two factors: Speed vs Robot Number
    and Speed vs Orientation. It saves the results to a text file.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    
    Returns
    -------
    None
    '''
    # Speed vs Robot number: one robot slow (ab) vs two robots slow(g) and one robot fast (cd) vs two robots fast (h)
    one_robot_slow = data[data['Session Type'].str.contains('[ab]')]
    two_robots_slow = data[data['Session Type'].str.contains('g')]
    one_robot_fast = data[data['Session Type'].str.contains('[cd]')]
    two_robots_fast = data[data['Session Type'].str.contains('h')]

    corr_one_robot_slow = one_robot_slow[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()
    corr_two_robots_slow = two_robots_slow[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()
    corr_one_robot_fast = one_robot_fast[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()
    corr_two_robots_fast = two_robots_fast[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()

    # Speed vs Orientation: right robot slow (ae) vs left robot slow (bf) and right robot fast (cf) vs left robot fast (de)
    right_robot_slow = data[data['Session Type'].str.contains('[ae]')]
    left_robot_slow = data[data['Session Type'].str.contains('[bf]')]
    right_robot_fast = data[data['Session Type'].str.contains('[cf]')]
    left_robot_fast = data[data['Session Type'].str.contains('[de]')]

    corr_right_robot_slow = right_robot_slow[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()
    corr_left_robot_slow = left_robot_slow[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()
    corr_right_robot_fast = right_robot_fast[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()
    corr_left_robot_fast = left_robot_fast[['Stress', 'Attention', 'Robot Predictability', 'Total Workload']].corr()

    # Convert each correlation table to a string and combine them.
    two_var_corr_str = "Speed vs Robot Number - One Robot Slow (ab) Sessions:\n" + corr_one_robot_slow.to_string() + "\n\n" + \
                    "Speed vs Robot Number - Two Robots Slow (g) Sessions:\n" + corr_two_robots_slow.to_string() + "\n\n" + \
                    "Speed vs Robot Number - One Robot Fast (cd) Sessions:\n" + corr_one_robot_fast.to_string() + "\n\n" + \
                    "Speed vs Robot Number - Two Robots Fast (h) Sessions:\n" + corr_two_robots_fast.to_string() + "\n\n" + \
                    "Speed vs Orientation - Right Robot Slow (ae) Sessions:\n" + corr_right_robot_slow.to_string() + "\n\n" + \
                    "Speed vs Orientation - Left Robot Slow (bf) Sessions:\n" + corr_left_robot_slow.to_string() + "\n\n" + \
                    "Speed vs Orientation - Right Robot Fast (cf) Sessions:\n" + corr_right_robot_fast.to_string() + "\n\n" + \
                    "Speed vs Orientation - Left Robot Fast (de) Sessions:\n" + corr_left_robot_fast.to_string()

    # Save to a text file.
    file_path = './stats/2_var_inner_corrs.txt'
    with open(file_path, 'w+') as file:
        file.write(two_var_corr_str)
            
def one_var_heatmaps(robots, speed, orientation):
    robots, speed, orientation = one_var_corrs(data)
    
    # Heatmap for one robot (abcd) vs two robots (efgh)
    corr_one_robot = robots.loc['One Robot']
    corr_two_robots = robots.loc['Two Robots']

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, 
                                   figsize=(16,7), 
                                   gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.1})

    ax1 = sns.heatmap(corr_one_robot, cmap='rocket_r', annot=True, ax=ax1, cbar=False)
    ax1.set_title('One Robot (abcd) Correlation')

    ax2 = sns.heatmap(corr_two_robots, cmap='rocket_r', annot=True, ax=ax2, cbar=False)
    ax2.set_title('Two Robot (efgh) Correlation')

    cbar_ax = fig.add_axes([0.91, 0.12, 0.03, 0.75])  # [x, y, width, height]
    sm = plt.cm.ScalarMappable(cmap='rocket_r', norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout()
    plt.savefig('./plots/correlations/one_v_two_robots', dpi=500)
    
    # Heatmap for fast sessions (cdh) vs slow sessions (abg) vs 1 fast 1 slow
    corr_fast = speed.loc['Fast']
    corr_slow = speed.loc['Slow']
    corr_mixed = speed.loc['Mixed']

    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True, 
                                        figsize=(22,7), 
                                        gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.1})

    ax1 = sns.heatmap(corr_fast, cmap='rocket_r', annot=True, ax=ax1, cbar=False)
    ax1.set_title('Fast (cdh) Correlation')

    ax2 = sns.heatmap(corr_slow, cmap='rocket_r', annot=True, ax=ax2, cbar=False)
    ax2.set_title('Slow (abg) Correlation')

    ax3 = sns.heatmap(corr_mixed, cmap='rocket_r', annot=True, ax=ax3, cbar=False)
    ax3.set_title('Mixed (ef) Correlation')

    cbar_ax = fig.add_axes([0.91, 0.12, 0.03, 0.75])  # [x, y, width, height]
    sm = plt.cm.ScalarMappable(cmap='rocket_r', norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout()
    plt.savefig('./plots/correlations/fast_v_slow_v_mixed', dpi=500)
    
    # Heatmap for right focus (acf) vs left focus (bde)
    corr_right = orientation.loc['Right Focus']
    corr_left = orientation.loc['Left Focus']

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, 
                                   figsize=(16,7), 
                                   gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.1})

    ax1 = sns.heatmap(corr_right, cmap='rocket_r', annot=True, ax=ax1, cbar=False)
    ax1.set_title('Right Focus (acf) Correlation')

    ax2 = sns.heatmap(corr_left, cmap='rocket_r', annot=True, ax=ax2, cbar=False)
    ax2.set_title('Left Focus (bde) Correlation')

    cbar_ax = fig.add_axes([0.91, 0.12, 0.03, 0.75])  # [x, y, width, height]
    sm = plt.cm.ScalarMappable(cmap='rocket_r', norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.tight_layout()
    plt.savefig('./plots/correlations/r_v_l_focus', dpi=500)

def cliff_delta(data1, data2):
    '''
    Compute Cliff's Delta between two sets of data.

    Parameters
    ----------
    data1 : array-like
        First set of data.
    data2 : array-like
        Second set of data.

    Returns
    -------
    float
        Cliff's Delta, a measure of effect size.

    Examples
    --------
    >>> data1 = [1, 2, 3, 4, 5]
    >>> data2 = [2, 3, 4, 5, 6]
    >>> cliff_delta(data1, data2)
    0.4
    '''
    n1, n2 = len(data1), len(data2)
    all_pairs = [(x, y) for x in data1 for y in data2]
    larger = sum(x > y for x, y in all_pairs)
    smaller = sum(x < y for x, y in all_pairs)
    delta = (larger - smaller) / (n1 * n2)
    return delta

def calc_cliff_delta(data, categories):
    '''
    Calculate Cliff's Delta for significant pairwise comparisons between groups within each category.

    Parameters
    ----------
    data : DataFrame
        Input DataFrame containing the data.
    categories : dict
        Dictionary containing category names as keys and lists of group names as values.

    Returns
    -------
    dict
        A dictionary containing Cliff's Delta values for significant pairwise comparisons between groups within each category.

    Examples
    --------
    >>> categories = {'Robots': ['One Robot', 'Two Robots'], 'Speed': ['Fast', 'Slow', 'Mixed']}
    >>> calc_cliff_delta(data, categories)
    {'Robots': {'Stress': [('One Robot', 'Two Robots', 0.3)], 'Attention': [('One Robot', 'Two Robots', 0.2)]},
     'Speed': {'Stress': [('Fast', 'Slow', -0.4)], 'Attention': [('Fast', 'Slow', -0.3)]}}
    '''
    res = {}

    # Iterate over each category and its groups
    for category, groups in categories.items():
        category_res = mw_one_var(data, category, groups)
        for var in numeric_cols:
            for result in category_res[var]:
                group1, group2, _, p_value, significant = result
                
                # Check for significance and calculate Cliff's Delta
                if significant:
                    data1 = data[data[category] == group1][var]
                    data2 = data[data[category] == group2][var]
                    delta = cliff_delta(data1, data2)
                    if category not in res:
                        res[category] = {}
                    if var not in res[category]:
                        res[category][var] = []
                    res[category][var].append((group1, group2, delta))
                    
    file_path = './stats/cliffs_delta.txt'
    with open(file_path, 'w+') as file:
        for category, variables in res.items():
            file.write(f"{category}:\n")
            for variable, groups in variables.items():
                file.write(f"  {variable}:\n")
                for group1, group2, delta in groups:
                    file.write(f"    {group1} vs {group2}: Cliff's Delta: {delta}\n")
            file.write("\n")

    return res

def main():
    calc_ks_one_var(data)
    calc_ks_two_var(data)
    calc_levene_one_var(data)
    calc_levene_two_var(data)
    calc_kruskal_one_var(data)
    calc_kruskal_two_var(data)
    calc_mw_one_var(data, categories)
    calc_mw_two_var(data)
    calc_cliff_delta(data, categories)

if __name__=='__main__':
    main()
