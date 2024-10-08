import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import spacy
import scipy.stats
import statsmodels.api as sm

font = {'size' : 15}

plt.rc('font', **font)

def box_plot_paragraph_df(data: pd.DataFrame, measure: str, measure_name):
    """
    plotting box plots of measures along data
    :param data: aligned paragraph or senence aggregated data with aggreagated measures 
                - each measure as two appearances: on elementry\advanced section.
    :param measure: eye movement measure in the table we want to present
    :param measure_name: name of eye movement measure
    """
  
    # dividing data to elementry and advanced
    adv_avgs = data[measure + '_adv'].values
    ele_avgs = data[measure + '_ele'].values

    # finding medians
    avgs = [ele_avgs, adv_avgs]
    medians = [round(np.median(ele_avgs), 3), round(np.median(adv_avgs), 3)]
    vertical_offset = 0.05

    # plotting
    box_plot = sns.boxplot(avgs)


    # defining ticks
    for xtick in box_plot.get_xticks():
        box_plot.text(xtick, medians[xtick], medians[xtick],
                      horizontalalignment='center', size='small', color='w', weight='semibold')

    # setting labels
    box_plot.set(xlabel='Reading level', ylabel=f'avg {measure_name}', title=f'Avg {measure_name} \n('
                                                                        'Advanced\Elementary Reading Level)')
    plt.xticks([0, 1], ['ele', 'adv'])

def box_plot_diff(data: pd, measure, measure_name):
    """
    plotting box plots of differences in measures
    :param data: aligned paragraph or senence aggregated data with aggreagated measures 
                - each measure as two appearances: on elementry\advanced section.
    :param measure: eye movement measure in the table we want to present
    :param measure_name: name of eye movement measure
    """
    # calculating differences
    data['diff'] = data.apply(lambda row: row[measure + '_adv'] - row[measure + '_ele'], axis=1)
    diff_avg = data['diff'].values

    # plotting
    median = round(np.median(diff_avg), 3)
    box_plot = sns.boxplot(diff_avg)
    box_plot.text(box_plot.get_xticks()[0], 1.05*median, median,
                  horizontalalignment='center', size='small', color='w', weight='semibold')
    box_plot.set(title=f'Difference of Avg {measure_name} \nin Advanced vs Elementary Reading Levels')


def plot_boxes_measures(data, measure_list, measure_names):
    """
    plotting box plots of measures
    :param data: aligned paragraph or senence aggregated data with aggreagated measures 
                - each measure as two appearances: on elementry\advanced section.
    :param measure_list: list of measures we want to present
    :param measure_names: list of measures names
    """

    # defining columns
    ele_cols = [measure + '_ele' for measure in measure_list]
    adv_cols = [measure + '_adv' for measure in measure_list]
    columns = []
    
    # alternating columns
    for i in range(len(measure_list)):
        columns.append(ele_cols[i])
        columns.append(adv_cols[i])

    # getting statistics
    stats = data[columns].agg(['min', 'max', 'mean', 'median', 'std'])
    print(f"Stats in advanced and Elementary Reading levels:")
    print(stats)

    # plotting each measure
    for measure, measure_name in zip(measure_list, measure_names):
        print(f"Plots for {measure}...\n")
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        box_plot_paragraph_df(data, measure, measure_name)
        plt.subplot(1, 2, 2)
        box_plot_diff(data, measure, measure_name)
        plt.tight_layout(pad=5.0)
        plt.show()
