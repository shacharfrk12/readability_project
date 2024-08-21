import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# box plots 

def box_plot_readability(data: pd.DataFrame, measure, ax):
    readability = data[measure].values

    medians = [round(np.median(readability), 3)]
    vertical_offset = 0.05

    box_plot = sns.boxplot(readability, ax=ax)

    for xtick in box_plot.get_xticks():
        box_plot.text(xtick, medians[xtick], medians[xtick],
                      horizontalalignment='center', size='small', color='w', weight='semibold')

    box_plot.set(title=f' {measure}')


def plot_boxes_readability(data, measure_list):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i,measure in enumerate(measure_list):
        plt.figure(figsize=(20,4))
        box_plot_readability(data, measure, axes[i])
    plt.tight_layout(pad=5.0)
    plt.show()


def get_lin_stats(data: pd.DataFrame, eye_measure:str, readability_measure:str):
    #eye_measure_diff = data[eye_measure+'_adv'] - data[eye_measure+'_ele']
    eye_measure_diff = np.array(data[eye_measure + '_adv']) - np.array(data[eye_measure + '_ele'])
    eye_measure_diff = sm.add_constant(eye_measure_diff)
    diff_readability = data[readability_measure]
    lin_model = sm.OLS(diff_readability,eye_measure_diff, missing='drop').fit()

    return lin_model.summary()

#bar plots
def plot_correclations(aligned_data:pd.DataFrame, measures:list, readability_measure:str, correlation_func = scipy.stats.pearsonr):
    aligned_data = aligned_data.dropna()
    diff_readability = aligned_data[readability_measure].to_numpy()
    corr_list = []
    r_list = []
    for eye_measure in measures:
        eye_measure_diff = np.array(aligned_data[eye_measure + '_adv']) - np.array(aligned_data[eye_measure + '_ele'])
        #eye_measure_diff = aligned_data.apply(lambda row: row[eye_measure + '_ele'] - row[eye_measure + '_adv'],
        # axis=1)

        corr, r = correlation_func(diff_readability, eye_measure_diff)
        corr_list.append(corr)
        r_list.append(r)
    print(corr_list)
    plt.bar(range(len(corr_list)), corr_list, tick_label=measures)
    plt.show()


# relationships
def plot_scatter(data: pd.DataFrame, eye_measure:str, readability_measure:str):
    eye_measure_diff = np.array(data[eye_measure + '_adv']) - np.array(data[eye_measure + '_ele'])
    #diff_readability = data[readability_measure]
    data[eye_measure+"_diff"] = eye_measure_diff
    #present_df = pd.DataFrame()
    #present_df['measure'] = eye_measure_diff
    #present_df['readability'] = diff_readability
    sns.regplot(x=eye_measure+"_diff", y=readability_measure, data=data, x_bins=20)
    plt.show()

    g = sns.jointplot(x=eye_measure+"_diff", y=readability_measure, data=data, kind='reg')
    # We're going to make the regression line red so it's easier to see
    regline = g.ax_joint.get_lines()[0]
    _ = regline.set_color('red')
    plt.show()



