import pandas as pd
import numpy as np


def align_with_measures(cleaned_data: pd.DataFrame, measures: list,
                        on_sentence_level: bool) -> pd.DataFrame:
    """
    Aligning data of elementary and advanced reading levels
    so that the matching (sentences\paragraphs) in each level will be in the same line
    :param cleaned_data: cleaned data (with sentence id)
    :param measures: measures we want to find the mean of on wanted level
    :param on_sentence_level: True if the alignment is on sentence level False if on paragraph level
    :return: dataframe of aligned sentences with eyes measures
    """
    word_measures = cleaned_data[['paragraph_id', 'sentence_id', 'IA_ID', 'is_adv', *measures]]

    if on_sentence_level:
        mean_on = ['paragraph_id', 'sentence_id']
    else:
        mean_on = ['paragraph_id']

    measure_means_on_subject = (word_measures.groupby([*mean_on, 'IA_ID', 'is_adv'])[measures].
                                mean().reset_index())

    measure_means_on_words = (measure_means_on_subject.groupby([*mean_on, 'is_adv'])[measures].
                              mean().reset_index())

    adv_measure_means = measure_means_on_words.loc[(measure_means_on_words['is_adv'] == 1)]
    ele_measure_means = measure_means_on_words.loc[(measure_means_on_words['is_adv'] == 0)]

    return (pd.merge(ele_measure_means, adv_measure_means, on=mean_on, how='inner', suffixes=('_ele', '_adv')).
            reset_index(drop=True).drop(columns=["is_adv_adv", "is_adv_ele"]))


def align_with_constrained_measures(cleaned_data: pd.DataFrame, measure: str, on_sentence_level: bool) -> pd.DataFrame:
    """
    aligning measures with constrains
    :param cleaned_data: data
    :param measure: constrained measure
    :param on_sentence_level: if True, on sentence level, if false on paragraph level
    :return: df of data with measures aggregated on sentence/paragraph level
    """

    if measure in ['IA_REGRESSION_OUT_COUNT', 'IA_REGRESSION_OUT_FULL_COUNT']:
        filtered_data = cleaned_data.loc[(cleaned_data['IA_FIRST_FIX_PROGRESSIVE'] == 1)]
    else:
        filtered_data = cleaned_data

    return align_with_measures(filtered_data, [measure], on_sentence_level)


def combine_measures(clean_data: pd.DataFrame, measures: list, constrained_measures: list, aligned_readability_path):
    """
    combining eye measures and aligned readability measures data
    :param clean_data:
    :param measures:
    :param constrained_measures:
    :return: aligned aggregated data with all measures (eye movements and readability)
    """
    data_with_measures = align_with_measures(clean_data, measures, on_sentence_level=True)
    data_with_readability = pd.read_csv(aligned_readability_path)
    data_with_readability.rename(columns={'text_id': 'paragraph_id'}, inplace=True)

    constrained_measures_dfs = [align_with_constrained_measures(clean_data, r_measure, True) for r_measure in
                                constrained_measures]
    combined = data_with_measures.merge(data_with_readability, on=['paragraph_id', 'sentence_id'], how='inner')
    for constrained_df in constrained_measures_dfs:
        combined = combined.merge(constrained_df, on=['paragraph_id', 'sentence_id'], how='inner')

    return combined


def main():
    aligned_readability_path = 'aligned_readability_measures.csv'
    measures = ['IA_DWELL_TIME', 'IA_SKIP']
    constrained_measures = ['IA_REGRESSION_OUT_COUNT', 'IA_REGRESSION_OUT_FULL_COUNT']
    cleaned = pd.read_csv('cleaned_data.csv')
    combine_measures(cleaned, measures, constrained_measures, aligned_readability_path).to_csv("aligned_all_measures")


if __name__ == '__main__':
    main()
