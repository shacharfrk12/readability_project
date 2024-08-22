import pandas as pd
import numpy as np

columns_to_keep = [
    'IA_FIRST_FIX_PROGRESSIVE',
    "IA_DWELL_TIME",
    "IA_ID",
    "IA_LABEL",
    "IA_REGRESSION_OUT_COUNT",
    "IA_REGRESSION_OUT_FULL_COUNT",
    "IA_SKIP",
    "has_preview",
    "reread",
    "unique_paragraph_id",
    "subject_id",
    "gpt2_Surprisal",
]


def separate_adv(data: pd.DataFrame):
    """
    The original data contains the paragrah ID in a x_x_adv_x format. This function splits this column into a binary
    'is_Adv' column and a paragraph id column.
    :param data: pandas df containing the eye tracking data.
    :return: the given dataframe with the "unique_paragraph_id" column replaced with a "is_adv" column and a
    "paragraph_id" column.
    """
    adv_condition = lambda x: 1 if "Adv" in x else 0
    id_condition = lambda x: x.replace("_Adv", "").replace("_Ele", "")
    data["is_adv"] = data["unique_paragraph_id"].apply(adv_condition)
    data["paragraph_id"] = data["unique_paragraph_id"].apply(id_condition)
    return data.drop(columns=['unique_paragraph_id'])

def clean_data(org_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Original data in order to be fit for our specific analysis. This function uses the drop_irrelevant_cols()
    and the separate_adv() functions as well as keeps only entries relevant to "gathering" and first time reading.
    :param org_data: a df containing the original eye tracking data.
    :return: a df of the cleaned data.
    """
    cleaned_data = org_data[columns_to_keep]
    cleaned_data = cleaned_data[cleaned_data['has_preview'] == 'Gathering']
    cleaned_data = cleaned_data[cleaned_data['reread'] == 0]
    cleaned_data = separate_adv(cleaned_data)
    return cleaned_data

def sentence_word_alignment(data: pd.DataFrame, aligned_path):
    """
    Fits each entry in the eye_tracking dataframe with a sentence ID as specifies in the aligned.csv file.
    The sentence ID is unique per paragraph (the ID is the sentence's index in the paragraph). When combined with the
    paragraph ID, this combination is unique for each sentence in the corpus.
    :param data: a pandas df containing the cleaned eye tracking data
    :return: the given pandas df with an added column of "sentence_id".
    """
    aligned_sentences = pd.read_csv(aligned_path)
    aligned_adv = aligned_sentences[['text_id', 'sentence_id', 'Text Adv Sentence']]
    aligned_adv['word'] = aligned_adv['Text Adv Sentence'].str.split()
    aligned_adv = aligned_adv.explode('word').reset_index(drop=True)
    aligned_adv = aligned_adv[['text_id', 'sentence_id', 'word']]
    aligned_adv = aligned_adv.dropna(subset=['text_id'])
    aligned_adv['word_index'] = aligned_adv.groupby('text_id').cumcount()
    aligned_adv['word_index'] = aligned_adv['word_index'].astype(int)

    aligned_ele = aligned_sentences[['text_id', 'sentence_id', 'Text Ele Sentence']]
    aligned_ele['word'] = aligned_ele['Text Ele Sentence'].str.split()
    aligned_ele = aligned_ele.explode('word').reset_index(drop=True)
    aligned_ele = aligned_ele[['text_id', 'sentence_id', 'word']]
    aligned_ele = aligned_ele.dropna(subset=['text_id', 'word'])
    aligned_ele['word_index'] = aligned_ele.groupby('text_id').cumcount()
    aligned_ele['word_index'] = aligned_ele['word_index'].astype(int)

    aligned_adv['is_adv'] = 1
    aligned_ele['is_adv'] = 0

    aligned_adv.rename(columns={'text_id': 'paragraph_id', 'word_index': 'IA_ID', 'word': 'IA_LABEL'}, inplace=True)
    aligned_ele.rename(columns={'text_id': 'paragraph_id', 'word_index': 'IA_ID', 'word': 'IA_LABEL'}, inplace=True)

    aligned_words = pd.concat([aligned_adv, aligned_ele], axis=0)

    return (pd.merge(data, aligned_words, on=['paragraph_id', 'IA_ID', 'is_adv', 'IA_LABEL'], how='inner').
            reset_index(drop=True))


def main():
    data_path = 'ia_data_enriched_360_05052024.csv'
    org_data = pd.read_csv(data_path)[columns_to_keep]
    cleaned_data = clean_data(org_data)
    sentence_word_alignment(cleaned_data, "aligned.csv").drop(
        columns=['reread', 'has_preview']).to_csv("cleaned_data.csv", index=False)


if __name__ == '__main__':
    main()
