import pandas as pd
import numpy as np
import spacy
import textstat
from sentence_transformers import SentenceTransformer, util
import statsmodels.api as sm
import seaborn as sns
import scipy.stats

nlp = spacy.load('en_core_web_md')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


def add_surprisal(cleaned_data: pd.DataFrame, aligned_sentences: pd.DataFrame)-> pd.DataFrame:
    """
    adding avg surprisal to sentence alignment
    :param data: cleaned data
    :return:
    """
    alignment_with_surprisal = (align_with_measures(cleaned_data, ['gpt2_Surprisal'], True).
                                rename(columns={"paragraph_id": "text_id"}))
    return pd.merge(aligned_sentences, alignment_with_surprisal, on=['text_id', 'sentence_id'], how='inner')

def add_spacy_similarity(row):
    ele_sen = row["Text Ele Sentence"]
    adv_sen = row["Text Adv Sentence"]

    if pd.isna(ele_sen) or pd.isna(adv_sen):
        return np.nan

    ele_sen_trans = nlp(ele_sen)
    adv_sen_trans = nlp(adv_sen)

    return ele_sen_trans.similarity(adv_sen_trans)


def add_flesch_reading_ease_difference(row):
    ele_sen = row["Text Ele Sentence"]
    adv_sen = row["Text Adv Sentence"]

    if pd.isna(ele_sen) or pd.isna(adv_sen):
        return np.nan

    return (textstat.flesch_reading_ease(ele_sen) -
            textstat.flesch_reading_ease(adv_sen))


def add_sentence_bert_similarity(row):
    ele_sen = row["Text Ele Sentence"]
    adv_sen = row["Text Adv Sentence"]

    if pd.isna(ele_sen) or pd.isna(adv_sen):
        return np.nan

    ele_embeddings = bert_model.encode(ele_sen, convert_to_tensor=True)
    adv_embeddings2 = bert_model.encode(adv_sen, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(ele_embeddings, adv_embeddings2).item()
    return cosine_sim


def add_stat_column_to_aligned(cleaned_data, stats_dict, sentence_alignment_path):
    """
    Adding all readability measures in dict to aligned data
    :param cleaned_data: data 
    """
    aligned_sentences = pd.read_csv(sentene_alignment_path)
    for name, func in stats_dict.items():
        aligned_sentences[name] = aligned_sentences.apply(func, axis=1)
    add_surprisal(cleaned_data, aligned_sentences)
    # aligned_sentences.to_csv("aligned_readabilty_measures.csv")
    return aligned_sentences

