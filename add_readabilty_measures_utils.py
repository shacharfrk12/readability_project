import pandas as pd
import numpy as np
import spacy
import textstat
from sentence_transformers import SentenceTransformer, util
from agg_eye_movement_measures_final import align_with_measures

nlp = spacy.load('en_core_web_md')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


def add_surprisal(cleaned_data: pd.DataFrame, aligned_sentences: pd.DataFrame)-> pd.DataFrame:
    """
    adding avg surprisal to sentence alignment
    :param data: cleaned data
    :return: aligned data on sentences with aggregated surprisal difference
    """
    alignment_with_surprisal = (align_with_measures(cleaned_data, ['gpt2_Surprisal'], True).
                                rename(columns={"paragraph_id": "text_id"}))
    alignment_with_surprisal['surprisal_diff'] = alignment_with_surprisal['gpt2_Surprisal_adv'] - alignment_with_surprisal['gpt2_Surprisal_ele']
    alignment_with_surprisal = alignment_with_surprisal[['surprisal_diff', 'text_id', 'sentence_id']]
    return pd.merge(aligned_sentences, alignment_with_surprisal, on=['text_id', 'sentence_id'], how='inner')


def add_spacy_similarity(row):
    ele_sen = row["Text Ele Sentence"]
    adv_sen = row["Text Adv Sentence"]

    if pd.isna(ele_sen) or pd.isna(adv_sen):
        return np.nan

    ele_sen_trans = nlp(ele_sen)
    adv_sen_trans = nlp(adv_sen)

    return ele_sen_trans.similarity(adv_sen_trans)


def add_textats_measures(row):
    ele_sen = row["Text Ele Sentence"]
    adv_sen = row["Text Adv Sentence"]

    if pd.isna(ele_sen) or pd.isna(adv_sen):
        return np.nan

    flesch_reading_ease = textstat.flesch_reading_ease(ele_sen) - textstat.flesch_reading_ease(adv_sen)
    flesch_kincaid_grade_score = textstat.flesch_kincaid_grade(adv_sen) - textstat.flesch_kincaid_grade(ele_sen)
    gunning_fog = textstat.gunning_fog(adv_sen) - textstat.gunning_fog(ele_sen)
    coleman_liau = textstat.coleman_liau_index(adv_sen) - textstat.coleman_liau_index(ele_sen)
    smog_index = textstat.smog_index(adv_sen) - textstat.smog_index(ele_sen)
    dale_chall = textstat.dale_chall_readability_score(adv_sen) - textstat.dale_chall_readability_score(ele_sen)

    return flesch_reading_ease, flesch_kincaid_grade_score, gunning_fog, coleman_liau, smog_index, dale_chall


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
    aligned_sentences = pd.read_csv(sentence_alignment_path)
    aligned_sentences = aligned_sentences[["text_id","sentence_id","Text Ele Sentence","Text Adv Sentence"]]
    for name, func in stats_dict.items():
        aligned_sentences[name] = aligned_sentences.apply(func, axis=1)

    for column in aligned_sentences.columns:
        if isinstance(column, tuple):
            tuple_df = aligned_sentences[column].apply(pd.Series)
            tuple_df.columns = list(column)
            aligned_sentences = pd.concat([aligned_sentences.drop(columns=column), tuple_df], axis=1)

    aligned_sentences = add_surprisal(cleaned_data, aligned_sentences)
    aligned_sentences.to_csv("aligned_readability_measures.csv", index=False)


cleaned = pd.read_csv("cleaned_data.csv")
sentence_alignment_path = 'aligned.csv'
stats_dict = metrics_dict = {
    "spacy_similarity": add_spacy_similarity,
    "sentence_bert_similarity": add_sentence_bert_similarity,
    ("flesch_reading_ease_diff",
    "flesch_kincaid_grade_score_diff",
    "gunning_fog_index_diff",
    "coleman_liau_index_diff",
    "smog_index_diff",
    "dale_chall_score_diff"): add_textats_measures
}


def main():
    add_stat_column_to_aligned(cleaned, stats_dict, sentence_alignment_path)


if __name__ == "__main__":
    main()
