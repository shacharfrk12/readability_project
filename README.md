# readability project

This repositort contains code used for a text readability and eye-movement project.

# File description
1.prepare_cleaned_data.py - contains the finctions used to take our raw data and transform it into useable, filtered data for our analysis.

2.basic_statistical_analysis_utils.py - code for measuring basic statistics about out cleaned data. includes box plot functions, etc.

3.agg_eye_movement_measures_utils.py - this file contains the code for aggregating token-level information such as TF for a token for a specific participant. The aggregation is an average across all participants for a word in the corpus. the stats are then averaged over sentences/paragraphs depending on input.

4.add_readability_measures_utils.py - Takes a given dataset of aligned advanced/elementry sentence pairs and adds difference in several text readability measures to each pair. Notice, the relevant packages must be downloaded before use.

5.advanced_analysis_utils.py - this file needs access to the aligned sentences file with the aggregated eye-movement measures and the text readability measure. This file contains analysis functions such as calcuating and ploting correlation of eye-movement measures and readability measures and basic linear regression models.
