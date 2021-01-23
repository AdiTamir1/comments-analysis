import pandas as pd
import numpy as np


with open('./data/bad-words.txt') as bad_words_open_instance:
    bad_words_list = bad_words_open_instance.read().split('\n')
    bad_words_set = set(bad_words_list)


comments_df = pd.read_csv('./data/comments.csv', sep=",", encoding='Latin-1')
comments_df = comments_df[['textOriginal']]
comments_df.rename(columns={ "textOriginal": "comment" }, inplace=True)

comments_transformed_texts = []
comments_scores = []
acceptable_comments_count = 0
for index, comment_row in comments_df.iterrows():
    if comment_row['comment'] is np.nan:
        continue
    
    comment_text_str = str(comment_row['comment'])
    comment_words_list = comment_text_str.split()
    is_comment_contains_bad_word = False

    for comment_word in comment_words_list:
        if comment_word in bad_words_set:
            is_comment_contains_bad_word = True
            break

    comment_score = 1 if is_comment_contains_bad_word else 0
    
    if comment_score == 0:
        acceptable_comments_count += 1
        if acceptable_comments_count < 20000:
            comments_scores.append(comment_score)
            comments_transformed_texts.append(comment_text_str)
    else:
        comments_scores.append(comment_score)
        comments_transformed_texts.append(comment_text_str)


transformed_comments_df = pd.DataFrame(
    data={
        "comment": comments_transformed_texts,
        "unacceptable": comments_scores
    }
)

transformed_comments_df.to_csv('./data/transformed-comments.csv')