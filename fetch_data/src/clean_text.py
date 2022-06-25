import pandas as pd
import pickle
import re


def read_pickle(path):
    f = open(path, "rb")
    return pickle.load(f)


def trim_whitespace(s):
    '''Remove trailing and multiple whitespaces'''
    return s.replace('\s+', ' ', regex=True).str.strip()


def text_to_lower(s):
    return s.str.lower()


# See notebook for rationale
def trim_motion_text(row):
    '''Motion structure specific trimming to appy row wise to pandas DataFrame.
    See notebook for rationale. Requires text to be lower case'''

    def trim_motion_text_by_subtitle(row):
        split = row['text'].split(row['subtitle'], 1)
        return split[-1].strip()


    def trim_motion_text_by_leading_title(row):
        if row['text'].startswith(row['title']):
            text = row['text'].split(row['title'], 1)[-1].strip()
        else:
            text = row['text']
        return text


    def trim_motion_text_by_proposed_decision(row):
        split = re.split("förslag till riksdagsbeslut riksdagen [A-ö0-9\s,]+\.\s|" +\
                         "riksdagen tillkännager för regeringen som sin mening [A-ö0-9\s,]+\.\s|" +\
                         "riksdagen ställer sig bakom det som anförs [A-ö0-9\s,]+\.\s",
                         row['text'].lower())
        return split[-1].strip()


    row['text'] = trim_motion_text_by_subtitle(row)
    row['text'] = trim_motion_text_by_leading_title(row)
    row['text'] = trim_motion_text_by_proposed_decision(row)
    return row['text']




def main():

    #d = read_pickle('../data/docs_dict.pkl')
    #df = pd.DataFrame(d)
    with open('../data/docs_dict.pkl', "rb") as input_file:
        df = pd.DataFrame(pickle.load(input_file))

    for col in df.select_dtypes(include='object').columns:
        df[col] = trim_whitespace(df[col])
        df[col] = text_to_lower(df[col])

        if col == 'text':
            df[col] = df.apply(trim_motion_text, axis=1)

    df.to_feather('../data/clean_text.feather')
