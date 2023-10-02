
from pandas import read_excel, ExcelFile
from sklearn.datasets import load_breast_cancer
import numpy as np

import mlrun

SHFAQ = 'FAQ_unite'
SHCOMP = 'Parole_Composte'
SHSYN = 'Sinonimi'
SHSTOP = 'Stopwords'
SHMODEL = 'Model_def'
SHLIST = [SHFAQ, SHCOMP, SHSYN, SHSTOP, SHMODEL]

def correctNaN(df):

    if np.isnan(df['n_cluster'].get(0)):
        df['n_cluster'] = 0

    if np.isnan(df['n_topics'].get(0)):
        df['n_topics'] = 0

    if not (type(df['wordvec_path'].get(0)) is str):
        if np.isnan(df['wordvec_path'].get(0)):
            df['wordvec_path'] = ''

    if np.isnan(df['bot_version'].get(0)):
        df['bot_version'] = 0
    return(df)

@mlrun.handler(outputs=["faq_knowledge_base", "faq_stopwords", "faq_synonyms", "faq_compounds"])
def parse_faq_excel():    
    """
    A function which generates the FAQ datasets from Excel file
    """
    path = 'faq/faq.xlsx'
    df = read_excel(path, SHFAQ, converters={'question_number': str}).replace(np.nan, '', regex=True)
    dfComp = read_excel(path, SHCOMP).replace(np.nan, '', regex=True)
    dfSyn = read_excel(path, SHSYN).replace(np.nan, '', regex=True)
    dfStop = read_excel(path, SHSTOP).replace(np.nan, '', regex=True)
    dfModel = read_excel(path, SHMODEL)
    dfModel = correctNaN(dfModel)

    return df, dfStop, dfSyn, dfComp
