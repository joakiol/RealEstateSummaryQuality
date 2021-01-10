from snorkel.labeling import labeling_function
from utils import score_readability
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

GOOD = 1
BAD = 0
ABSTAIN = -1
MATCHED_POLICIES = pd.read_csv("data/VenduData/matched_policies.zip")
CLAIMS = pd.read_csv('data/VenduData/claims.csv')

def prepare_data_for_labeling_functions(data):
    """
    Make a dataframe that has all the relevant info for labeling functions,
    such that they can be called much more efficiently on a prepared dataset. 

    :return: Pandas df with all necessary info for labeling functions
    """
    print("Extracting relevant info for labeling functions from dataset...")
    # Initialize lists for relevant info
    keys = []
    summary_length = []
    mention_kitchen = []
    mention_bath = []
    mention_roof = []
    TG2_kitchen = []
    TG2_bath = []
    TG2_roof = []
    TG3_kitchen = []
    TG3_bath = []
    TG3_roof = []
    adjusted_kitchen = []
    adjusted_bath = []
    adjusted_roof = []
    summary_liks = []
    summary_ovr = []
    common_words_div_summary = []
    common_words_div_report = []
    author = []
    year_list = []

    # Additional parameters
    kitchen_words = {'kjøkken', 'kjøkkenet', 'kjøkkeninnredning', 'kjøkkeninnredningen', 'kjøkkenskap', 'kjøkkenskapet', 'kjøkkenskapene'}
    bath_words = {'bad', 'badet', 'badene', 'baderom', 'baderommet', 'baderommene', 'våtrom', 'våtrommet', 'våtrommene', 'wc', 'baderomsgulv', 'baderomsgulvet', 'baderomsgulvene', 'baderomsvegg', 'baderomsveggen', 'baderomsvegger', 'baderomsveggene'}
    roof_words = {'tak', 'taket', 'tekking', 'tekket',  'taktekket', 'taktekking', 'takkonstruksjon', 'takkonstruksjoner', 'pipe', 'saltak', 'saltaket', 'takstoler', 'takstol', 'takstolene', 'takstein', 'skifertakstein', 'flattak', 'undertak', 'undertaket', 'yttertak', 'yttertaket', 'yttertaktekking', 'yttertaktekkingen'}
    stopwords_list = stopwords.words('norwegian')


    for report in data:

        # Get necessary report info:
        report_words = report.get_report_words()
        summary_words = report.get_summary_words()
        summary_sentences = report.get_summary_sentences()
        report_set = set(report_words)
        report_special_set = set([word for word in report_words if word not in stopwords_list])
        summary_set = set(summary_words)
        summary_special_set = set([word for word in summary_words if word not in stopwords_list])
        liks, ovr = score_readability(summary_sentences, summary_words)
        common_special_words = report_special_set.intersection(summary_special_set)
        year = None
        if report.date != 'None':
            year = int(report.date.split('-')[0])

        
        # Initialize boolean values
        curr_TG2_kitchen = False
        curr_TG2_bath = False
        curr_TG2_roof = False
        curr_TG3_kitchen = False
        curr_TG3_bath = False
        curr_TG3_roof = False
        curr_adjusted_kitchen = False
        curr_adjusted_bath = False
        curr_adjusted_roof = False

        # Set boolean values
        for element in report.condition:
            if (element.room == 6 or element.room == 7) and (element.degree == 2):
                curr_TG2_roof = True
            if (element.room == 14) and (element.degree == 2):
                curr_TG2_bath = True
            if (element.room == 18) and (element.degree == 2):
                curr_TG2_kitchen = True
            if (element.room == 6 or element.room == 7) and (element.degree == 3):
                curr_TG3_roof = True
            if (element.room == 14) and (element.degree == 3):
                curr_TG3_bath = True
            if (element.room == 18) and (element.degree == 3):
                curr_TG3_kitchen = True
            if (element.room == 6 or element.room == 7) and (element.adjusted == True):
                curr_adjusted_roof = True
            if (element.room == 14) and (element.adjusted == True):
                curr_adjusted_bath = True
            if (element.room == 18) and (element.adjusted == True):
                curr_adjusted_kitchen = True
        

        # Append relevant info to respective lists
        keys.append(report.id)
        summary_length.append(len(summary_words))
        mention_kitchen.append(len(kitchen_words.intersection(summary_set)) > 0)
        mention_bath.append(len(bath_words.intersection(summary_set)) > 0)
        mention_roof.append(len(roof_words.intersection(summary_set)) > 0)
        TG2_kitchen.append(curr_TG2_kitchen)
        TG2_bath.append(curr_TG2_bath)
        TG2_roof.append(curr_TG2_roof)
        TG3_kitchen.append(curr_TG3_kitchen)
        TG3_bath.append(curr_TG3_bath)
        TG3_roof.append(curr_TG3_roof)
        adjusted_kitchen.append(curr_adjusted_kitchen)
        adjusted_bath.append(curr_adjusted_bath)
        adjusted_roof.append(curr_adjusted_roof)
        summary_liks.append(liks)
        summary_ovr.append(ovr)
        common_words_div_summary.append(len(common_special_words)/max(len(summary_special_set), 1))
        common_words_div_report.append(len(common_special_words)/max(len(report_special_set), 1))
        author.append(report.author)
        year_list.append(year)

    df = pd.DataFrame({'id': keys, 
                       'summary_length': summary_length, 
                       'mention_kitchen': mention_kitchen, 
                       'mention_bath': mention_bath, 
                       'mention_roof': mention_roof, 
                       'TG2_kitchen': TG2_kitchen, 
                       'TG2_bath': TG2_bath, 
                       'TG2_roof': TG2_roof, 
                       'TG3_kitchen': TG3_kitchen, 
                       'TG3_bath': TG3_bath, 
                       'TG3_roof': TG3_roof, 
                       'adjusted_kitchen': adjusted_kitchen, 
                       'adjusted_bath': adjusted_bath, 
                       'adjusted_roof': adjusted_roof, 
                       'summary_liks': summary_liks, 
                       'summary_ovr': summary_ovr, 
                       'common_words_div_summary': common_words_div_summary, 
                       'common_words_div_report': common_words_div_report, 
                       'author': author, 
                       'year': year_list})
    print("Done!")
    
    print("Adding claim data...")
    policy_data = MATCHED_POLICIES
    policy_data['id'] = policy_data['ts_report_id']
    policy_data = policy_data[['id', 'Polisenr']]

    claims = CLAIMS
    claims['has_claim'] = True
    claims = claims[['Polisenr', 'has_claim']]
    policy_data = policy_data.merge(claims, on='Polisenr', how='left')
    policy_data = policy_data[['id', 'has_claim']].groupby(by='id').max().reset_index()

    df = df.merge(policy_data, on='id', how='left')
    df['has_claim'] = df['has_claim'].fillna(False)
    print("Done!")

    print("Adding extra author information...")
    num_per_year = df[['id', 'year', 'author']].groupby(['author', 'year']).size().reset_index(name='reports_per_year_author')
    num_total = df[['id', 'author']].groupby('author').size().reset_index(name='reports_per_author')
    claims = df[df['has_claim'] == True][['id', 'author']].groupby('author').size().reset_index(name='num_with_claim')
    liks = df[df['summary_liks'] > 55][['id', 'author']].groupby('author').size().reset_index(name='num_with_high_liks')
    ovr = df[df['summary_ovr'] > 96][['id', 'author']].groupby('author').size().reset_index(name='num_with_high_ovr')

    df = df.merge(num_total, on='author', how='left')
    df = df.merge(num_per_year, on=['author', 'year'], how='left')
    df = df.merge(claims, on='author', how='left')
    df = df.merge(liks, on='author', how='left')
    df = df.merge(ovr, on='author', how='left')

    df[['reports_per_year_author', 'reports_per_author', 'num_with_claim', 'num_with_high_liks', 'num_with_high_ovr']] = df[['reports_per_year_author', 'reports_per_author', 'num_with_claim', 'num_with_high_liks', 'num_with_high_ovr']].fillna(0)
    print("Done!")

    return df




@labeling_function()
def too_short(x):
    if x['summary_length'] < 50:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def too_long(x):
    if x['summary_length'] > 400:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def TG3_bath_no_mention(x):
    if x['TG3_bath'] and x['mention_bath'] == False:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def TG3_kitchen_no_mention(x):
    if x['TG3_kitchen'] and x['mention_kitchen'] == False:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def TG3_roof_no_mention(x):
    if x['TG3_roof'] and x['mention_roof'] == False:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def TG2or3_bath_mention(x):
    if (x['TG2_bath'] or x['TG3_bath']) and x['mention_bath']:
        return GOOD
    else:
        return ABSTAIN

@labeling_function()
def TG2or3_kitchen_mention(x):
    if (x['TG2_kitchen'] or x['TG3_kitchen']) and x['mention_kitchen']:
        return GOOD
    else:
        return ABSTAIN

@labeling_function()
def TG2or3_roof_mention(x):
    if (x['TG2_roof'] or x['TG3_roof']) and x['mention_roof']:
        return GOOD
    else:
        return ABSTAIN

@labeling_function()
def TG_adjusted_bath_no_mention(x):
    if x['adjusted_bath'] and x['mention_bath'] == False:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def TG_adjusted_kitchen_no_mention(x):
    if x['adjusted_kitchen'] and x['mention_kitchen'] == False:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def TG_adjusted_roof_no_mention(x):
    if x['adjusted_roof'] and x['mention_roof'] == False:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def high_liks(x):
    if x['summary_liks'] > 55:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def high_ovr(x):
    if x['summary_ovr'] > 96:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def has_claim(x):
    if x['has_claim']:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def author_many_claims(x):
    if x['num_with_claim']/x['reports_per_author'] > 0.075:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def author_many_high_liks(x):
    if x['num_with_high_liks']/x['reports_per_author'] > 0.4:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def author_many_high_ovr(x):
    if x['num_with_high_ovr']/x['reports_per_author'] > 0.4:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def author_few_reports(x):
    if x['reports_per_year_author'] < 10:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def common_words_by_summary_low(x):
    if x['common_words_div_summary'] < 0.2:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def common_words_by_summary_high(x):
    if x['common_words_div_summary'] > 0.7:
        return GOOD
    else:
        return ABSTAIN

@labeling_function()
def common_words_by_report_low(x):
    if x['common_words_div_report'] < 0.03:
        return BAD
    else:
        return ABSTAIN

@labeling_function()
def common_words_by_report_high(x):
    if x['common_words_div_report'] > 0.2:
        return GOOD
    else:
        return ABSTAIN

LABELING_FUNCTIONS = [
    too_short, 
    too_long, 
    TG3_bath_no_mention, 
    TG3_kitchen_no_mention, 
    TG3_roof_no_mention, 
    TG2or3_bath_mention, 
    TG2or3_kitchen_mention, 
    TG2or3_roof_mention, 
    TG_adjusted_bath_no_mention, 
    TG_adjusted_kitchen_no_mention, 
    TG_adjusted_roof_no_mention, 
    high_liks, 
    high_ovr, 
    has_claim, 
    author_many_claims, 
    author_many_high_liks, 
    author_many_high_ovr, 
    author_few_reports, 
    common_words_by_summary_low, 
    common_words_by_summary_high, 
    common_words_by_report_low, 
    common_words_by_report_high
    ]