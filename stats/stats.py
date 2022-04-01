import json
from collections import Counter

import pandas as pd
import spacy
from spacy.attrs import ORTH
import statistics
import plotly.express as px
import plotly.io as pio


def tokenize(model, text):
    return [token for token in model.tokenizer(text)]


pio.kaleido.scope.mathjax = None

stats = {}
offers = pd.read_csv('../data/offers.tsv', sep='\t')
annotated = json.load(open('../data/fijo.json', "r"))

stats['Nb offers'] = len(offers)

# small cleaning.
# Drop all empty offers
offers = offers.dropna()
stats['Nb non empty offers'] = len(offers)

# Remove N/D offers
offers = offers[offers['Description Interne'] != 'N/D']
stats['Nb french offers'] = len(offers)

# remove crlf
offers = offers.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=[" ", " "], regex=True)

nlp = spacy.load('fr_core_news_lg')

special_cases = [('-Sens', [{ORTH: "-"}, {ORTH: "Sens"}]),
                 ('-Recueillement', [{ORTH: "-"}, {ORTH: "Recueillement"}]),
                 ('-Autonomie', [{ORTH: "-"}, {ORTH: "Autonomie"}]),
                 ('organisation-', [{ORTH: "organisation"}, {ORTH: "-"}]),
                 ('équipe-', [{ORTH: "équipe"}, {ORTH: "-"}]),
                 ('Responsable-', [{ORTH: "Responsable"}, {ORTH: "-"}]),
                 ('.Commande', [{ORTH: "."}, {ORTH: "Commande"}]),
                 ('.Reçoit', [{ORTH: "."}, {ORTH: "Reçoit"}]),
                 ('.Répondre', [{ORTH: "."}, {ORTH: "Répondre"}]),
                 ('.Envoi', [{ORTH: "."}, {ORTH: "Envoi"}]),
                 ('-Professionnalisme', [{ORTH: "-"}, {ORTH: "Professionnalisme"}]),
                 ("<anon_company>", [{ORTH: "<anon_company>"}]),
                 ("<anon_company>.", [{ORTH: "<anon_company>."}]),
                 ("<anon_company>,", [{ORTH: "<anon_company>,"}]),
                 ("chez<anon_company>", [{ORTH: "chez"}, {ORTH: "<anon_company>"}]),
                 ("l'<anon_company>,", [{ORTH: "l'<anon_company>,"}]),
                 ("<anon_company>'s", [{ORTH: "<anon_company>'s"}]),
                 ("<anon_misc>", [{ORTH: "<anon_misc>"}]),
                 ("<anon_misc>.", [{ORTH: "<anon_misc>."}]),
                 ("<anon_misc>,", [{ORTH: "<anon_misc>,"}]),
                 ("<anon_location>", [{ORTH: "<anon_location>"}]),
                 ("<anon_location>.", [{ORTH: "<anon_location>."}]),
                 ("<anon_location>,", [{ORTH: "<anon_location>,"}]),
                 ("<anon_location>/<anon_location>", [{ORTH: "<anon_location>/<anon_location>"}]),
                 ("<anon_location>.]", [{ORTH: "<anon_location>.]"}])
                 ]

for case in special_cases:
    nlp.tokenizer.add_special_case(case[0], case[1])

# ----
# Non Annotated stats
# ----

offers['tokens'] = offers['Description Interne'].apply(
    lambda x: [token.text for token in tokenize(nlp, x)])

offers['nb tokens'] = offers['tokens'].apply(lambda x: len(x))

stats['nb words'] = int(offers['nb tokens'].sum())
stats['avg nb words'] = float(offers['nb tokens'].mean())
stats['stddev nb words'] = float(offers['nb tokens'].std(ddof=0))
stats['nb offers len outlier'] = len(offers[offers['nb tokens'] > 572])

all_tokens = []
for list_t in offers['tokens']:
    all_tokens.extend(list_t)
uniq_tokens = set(all_tokens)
stats['nb uniq word'] = len(uniq_tokens)
stats['Lexical richness'] = stats['nb uniq word'] / stats['nb words']

nlp.add_pipe("sentencizer")
offers['sentences'] = offers['Description Interne'].apply(lambda x: list(nlp(x).sents))
offers['nb sentences'] = offers['sentences'].apply(lambda x: len(x))
offers['len sentences'] = offers['sentences'].apply(lambda x: [len(s) for s in x])

stats['avg nb sentences'] = float(offers['nb sentences'].mean())

all_sent_len = []
for list_s in offers['len sentences']:
    all_sent_len.extend(list_s)
stats['avg len sentences'] = statistics.mean(all_sent_len)

# figure 1
fig1 = px.histogram(offers, x="nb tokens", marginal="box",
                    labels={'nb tokens': 'Offers length (token)'})
fig1.update_layout(yaxis_title='Number of offers',
                   margin=dict(l=10, r=10, t=10, b=10),
                   font=dict(size=18)
                   )
fig1.write_image('sentence_len_trunc.pdf', engine="kaleido")

# ----
# Annotated stats
# ----

stats['annotated'] = {}

ids = [a['_id'] for a in annotated['samples']]

id_offers = [i.split('.')[0] for i in ids]
stats['annotated']['nb annotated offers'] = len(set(id_offers))
stats['annotated']['nb annotated sentences'] = len(id_offers)

entities = [a['annotation']['entities'] for a in annotated['samples']]
labels = [item['label'] for sublist in entities for item in sublist]
stats['annotated']['nb annotations'] = len(labels)
stats['annotated']['nb_annotations_per_label'] = {}
for label in set(labels):
    stats['annotated']['nb_annotations_per_label'][label] = len([lbl for lbl in labels if lbl == label])

documents = [a['document'] for a in annotated['samples']]
words = []
sentences = []
for d in documents:
    words.extend([token.text for token in tokenize(nlp, d)])
    sentences.extend(list(nlp(d).sents))
stats['annotated']['nb words'] = len(words)

stats['annotated']['nb unique words'] = len(set(words))

stats['annotated']['avg nb word'] = stats['annotated']['nb words'] / stats['annotated']['nb annotated offers']

stats['annotated']['nb sentences'] = len(sentences)
stats['annotated']['avg nb sentence'] = stats['annotated']['nb sentences'] / stats['annotated']['nb annotated offers']

# figure 4
trans = {'pensee': 'Thoughts',
         'personnel': 'Personal',
         'relationnel': 'Relational',
         'resultats': 'Results'}
sorted_lbl_value = list(zip(*sorted(stats['annotated']['nb_annotations_per_label'].items())))
data_fig4 = {'Groups': [trans[lbl] for lbl in list(sorted_lbl_value[0])],
             'Number of skills': [lbl for lbl in list(sorted_lbl_value[1])]}
fig4 = px.bar(data_fig4, x='Groups', y='Number of skills')
fig4.update_layout(yaxis_title='Number of skills',
                   xaxis_title='Groups',
                   margin=dict(l=10, r=10, t=10, b=10),
                   font=dict(size=18)
                   )
fig4.write_image('entities_numbers.pdf', engine="kaleido")

# figure 5
all_stopwords = nlp.Defaults.stop_words
words_per_annot = []
text_annotation = [item['text'] for sublist in entities for item in sublist]
for d in text_annotation:
    words_per_annot.append([token.text for token in tokenize(nlp, d)])
len_words_per_annot = [len(w) for w in words_per_annot]
stats['annotated']['min_length_annotated'] = min(len_words_per_annot)
stats['annotated']['max_length_annotated'] = max(len_words_per_annot)
stats['annotated']['mean_length_annotated'] = statistics.mean(len_words_per_annot)
stats['annotated']['std_length_annotated'] = statistics.stdev(len_words_per_annot)
stats['annotated']['median_length_annotated'] = statistics.median(len_words_per_annot)
df_len_words_wo_stpw_per_annot = pd.DataFrame(data=len_words_per_annot,
                                              columns=['Skills length (token)'])
fig5 = px.histogram(df_len_words_wo_stpw_per_annot, x='Skills length (token)', marginal="box", nbins=100)
fig5.update_layout(yaxis_title='Number of skills',
                   margin=dict(l=10, r=10, t=10, b=10),
                   font=dict(size=18)
                   )
fig5.write_image('entities_len.pdf', engine="kaleido")

# figure 6
words_in_annotations = [item for sublist in words_per_annot for item in sublist]
words_out_annotations = []
for sample in annotated['samples']:
    text_out_annotation = sample['document']
    for ent in sample['annotation']['entities']:
        text_out_annotation = text_out_annotation.replace(ent['text'], '')
    words_out_annotations.extend([token.text for token in tokenize(nlp, text_out_annotation)])
count_stop_words_in = Counter([w for w in words_in_annotations if w in all_stopwords])
count_stop_words_out = Counter([w for w in words_out_annotations if w in all_stopwords])
stopwords_total_df = pd.DataFrame({'In annotation': count_stop_words_in, 'Out of annotations': count_stop_words_out})
stopwords_total_df.sort_values(by=['In annotation'], inplace=True)
stopwords_total_df = stopwords_total_df.dropna()
fig6 = px.bar(stopwords_total_df[-20:],
              x=["In annotation", "Out of annotations"], barmode="overlay", opacity=1)
fig6.update_layout(xaxis_title='Number of stopwords',
                   yaxis_title="",
                   margin=dict(l=10, r=10, t=10, b=10),
                   font=dict(size=14),
                   legend=dict(yanchor="bottom",
                               y=0.01,
                               xanchor="right",
                               x=0.99,
                               title=dict(text=""))
                   )
fig6.write_image('stopwords.pdf', engine="kaleido")

with open('stats.json', 'w', encoding='utf8') as f:
    json.dump(stats, f, ensure_ascii=False)
