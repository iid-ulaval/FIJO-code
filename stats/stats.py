import json
import pandas as pd
import spacy
from spacy.attrs import ORTH
import statistics

stats = {}
offers = pd.read_csv('../data/offers.tsv', sep='\t')

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

offers['tokens'] = offers['Description Interne'].apply(lambda x: [token.text for token in nlp.tokenizer(x)])

offers['nb tokens'] = offers['tokens'].apply(lambda x: len(x))

stats['nb words'] = int(offers['nb tokens'].sum())
stats['avg nb words'] = float(offers['nb tokens'].mean())
stats['stddev nb words'] = float(offers['nb tokens'].std(ddof=0))
stats['nb offers len outlier'] = len(offers[offers['nb tokens'] >= 600])

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

with open('stats.json', 'w', encoding='utf8') as f:
    json.dump(stats, f, ensure_ascii=False)
