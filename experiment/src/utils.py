import json
import string
from typing import List, Tuple

from spacy.training import offsets_to_biluo_tags
import spacy
from spacy.attrs import ORTH

import warnings


def prepare_raw_data(data_path: str) -> Tuple[List, dict]:
    warnings.filterwarnings("error", category=UserWarning)
    nlp = spacy.load('fr_core_news_lg')

    tokenizer = nlp.tokenizer

    tokenizer.add_special_case("<anon_company>", [{ORTH: "<anon_company>"}])
    tokenizer.add_special_case("<anon_company>.", [{ORTH: "<anon_company>."}])
    tokenizer.add_special_case("<anon_company>,", [{ORTH: "<anon_company>,"}])
    tokenizer.add_special_case("chez<anon_company>", [{
        ORTH: "chez"
    }, {
        ORTH: "<anon_company>"
    }])
    tokenizer.add_special_case("l'<anon_company>,", [{
        ORTH: "l'<anon_company>,"
    }])
    tokenizer.add_special_case("<anon_company>'s", [{
        ORTH: "<anon_company>'s"
    }])

    tokenizer.add_special_case("<anon_misc>", [{ORTH: "<anon_misc>"}])
    tokenizer.add_special_case("<anon_misc>.", [{ORTH: "<anon_misc>."}])
    tokenizer.add_special_case("<anon_misc>,", [{ORTH: "<anon_misc>,"}])

    tokenizer.add_special_case("<anon_location>", [{ORTH: "<anon_location>"}])
    tokenizer.add_special_case("<anon_location>.", [{
        ORTH: "<anon_location>."
    }])
    tokenizer.add_special_case("<anon_location>,", [{
        ORTH: "<anon_location>,"
    }])
    tokenizer.add_special_case("<anon_location>/<anon_location>",
                               [{
                                   ORTH: "<anon_location>/<anon_location>"
                               }])
    tokenizer.add_special_case("<anon_location>.]", [{
        ORTH: "<anon_location>.]"
    }])

    data = json.load(open(data_path, "r"))

    samples = data["samples"]

    labels = [label["id"] for label in data["interface"]["labels"]]

    tags_to_idx = {"O": 0}

    idx = 1
    for entity in labels:
        tags_to_idx["I-" + entity] = idx

        idx += 1

    training_data = []

    for sample in samples:
        doc = nlp(sample["document"])

        entities = []

        for entity in sample["annotation"]["entities"]:
            entities.append((entity["start"], entity["end"], entity["label"]))

        try:
            tags = offsets_to_biluo_tags(doc, entities)

        except UserWarning:
            continue

        for i in range(len(tags)):
            tags[i] = tags[i].replace("L-",
                                      "I-").replace("U-",
                                                    "I-").replace("B-", "I-")

        sentence = [str(token).lower().strip() for token in doc]
        dels = []

        for i in range(len(sentence)):
            token = sentence[i]

            if token in string.punctuation:
                dels.append(i)

        offset = 0
        for i in dels:
            del sentence[i - offset]
            del tags[i - offset]

            offset += 1

        training_data.append((sentence, tags))

    return training_data, tags_to_idx
