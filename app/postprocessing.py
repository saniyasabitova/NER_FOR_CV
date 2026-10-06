
import torch

from app.inference import tokenizer


id2tag = {
    0: "B-COLLEGE NAME",
    1: "B-COMPANIES WORKED AT",
    2: "B-DEGREE",
    3: "B-DESIGNATION",
    4: "B-EMAIL ADDRESS",
    5: "B-GRADUATION YEAR",
    6: "B-LOCATION",
    7: "B-NAME",
    8: "B-SKILLS",
    9: "B-UNKNOWN",
    10: "B-YEARS OF EXPERIENCE",
    11: "I-COLLEGE NAME",
    12: "I-COMPANIES WORKED AT",
    13: "I-DEGREE",
    14: "I-DESIGNATION",
    15: "I-EMAIL ADDRESS",
    16: "I-GRADUATION YEAR",
    17: "I-LOCATION",
    18: "I-NAME",
    19: "I-SKILLS",
    20: "I-UNKNOWN",
    21: "I-YEARS OF EXPERIENCE",
    22: "O",
}


def build_response(text, probabilities):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
    )

    offsets = encoding["offset_mapping"]
    word_ids = encoding.word_ids()

    probabilities = probabilities.squeeze(0)

    words = {}

    word_to_tokens = {}

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        if word_id not in word_to_tokens:
            word_to_tokens[word_id] = []

        word_to_tokens[word_id].append(token_idx)

    for word_id, token_indices in word_to_tokens.items():

        best_probability = -1
        best_class_id = None

        for token_idx in token_indices:
            token_probs = probabilities[token_idx]

            class_id = torch.argmax(token_probs).item()
            probability = token_probs[class_id].item()

            if probability > best_probability:
                best_probability = probability
                best_class_id = class_id

        start = offsets[token_indices[0]][0]
        end = offsets[token_indices[-1]][1]

        word_text = text[start:end]

        tag = id2tag[best_class_id]

        print(
            repr(word_text),
            "->",
            tag,
            round(best_probability, 3)
        )

        words[word_text] = tag

    return words


def merge_entities(words):
    entities = {}

    current_words = []
    current_tag = None

    for word, tag in words.items():

        if tag == "O":
            continue

        if tag.startswith("B-") or tag.startswith("I-"):
            entity_tag = tag[2:]
        else:
            entity_tag = tag

        if current_tag is None:
            current_tag = entity_tag
            current_words = [word]

        elif entity_tag == current_tag:
            current_words.append(word)

        else:
            entity_text = " ".join(current_words)
            entities[entity_text] = current_tag

            current_tag = entity_tag
            current_words = [word]

    if current_words:
        entity_text = " ".join(current_words)
        entities[entity_text] = current_tag

    return entities

