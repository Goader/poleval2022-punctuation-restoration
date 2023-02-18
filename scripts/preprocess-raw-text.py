from argparse import ArgumentParser
from typing import Any
from pathlib import Path
import json

from spacy.lang.pl import Polish

tokenizer = Polish().tokenizer

DIALOG_PREFIXES = ['—']


def tokenize_text(text: str) -> list[dict[str, str | bool]]:
    tokens = []

    doc = tokenizer(text)
    for token in doc:
        if token.is_space:
            if tokens:
                tokens[-1]['space_after'] = True
            continue

        if token.is_punct:
            if tokens:
                tokens[-1]['punctuation'] = token.text
                tokens[-1]['space_after'] = tokens[-1]['space_after'] or token.whitespace_ != ''
            continue

        tokens.append({
            'word': token.text.lower(),
            'punctuation': '',
            'space_after': token.whitespace_ != ''
        })

    return tokens


def preprocess_text(title: str, text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.strip('\n').split('\n') if line]
    # TODO preprocess dialogs?

    documents = [
        {
            'title': title + f'-line{i}',
            'words': tokenize_text(line)
        }
        for i, line in enumerate(lines)
    ]

    return documents


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('text_files', nargs='+', type=str, help='txt files to preprocess')
    parser.add_argument('output', type=str, help='path for the output JSON file')
    args = parser.parse_args()

    documents = []
    for filepath in args.text_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        documents.extend(preprocess_text(Path(filepath).name.removesuffix('.txt'), text))

    print(xx)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
