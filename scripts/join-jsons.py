from argparse import ArgumentParser
import json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', nargs='+', help='list of JSON files to concatenate')
    parser.add_argument('output', help='path to JSON file for output')
    args = parser.parse_args()

    concatenated = []
    for filepath in args.source:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            if isinstance(data, list):
                concatenated.extend(data)
            else:
                concatenated.append(data)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(concatenated, f, indent=2, ensure_ascii=False)
