from argparse import ArgumentParser
import json


def statistics(filepath: str) -> None:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(filepath)
    # TODO tokens count, punctuation counts (percentage), so on..


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filepaths', nargs='+', help='filepath to JSON file/files containing data')
    args = parser.parse_args()

    for filepath in args.filepaths:
        statistics(filepath)
