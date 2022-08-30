from argparse import ArgumentParser
import json

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('source', help='path to JSON file')
    parser.add_argument('output', help='path for output with <> to be replaced with train, val, or test')
    parser.add_argument('--val', type=float, default=0.1, help='val part')
    parser.add_argument('--test', type=float, default=0.1, help='test part')
    args = parser.parse_args()

    with open(args.source, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_val, test = train_test_split(data, test_size=args.test)
    train, val = train_test_split(train_val, test_size=(args.val / (1 - args.test)))

    with open(args.output.replace('<>', 'train'), 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    with open(args.output.replace('<>', 'val'), 'w', encoding='utf-8') as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    with open(args.output.replace('<>', 'test'), 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
