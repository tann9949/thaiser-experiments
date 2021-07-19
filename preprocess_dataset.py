from argparse import ArgumentParser, Namespace


def run_parser() -> Namespace:
    parser: ArgumentParser = ArgumentParser("Preprocessing dataset and format them to 16k sampling rate mono channel");
    parser.add_argument("--raw-path", metavar=DIR, type=str, default="dataset")