from argparse import ArgumentParser, Namespace

from thaiser import preprocess_THAISER
from iemocap import preprocess_IEMOCAP


def run_parser() -> Namespace:
    """
    Run argument parser for the program

    Return
    ------
    args: Namespace
        A namespace object contains program argument
    """
    parser: ArgumentParser = ArgumentParser("Preprocessing dataset and format them to 16k sampling rate mono channel");
    parser.add_argument("--raw-path", metavar="DIR", type=str, default="dataset");
    parser.add_argument("--n-workers", type=int, default=None, help="Number of workers needed. Default as number of CPU")
    return parser.parse_args();


def main(args: Namespace) -> None:
    # argument parser options
    raw_path: str = args.raw_path;
    n_workers: int = args.n_workers

    # process listed dataset
    preprocess_THAISER(raw_path=raw_path, n_workers=n_workers);  # preprocess THAI SER
    preprocess_IEMOCAP(raw_path);
    # TODO:
    # preprocess_EmoDB(raw_path);
    # preprocess_EMOVO(raw_path);


if __name__ == "__main__":
    args: Namespace = run_parser();
    main(args);
