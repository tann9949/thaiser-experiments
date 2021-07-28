from argparse import ArgumentParser, Namespace

from thaiser import preprocess_THAISER
from iemocap import preprocess_IEMOCAP
from emodb import preprocess_EmoDB
from emovo import preprocess_EMOVO


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
    print("-"*20);
    print("Processing THAI SER...");
    print("-"*20);
    preprocess_THAISER(raw_path=raw_path, n_workers=n_workers);  # preprocess THAI SER
    print("-"*20);
    print("Processing IEMOCAP...");
    print("-"*20);
    preprocess_IEMOCAP(raw_path=raw_path, n_workers=n_workers);
    print("-"*20);
    print("Processing EmoDB...");
    print("-"*20);
    preprocess_EmoDB(raw_path=raw_path, n_workers=n_workers);
    print("-"*20);
    print("Processing EMOVO...");
    print("-"*20);
    preprocess_EMOVO(raw_path=raw_path, n_workers=n_workers);


if __name__ == "__main__":
    args: Namespace = run_parser();
    main(args);
