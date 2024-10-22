__author__ = 'Shayan Fazeli'
__email__ = 'shayan@cs.ucla.edu'
__credit__ = 'erLab - University of California, Los Angeles'

import argparse
import os
from word_embedding_warehouse.library.embedding_builder import WordEmbeddingWarehouseBuilder


def main(args: argparse.Namespace) -> None:
    """
    The main method of this app

    Parameters
    ----------
    args: `argparse.Namespace`, required
        The arguments
    """
    builder = WordEmbeddingWarehouseBuilder(
        output_path=os.path.abspath(args.warehouse_folder),
        download_links_file=os.path.abspath(os.path.join(os.path.dirname(__file__), '../configuration.json')),
        clear_cache=(args.clear_cache == 1)
    )
    builder.build_warehouse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Building word embedding warehouse")
    parser.add_argument("--warehouse_folder", required=True, type=str,
                        help="The path to the output folder that the user designated to be the warehouse.")
    parser.add_argument("--clear_cache", default=0, type=int, help="If set to 1, this will clear the warehouse if it already exists.")

    args = parser.parse_args()
    main(args)
