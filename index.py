import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from logic import Logic

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Indexing a genome for targets")
    parser.add_argument(
        "genome", 
        metavar="Genome",
        help="A genome of interest in fasta",
        action="store"
    )
    parser.add_argument(
        "regex", 
        metavar="Regex",
        help="A regex for PAM",
        action="store"
    )
    parser.add_argument(
        "index", 
        metavar="Index",
        help="An output index file",
        action="store"
    )
    parser.add_argument(
        "-b", "--before",
        help="Does PAM come closer to 5' end than the guide?",
        action="store_true", default=False
    )
    parser.add_argument(
        "-l", "--length",
        help="Length of a guide",
        action="store", default="20"
    )
    args = parser.parse_args()
    sequence = str([a for a in SeqIO.parse(args.genome, "fasta")][0].seq)
    guides = list(
        zip(
            *[list(a) for a in Logic.find_guides(sequence, args.regex, int(args.length), args.before)]
        )
    )
    result = pd.DataFrame(
        {
            "guide": guides[0], "start": guides[1], 
            "end": guides[2], "PAM": guides[3]
        },
        columns=["guide", "start", "end", "PAM"]
    )
    result.to_pickle(args.index)