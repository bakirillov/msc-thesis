import os
import numpy as np
import os.path as op
from weblogolib import *


class LogoWorker():
    
    def __init__(self, tempdir):
        self.tempdir = tempdir
        
    def convert(self, sequences):
        fname = op.join(
            self.tempdir,
            "".join([str(a) for a in np.random.choice(np.arange(10), 10)])+".fasta"
        )
        with open(fname, "w") as oh:
            for a in sequences:
                oh.write(">\n")
                oh.write(a+"\n")
        with open(fname, "r") as ih:
            seqs = read_seq_data(ih)
        os.remove(fname)
        return(seqs)
    
    def make_logo(self, seqs, outfile):
        data = LogoData.from_seqs(seqs)
        options = LogoOptions()
        options.title = "A Logo Title"
        fmt = LogoFormat(data, options)
        fout = open(op.join(self.tempdir, outfile), 'wb') 
        fout.write(png_formatter(data, fmt))
        
    def logo_of_list(self, sequences, outfile="current.png"):
        seqs = self.convert(sequences)
        self.make_logo(seqs, outfile)