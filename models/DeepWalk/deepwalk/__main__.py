#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

import numpy as np

from . import graph
from . import walks as serialized_walks
from gensim.models import Word2Vec
from .skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()


def process(args):
    # directly pass the edgelist
    G = graph.trans_edgelist(args['edge'], undirected=args['undirected'])

    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args['number_walks']

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args['walk_length']

    print("Data size (walks*length): {}".format(data_size))

    if data_size < args['max_memory_data_size']:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=args['number_walks'],
                                            path_length=args['walk_length'], alpha=0, rand=random.Random(args['seed']))
        print("Training...")
        model = Word2Vec(walks, size=args['representation_size'], window=args['window_size'], min_count=0, sg=1, hs=1,
                         workers=args['workers'])
    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size,
                                                                                                             args[
                                                                                                                 'max_memory_data_size']))
        print("Walking...")

        walks_filebase = args['output'] + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args['number_walks'],
                                                          path_length=args['walk_length'], alpha=0,
                                                          rand=random.Random(args['seed']),
                                                          num_workers=args['workers'])

        print("Counting vertex frequency...")
        if not args['vertex_freq_degree']:
            vertex_counts = serialized_walks.count_textfiles(walk_files, args['workers'])
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                         vector_size=args['representation_size'],
                         window=args['window_size'], min_count=0, trim_rule=None, workers=args['workers'])

    # output to file
    model.wv.save_word2vec_format(args['output'])

    # fuse as numpy array
    feature = model.wv.vectors
    index_to_key = np.asarray(model.wv.index_to_key).astype(int)
    feature = np.hstack([feature[np.argsort(index_to_key), :]])
    index_to_key = np.sort(index_to_key)

    return index_to_key, feature

def embedding_process(debug=False, edge=None, log='INFO', matfile_variable_name='network',
                      max_memory_data_size=1000000000, number_walks=10, output='out', representation_size=64,
                      seed=0, undirected=True, vertex_freq_degree=False, walk_length=40, window_size=5, workers=1):
    numeric_level = getattr(logging, log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    if debug:
        sys.excepthook = debug

    return process(
        {'debug': debug,
         'edge': edge,
         'log': log,
         'matfile_variable_name': matfile_variable_name,
         'max_memory_data_size': max_memory_data_size,
         'number_walks': number_walks,
         'output': output,
         'representation_size': representation_size,
         'seed': seed,
         'undirected': undirected,
         'vertex_freq_degree': vertex_freq_degree,
         'walk_length': walk_length,
         'window_size': window_size,
         'workers': workers})


class DeepWalk(object):
    def __init__(self):
        self.debug = False
        self.edge = None
        self.log = 'INFO'
        self.matfile_variable_name = 'network'
        self.max_memory_data_size = 1000000000
        self.number_walks = 10
        self.output = 'out'
        self.representation_size = 64,
        self.seed = 0
        self.undirected = True
        self.vertex_freq_degree = False
        self.walk_length = 40
        self.window_size = 5
        self.workers = 1

    def process(self):
        return embedding_process(self.debug,
                                 self.edge,
                                 self.log,
                                 self.matfile_variable_name,
                                 self.max_memory_data_size,
                                 self.number_walks,
                                 self.output,
                                 self.representation_size,
                                 self.seed,
                                 self.undirected,
                                 self.vertex_freq_degree,
                                 self.walk_length,
                                 self.window_size,
                                 self.workers)


if __name__ == "__main__":
    embedding_process(format='edgelist',
                      input='../example_graphs/p2p-Gnutella08.edgelist',
                      max_memory_data_size=0,
                      number_walks=80,
                      representation_size=128,
                      walk_length=40,
                      window_size=10,
                      workers=8,
                      output='../example_graphs/blogcatalog.embeddings')
