"""Parser for arguments

Put all arguments in one file and group similar arguments
"""
import argparse


class Parser():

    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        self.parser.add_argument(
            '--batch_size', type=int, default=16,
            help='batch size for training and validation (default: 16)')
        self.parser.add_argument(
            '--device', type=str, default=5,
            help='which gpu device to use (default: 5)')


        # loss
        self.parser.add_argument(
            '--b', type=float, default=0,
            help='beta * loss2')
        # net
        self.parser.add_argument(
            '--hidden_dim', type=int, default=20,
            help='number of hidden units (default: 20)')

        # graph
        self.parser.add_argument(
            '--learn_eps', action="store_true",
            help='learn the epsilon weighting')

        self.parser.add_argument(
            '--lr_ratio', type=float, default=1,
            help='random lr_ratio (default: 1)')
        self.parser.add_argument(
            '--epochs', type=int, default=1000,
            help='number of epochs to train (default: 1000)')
        self.parser.add_argument(
            '--lr', type=float, default=1e-2,
            help='learning rate (default: 1e-2)')
        self.parser.add_argument(
            '--input_dropout', type=float, default=0,
            help='input layer dropout (default: 0)')
        self.parser.add_argument(
            '--final_dropout', type=float, default=0,
            help='final layer dropout (default: 0)')
        self.parser.add_argument(
            '--weight_decay_rate', type=float, default=1e-4,
            help='weight decay rate (default: 1e-4)')
        self.parser.add_argument(
            '--feature_dim', type=int, default=1,
            help='feature dim (default: 1)')
        
        
        # data file path
        self.parser.add_argument(
            '--ppi_pth', type=str, default='../data/networks/CPDB_symbols_edgelist.tsv',
            help='path of ppi file')
        
        self.parser.add_argument(
            '--exp_pth', type=str, default='../data/expression/sample_matrix_fc_gtex.h5',
            help='path of TCGA gene exp file')
        
        self.parser.add_argument(
            '--clin_pth', type=str, default='../data/clinical/merge_biolinks_clin.csv',
            help='path of clinical file')

        # done
        self.args = self.parser.parse_args()