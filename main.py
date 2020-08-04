# coding: utf-8

import time
import csv
import operator
import sys, getopt
import argparse
from utils import scalers_list, Writer, Reader, StatsRecorder
import json
import numpy as np
import logging

log = logging.getLogger("my-logger")

class Preprocesor:

    def __init__(self, config: dict):
        self.config = config

        data_types = {'int': int,
                      'float': float,
                      'str': str}
        self.feature_data_type = data_types[config['feature_data_type']]
        self.mean = None
        self.std = None

    def train(self, train_file_path):
        reader = Reader(train_file_path, self.feature_data_type)
        stats = StatsRecorder()

        for line in reader:
            stats.update(np.array(line[1:]))

        self.mean = stats.mean.tolist()
        self.std = stats.std.tolist()

    def preprocess(self, file_path: str, scaler=None):

        self.scaler = scaler(default_mean=self.mean, default_std=self.std)

        file_path2save = '{}_procecced.tsv'.format('.'.join(file_path.split('.')[:-1]))
        reader = Reader(file_path, self.feature_data_type)
        writer = Writer(file_path2save, self._get_writer_column_names(reader))
        start = time.time()
        n = 0
        for _id, *values in reader:

            line_processed = self._process_sample(sample_values=values)

            n += 1
            writer.writerow([_id, *line_processed])

            if n % 1000 == 0:
                print("Processed {} rows. Time spent: {}".format(n, time.time() - start))


        print("Total processed {} rows. Total time spent:{}".format(n, time.time() - start))

    def _process_sample(self, sample_values: list):

        scaled_values = self.scaler.transform_sample(sample_values)
        max_index, max_value = max(enumerate(sample_values), key=operator.itemgetter(1))
        max_feature_abs_mean_diff = abs(max_value - self.mean[max_index])

        return [*scaled_values, max_index, max_feature_abs_mean_diff]

    def _get_writer_column_names(self, reader):
        return [
            reader.column_names[0],
            *("feature_{}_stand_{}".format(reader.param_name, i) for i in range(reader.values_n)),
            "max_feature_{}_index".format(reader.param_name),
            "max_feature_{}_abs_mean_diff".format(reader.param_name)
        ]

def main(args):

    with open(args.config_path) as json_file:
        config = json.load(json_file)

    for k in config.keys():
        if args.__dict__[k]:
            config[k] = args.__dict__[k]

    config['scale_method'] = scalers_list[config['scale_method']]


    preprocesor = Preprocesor(config=config)
    preprocesor.train(config['train_path'])
    preprocesor.preprocess(config['test_path'], scaler=config['scale_method'])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This script preprocess the dataset.\n'
                                                 'Preprocessing include applying z-score normalization and adding some new features')

    parser.add_argument('--train_path', action='store', type=str, required=False)
    parser.add_argument('--test_path', action='store', type=str, required=False)
    parser.add_argument('--scale_method', action='store', type=str, required=False)
    parser.add_argument('--feature_data_type', action='store', type=str, required=False)
    parser.add_argument('--config_path', action='store', type=str, required=False,
                        default='./configs/StandardScaler_config.json')


    args = parser.parse_args()

    main(args)




