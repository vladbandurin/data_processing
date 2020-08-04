
import numpy as np
import csv


class StandardScaler():
    """Standardize features by removing the mean and scaling to unit variance
    The standard score of a sample `x` is calculated as:
        z = (x - u) / s
    where `u` is the mean of the training samples,
    and `s` is the standard deviation of the training samples. """

    def __init__(self, default_mean=None, default_std=None):
        self.mean_ = default_mean
        self.std_ = default_std

    def transform_sample(self, values: list):
        """Perform standardization by centering and scaling"""
        values = [(values[i] - self.mean_[i]) / self.std_[i] for i in range(len(values))]
        return values

class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m / (m + n) * tmp + n / (m + n) * newmean
            self.std = m / (m + n) * self.std ** 2 + n / (m + n) * newstd ** 2 + \
                       m * n / (m + n) ** 2 * (tmp - newmean) ** 2
            self.std = np.sqrt(self.std)

            self.nobservations += n

class Reader:
    def __init__(self, filename, data_type):
        self.__file = open(filename, 'rt')
        self.__reader = csv.reader(self.__file, delimiter="\t")
        self.column_names = next(self.__reader)
        self.first_row = next(self.__reader)
        self.param_name, *values = (x for x in self.first_row[1].split(','))
        self.values_n = len(values)
        self.data_type = data_type

    def __split_values(self, row):
        return (row[0], *(self.data_type(x) for x in row[1].split(',')[1:]))

    def __del__(self):
        self.__file.close()

    def __iter__(self):
        yield self.__split_values(self.first_row)
        for line in self.__reader:
            yield (line[0], *(self.data_type(x) for x in line[1].split(',')[1:]))

class Writer:
    def __init__(self, filename, column_names):
        self.file = open(filename, 'wt')
        self.writer = csv.writer(self.file, delimiter="\t")
        self.column_names = column_names
        self.column_N = len(column_names)
        self.writer.writerow(column_names)

    def __del__(self):
        self.file.close()

    def writerow(self, values):
        self.writer.writerow(values)

scalers_list = {'StandardScaler': StandardScaler}
