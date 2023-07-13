from csv import reader
from math import log, sqrt
from math import pi
from math import exp
from random import randrange
import random

def calculate_std_for_attribute(attr, dataset, mean):
    std = 0
    no_examples = len(dataset)
    for i in range(no_examples):
        std += pow(dataset[i][attr] - mean, 2)
    std = sqrt(std / (no_examples - 1))
    # very small diff in decimal points
    return std

def calculate_mean_for_attribute(attr, dataset):
    mean = 0
    no_examples = len(dataset)
    for i in range(no_examples):
        mean += dataset[i][attr]
    mean /= no_examples
    return mean

def normalize(dataset):
    no_attrs = len(dataset[0]) - 1
    for attr in range(no_attrs):
        # very small diff in decimal points
        mean = calculate_mean_for_attribute(attr, dataset)
        std = calculate_std_for_attribute(attr, dataset, mean)
        no_examples = len(dataset)
        for i in range(no_examples):
            dataset[i][attr] = (dataset[i][attr] - mean) / std

# Load a CSV file
def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

def load_dataset(dataset_filename):
    filename = f'{dataset_filename}.csv'
    dataset = load_csv(filename)
    # strip out first row
    dataset = dataset[1:]
    print(f'Loaded data file {filename} with {len(dataset)} rows and {len(dataset[0])} columns')

    for i in range(len(dataset[0])-1):
        # convert numeric columns to floats
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)

    # shuffle data
    random.shuffle(dataset)

    # normalize data
    normalize(dataset)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def get_uniques(values):
    '''
    Returns a list of ordered unique values. 
    If the list of unique values is not ordered, the confusion matrix won't work.
    The function was taken from this SO thread:
    https://stackoverflow.com/questions/44628186/convert-python-list-to-ordered-unique-values
    '''
    lookup = set()  # a temporary lookup set
    return [x for x in values if x not in lookup and lookup.add(x) is None]

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = get_uniques(class_values)
    lookup = dict()
    for i in range(len(unique)):
        value = unique[i]
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


def log_probability(x, y):
    x = log(x)
    y = log(y)
    product = exp(x + y)
    return product


def accuracy_score(predictions, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predictions[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def generate_empty_matrix(dim):
    return [ [0]*dim for i in range(dim)]

def confusion_matrix(predictions, actual, dim=3):
    # generate empty matrix
    matrix = generate_empty_matrix(dim)

    # iterate and fill matrix
    for i in range(len(predictions)):
        p = predictions[i]
        a = actual[i]
        matrix[p][a] += 1

    # return output
    return matrix

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    # generate folds of instances in random order
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            # select a random index from copied dataset
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def calculate_mean(rows):
    no_attributes = len(rows[0])
    mean_values = [0] * no_attributes
    for row in rows:
        for attr in range(no_attributes):
            mean_values[attr] += row[attr]

    for attr in range(no_attributes):
        mean_values[attr] /= len(rows)

    return mean_values

def calculate_std(rows, mean_values):
    no_attributes = len(rows[0])
    std_values = [0] * no_attributes
    for row in rows:
        for attr in range(no_attributes):
            std_values[attr] += pow(row[attr] - mean_values[attr], 2)

    for attr in range(no_attributes):
        std_values[attr] /= len(rows) - 1
        std_values[attr] = sqrt(std_values[attr])

    return std_values