import argparse
from numpy import array
from helpers import accuracy_score, confusion_matrix, cross_validation_split, load_dataset
from naive_bayes import NaiveBayes

parser = argparse.ArgumentParser(description='Naïve Bayes Classifier. Select a dataset by typing its filename.')

parser.add_argument('--dataset', type=str, help='the dataset filename', default='iris', choices=['iris', 'banknote_authentication'])
parser.add_argument('--crossval', type=int, help='the number of folds for cross validation prediction', choices=[3, 5, 10])

args = parser.parse_args()

dataset = load_dataset(args.dataset)
naive = NaiveBayes()
naive.fit(dataset)

if args.crossval:
    folds = cross_validation_split(dataset, args.crossval)
    predictions = naive.crossval_predict(folds)
    actual = [row[-1] for fold in folds for row in fold]
    print('Accuracies by fold:')
    accuracies = [str(round(a, 2)) + '%' for a in naive.accuracy_by_fold()]
    accuracies_str = ', '.join(accuracies)
    print(accuracies_str)
else:
    predictions = naive.predict(dataset)
    actual = [c[-1] for c in dataset]

print(' ')

# generate confusion matrix
print('Confusion matrix:')
matrix = confusion_matrix(predictions, actual, naive.get_no_of_class_values())

print(' ')

# print matrix with numpy.array()
print(array(matrix))

print(' ')

# print accuracy
print('Accuracy:')
print(str(round(accuracy_score(predictions, actual), 2)) + '%')