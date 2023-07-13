from helpers import accuracy_score, calculate_mean, calculate_probability, calculate_std, log_probability

class NaiveBayes:
    def __init__(self):
        self.summaries = dict()
        self.accuracies = list()

    @classmethod
    def fit(self, dataset):
        self.summaries = self.summarize_by_class(dataset)

    @classmethod
    def predict(self, dataset):
        predictions = list()
        for row in dataset:
            output = self.predict_row(row)
            predictions.append(output)
        return predictions

    @classmethod
    def predict_row(self, row):
        probabilities = self.calculate_class_probabilities(row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            # update the best label if better is found
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    @classmethod
    def calculate_class_probabilities(self, row):
        total_rows = sum([self.summaries[label]['total_rows'] for label in self.summaries])
        probabilities = dict()
        for class_value, class_summaries in self.summaries.items():
            probabilities[class_value] = self.summaries[class_value]['total_rows']/float(total_rows)
            for i in range(len(class_summaries['mean'])):
                mean = class_summaries['mean'][i]
                std = class_summaries['std'][i]
                # use gaussian pdf to calculate probability
                probability = calculate_probability(row[i], mean, std)
                # log probability to avoid numerical underflows
                product = log_probability(probabilities[class_value], probability)
                probabilities[class_value] = product
        return probabilities

    @classmethod
    def summarize_by_class(self, dataset):
        # divide dataset into categories
        separated = self.separate_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            # calculate the mean and std for each instance
            data_rows = [r[:-1] for r in rows]
            class_summaries = dict()
            mean_values = calculate_mean(data_rows)
            std_values = calculate_std(data_rows, mean_values)
            class_summaries['mean'] = mean_values
            class_summaries['std'] = std_values
            class_summaries['total_rows'] = len(rows)
            summaries[class_value] = class_summaries
        return summaries

    @classmethod
    def separate_by_class(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    @classmethod
    def crossval_predict(self, folds):
        accuracies = list()
        predictions = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.predict(test_set)
            predictions.extend(predicted)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_score(predicted, actual)
            accuracies.append(accuracy)
        # store accuracies
        self.accuracies = accuracies
        return predictions

    @classmethod
    def accuracy_by_fold(self):
        return self.accuracies

    @classmethod
    def total_accuracy(self):
        return sum(self.accuracies) / len(self.accuracies)

    @classmethod
    def get_no_of_class_values(self):
        return len(self.summaries.keys())