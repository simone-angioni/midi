import itertools
import json
import sys
import warnings

from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

sys.path.append(
    r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\libraries\tegridy-tools\tegridy-tools')

import TMIDIX

from GPT2RGAX import *

import numpy as np

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, GridSearchCV

from Midi_Processing.labels_manager import count_labels
from transformer_classifier import TransformerClassifier

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, accuracy_score, make_scorer, confusion_matrix, multilabel_confusion_matrix

from visualize_metrics import display_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import tensorflow as tf


def dict_mean(reports, target):

    k = ['accuracy', 'macro avg', 'weighted avg']
    keys1 = target + k
    keys2 = ['precision', 'recall', 'f1-score', 'support']
    mean_dict = dict.fromkeys(keys1, {})

    for g in keys1:
        if isinstance(reports[0][g], dict):
            mean_dict[g] = dict.fromkeys(keys2, float)
            for key in keys2:
                mean_dict[g][key] = sum(d[g][key] for d in reports) / len(reports)
        else:
            mean_dict[g] = sum(d[g] for d in reports) / len(reports)

    return mean_dict


def plot_confusion_matrix(cm, classes, classifier_name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    #plt.show()
    plt.savefig(classifier_name + '_confusion_matrix.png')
    plt.clf()
    plt.cla()
    plt.close()


def metrics_calculation(predictedclass, originalclass, final_report, classifier_name, num_classes):

    # Code block to define and calculate some metrics
    predictedclass = np.array(predictedclass)
    originalclass = np.array(originalclass)

    if predictedclass.ndim > 1:
        predicted_y = np.argmax(predictedclass, axis=1)
        original_y = np.argmax(originalclass, axis=1)
    # This case is necessary for the svm output that is already in argmax form
    else:
        predicted_y = predictedclass
        original_y = originalclass

        predictedclass = tf.keras.utils.to_categorical(predictedclass, dtype=int, num_classes=num_classes)
        originalclass = tf.keras.utils.to_categorical(originalclass, dtype=int, num_classes=num_classes)

    conf_matr = confusion_matrix(original_y, predicted_y)

    plot_confusion_matrix(conf_matr, target, classifier_name)

    acc_score = accuracy_score(original_y, predicted_y)
    auc_score = roc_auc_score(originalclass, predictedclass, multi_class='ovr')

    # In confusion matrix i-th row and j-th column entry indicates the number of samples with true label being
    # i-th class and predicted label being j-th class
    print(conf_matr)
    print(acc_score)
    print(auc_score)

    d = {
        "Final Report calculate on tests sets": final_report,
        "Confusion matrix": conf_matr.tolist(),
        "Accuracy Score": acc_score,
        "ROC AUC Score (One vs Rest)": auc_score
    }

    # Write dictionary to text file
    with open(classifier_name + '_metrics_values_across_folds.txt', 'w') as convert_file:
        convert_file.write(json.dumps(d))

    return


def train_k_neighbour(train_data_x, train_data_y, target):
    # Variables for average classification report
    originalclass = []
    predictedclass = []

    # Make our customer score
    def classification_report_with_accuracy_score(y_true, y_pred):
        originalclass.extend(y_true)
        predictedclass.extend(y_pred)
        return accuracy_score(y_true, y_pred)  # return accuracy score

    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        knn_cv = KNeighborsClassifier(n_neighbors=3)
        # Nested CV with parameter optimization
        nested_score = cross_val_score(knn_cv, X=train_data_x, y=train_data_y, cv=10,
                                       scoring=make_scorer(classification_report_with_accuracy_score))

        # Average values in classification report for all folds in a K-fold Cross-validation
        final_report = classification_report(originalclass, predictedclass, target_names=target, output_dict=True)
        print(final_report)

        display_report(final_report, 'kNeigbour_folds_average', len(target))

        num_classes = len(target)
        metrics_calculation(predictedclass, originalclass, final_report, 'k_neighbour', num_classes)

    return


def train_random_forest(train_data_x, train_data_y, target):
    # Variables for average classification report
    originalclass = []
    predictedclass = []

    # Make our customer score
    def classification_report_with_accuracy_score(y_true, y_pred):
        originalclass.extend(y_true)
        predictedclass.extend(y_pred)
        return accuracy_score(y_true, y_pred)  # return accuracy score

    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        RFC = RandomForestClassifier(n_estimators=100)
        # Nested CV with parameter optimization
        nested_score = cross_val_score(RFC, X=train_data_x, y=train_data_y, cv=10,
                                       scoring=make_scorer(classification_report_with_accuracy_score))

        # Average values in classification report for all folds in a K-fold Cross-validation
        final_report = classification_report(originalclass, predictedclass, target_names=target, output_dict=True)
        print(final_report)

        display_report(final_report, 'randomForest_folds_average', len(target))

        num_classes = len(target)
        metrics_calculation(predictedclass, originalclass, final_report, 'random_forest', num_classes)

    return


def train_svm(train_data_x, train_data_y, target):
    # Variables for average classification report
    originalclass = []
    predictedclass = []

    # Make our customer score
    def classification_report_with_accuracy_score(y_true, y_pred):
        originalclass.extend(y_true)
        predictedclass.extend(y_pred)
        return accuracy_score(y_true, y_pred)  # return accuracy score

    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        lin_clf = svm.LinearSVC()
        # Nested CV with parameter optimization
        nested_score = cross_val_score(lin_clf, X=train_data_x, y=train_data_y.argmax(1), cv=10,
                                       scoring=make_scorer(classification_report_with_accuracy_score))

        # Average values in classification report for all folds in a K-fold Cross-validation
        final_report = classification_report(originalclass, predictedclass, target_names=target, output_dict=True)
        print(final_report)

        display_report(final_report, 'svm_folds_average', len(target))

        num_classes = len(target)
        metrics_calculation(predictedclass, originalclass, final_report, 'svm', num_classes)

    return


# This version divide the dataset into three non overlapping training, validation and test set
def train_transformer_cross_validation2(train_data_x, train_data_y, target):
    num_folds = 5

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    conf_matr_total = []
    acc_score_total = []
    auc_score_total = []

    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=32)

    reports = list()

    # count_labels(y_test.argmax(1))

    for fold_no, (train, test) in enumerate(kfold.split(train_data_x, train_data_y.argmax(1))):
        try:
            X_train, X_val, y_train, y_val = train_test_split(train_data_x[train], train_data_y[train],
                                                              test_size=0.1,
                                                              stratify=train_data_y[train])
            X_test = train_data_x[test]
            y_test = train_data_y[test]

            # Define the model

            # model = LSTM(input_shape=feed_shape, vocabulary_size=vocab_size)

            config = {'epochs': 20, 'batch_size': 32, 'train_size': len(X_train), 'num_classes': len(target)}
            model = TransformerClassifier(feed_shape, vocabulary_size=vocab_size, config=config)

            # plot_model(model.model,
            #            to_file=r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\Classification\model_plot'
            #                    r'.png',
            #            show_shapes=True, show_layer_names=True)
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            print("Number of train sample  = ", str(len(X_train)))
            print('Number of validation sample: x = ' + str(len(X_val)))
            print('Number of test sample: x = ' + str(len(X_test)))

            # LSTM train
            # history = model.fit(X_train, X_train[val], y_train, y_train[val])

            # Transformer attention train
            history = model.fit(X_train, X_val, y_train, y_val, fold_no)

            # Generate generalization metrics
            scores = model.model.evaluate(X_test, y_test, verbose=1, batch_size=2)

            print(
                f'Score for fold {fold_no}: {model.model.metrics_names[0]} of {scores[0]}; '
                f'{model.model.metrics_names[1]} of {scores[1] * 100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            y_pred = model.model.predict(X_test, batch_size=1, verbose=1)
            y_pred_bool = np.argmax(y_pred, axis=1)
            y_test1 = np.argmax(y_test, axis=1)

            report = classification_report(y_test1, y_pred_bool,
                                           target_names=target,
                                           output_dict=True)

            # print(report)

            string = "fold_n_" + str(fold_no)
            display_report(report, string, len(target))

            reports.append(report)

            conf_matr = confusion_matrix(y_test1, y_pred_bool)
            acc_score = accuracy_score(y_test1, y_pred_bool)

            y_test2 = tf.keras.utils.to_categorical(y_test1, dtype=int, num_classes=len(target))
            y_pred_bool2 = tf.keras.utils.to_categorical(y_pred_bool, dtype=int, num_classes=len(target))
            auc_score = roc_auc_score(y_test2, y_pred_bool2, multi_class='ovr')

            conf_matr_total.append(conf_matr)
            acc_score_total.append(acc_score)
            auc_score_total.append(auc_score)

            print(conf_matr)
            print(acc_score)
            print(auc_score)

            # Increase fold number
            fold_no = fold_no + 1

        except KeyboardInterrupt:
            print('Saving current progress and quitting...')
            break

    print("Accuracy per fold: ")
    print(acc_per_fold)

    avg_accuracy = sum(acc_score_total) / len(acc_score_total)
    print("Average accuracy across folds: ", avg_accuracy)

    print("Loss per fold: ")
    print(loss_per_fold)

    avg_loss = sum(loss_per_fold) / len(loss_per_fold)
    print("Average loss across folds: ", avg_loss)

    auc_score_avg = sum(auc_score_total) / len(auc_score_total)
    print("Average AUC score across folds: ", auc_score_avg)

    confusion_matrix_final = sum(conf_matr_total)
    print("Final confusion matrix: ", confusion_matrix_final)

    plot_confusion_matrix(confusion_matrix_final, target, 'transformer')

    final_report = dict_mean(reports, target)
    print(final_report)

    display_report(final_report, 'folds_average', len(target))


    # Create a dictionary with results values
    d = dict()

    d = {
        "Average accuracy across folds calculate on test sets": avg_accuracy,
        "Average loss across folds calculate on test sets": avg_loss,
        "Final Report calculate on tests sets": final_report,
        "ROC AUC score avg across folds (One vs Rest) ": auc_score_avg
    }

    # Write dictionary to text file
    with open('transform_metrics_values_across_folds.txt', 'w') as convert_file:
        convert_file.write(json.dumps(d))


def train_full_model(train_data_x, train_data_y, target):
    config = {'epochs': 50, 'batch_size': 32, 'train_size': len(train_data_x)}

    X_train, X_test, y_train, y_test = train_test_split(train_data_x, train_data_y,
                                                        test_size=0.1,
                                                        stratify=train_data_y)

    model = TransformerClassifier(feed_shape, vocabulary_size=vocab_size, config=config)

    print('Number of train sample: x = ' + str(len(X_train))
          + ' y = ' + str(len(y_train)))

    # Transformer attention train
    history = model.fit(X_train, X_test, y_train, y_test, 50)


def associate_labels_with_chunks(train_data_x, train_data_y):
    new_data = []
    new_labels = []

    for idx, x in enumerate(train_data_x):
        for c in x:
            new_data.append(c)
            new_labels.append(train_data_y[idx])

    new_data = np.array(new_data)
    new_labels = np.array(new_labels)

    return new_data, new_labels


if __name__ == "__main__":

    # Load data already processed
    # Remove the comment from the dataset you want to train on

    # Nes ints data with y label
    pickle_path = r'../dataset/nes/pickle/nes_int_with_label2'
    target = ['rpg', 'sport', 'fighting', 'shooting', 'puzzle']

    # Rock ints data with y label
    # pickle_path = r'../dataset/rock/pickle/ints_rock_dataset_labelled'
    # target = ['Clapton', 'Queen', 'Beatles', 'Rolling Stones']

    # Classic ints data with y label
    # pickle_path = r'../dataset/classic/pickle/ints_classic_dataset_labelled'
    # target = ['Albanez', 'Beethoven', 'Mozart']

    # Intra db data with y label
    # pickle_path = r'../dataset/db_merged/ints_db_merged_labelled'
    # target = ['Nes', 'Rock', 'Classic']

    # Read the dataset
    train_data_x, train_data_y = TMIDIX.Tegridy_Any_Pickle_File_Reader(pickle_path)

    # Remove this comments if you want to train on nes dataset
    # If nes transform the y from multi label [1,0,0,0,1] to multi class so each array has only a 1
    # for arr in train_data_y:
    #     idx_non_zero = [i for i, e in enumerate(arr) if e != 0]
    #     if len(idx_non_zero) > 1:
    #         arr[idx_non_zero[0]] = 0

    # Split data into shorter list
    # train_data_x, train_data_y = reduce_data_length(train_data_x, train_data_y)

    # Associate each chunk of one song with multiple same labels
    train_data_x, train_data_y = associate_labels_with_chunks(train_data_x, train_data_y)

    # Inspect database
    count_labels(train_data_y.argmax(1))

    vocab_size = 512

    feed_shape = np.shape(train_data_x[0])[0]

    print("Values in the database are between 0 and ", int(np.amax(train_data_x)))
    print('Total lists of songs labelled:', len(train_data_x))
    print('Feed Shape:', feed_shape)
    print('Vocab size:', vocab_size)
    print('Unique INTs:', len(np.unique(train_data_x)))

    train_k_neighbour(train_data_x, train_data_y, target)
    train_random_forest(train_data_x, train_data_y, target)
    train_svm(train_data_x, train_data_y, target)

    train_transformer_cross_validation2(train_data_x, train_data_y, target)

    train_full_model(train_data_x, train_data_y, target)
