import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score, roc_auc_score, accuracy_score
from sklearn import metrics
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt

vectorizer = CountVectorizer(stop_words='english')
tfidf_transformer = TfidfTransformer()
comments_df = pd.read_csv('./data/transformed-comments.csv')
selected_clf = LinearSVC()
clfs = [
    LinearSVC(),
    MultinomialNB(),
    KNeighborsClassifier(),
]


def get_tfidf_features(X):
    X_counts = vectorizer.fit_transform(X)
    return tfidf_transformer.fit_transform(X_counts)


def get_model_data(clf_instance):
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        comments_df.comment, 
        comments_df.unacceptable, 
        comments_df.index,
        random_state=0
    )
    model = clf_instance.fit(
        get_tfidf_features(X_train),
        y_train
    )
    y_pred = model.predict(
        vectorizer.transform(X_test)
    )
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "indices_train": indices_train,
        "indices_test": indices_test,
        "y_pred": y_pred,
    }


def plot_distribution():
    plt.figure(figsize=(8,6))
    comments_df.groupby('unacceptable').comment.count().plot.bar(ylim=0)
    plt.show()


def plot_cv_accuracies():
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(clfs)))
    entries = []
    for classifier in clfs:
        clf_name = classifier.__class__.__name__
        accuracies = cross_val_score(classifier, get_tfidf_features(comments_df.comment), comments_df.unacceptable, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((clf_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['clf_name', 'fold_idx', 'accuracy'])
    sns.boxplot(x='clf_name', y='accuracy', data=cv_df)
    sns.stripplot(x='clf_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    print(cv_df.groupby('clf_name').accuracy.mean())


def plot_rocs():
    for classifier in clfs:
        model_data = get_model_data(classifier)
        fpr, tpr, thresh = metrics.roc_curve(model_data['y_test'], model_data['y_pred'])
        auc = metrics.roc_auc_score(model_data['y_test'], model_data['y_pred'])
        plt.plot(fpr,tpr,label=f"{classifier.__class__.__name__} auc: {str(round(auc, 3))}")
    plt.legend(loc=0)
    plt.show()


def get_metrics():
    clfs_names = []
    percisions = []
    f1s = []
    recalls = []
    rocs = []
    fprs = []
    tprs = []
    accuracies = []
    for classifier in clfs:
        model_data = get_model_data(classifier)
        clfs_names.append(classifier.__class__.__name__)
        percisions.append(precision_score(model_data['y_test'], model_data['y_pred']))
        f1s.append(f1_score(model_data['y_test'], model_data['y_pred']))
        recalls.append(recall_score(model_data['y_test'], model_data['y_pred']))
        rocs.append(roc_auc_score(model_data['y_test'], model_data['y_pred']))
        fpr, tpr, thresh = metrics.roc_curve(model_data['y_test'], model_data['y_pred'])
        fprs.append(fpr[1])
        tprs.append(tpr[1])
        accuracies.append(accuracy_score(model_data['y_test'], model_data['y_pred']))
    metrics_df = pd.DataFrame(
        data={
            'algorithm': clfs_names,
            'percision': percisions,
            'recall': recalls,
            'f1_score': f1s,
            'roc': rocs,
            'fpr': fprs,
            'tpr': tprs,
            'accuracy': accuracies
        }
    )
    metrics_df.to_csv('./data/metrics.csv')


def get_conf_matrix_data():
    model_data = get_model_data(selected_clf)
    conf_mat = confusion_matrix(model_data["y_test"], model_data["y_pred"])
    return {
        "matrix": conf_mat,
        **model_data,
    }


def plot_confusion_matrix():
    cf_data = get_conf_matrix_data()
    plt.subplots(figsize=(10,10))
    sns.heatmap(cf_data['matrix'], annot=True, fmt='d')
    plt.title(selected_clf.__class__.__name__)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# plot_distribution()
# plot_cv_accuracies()
# plot_confusion_matrix()
# plot_rocs()
# get_metrics()
