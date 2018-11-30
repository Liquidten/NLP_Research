from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import ensemble
from sklearn import metrics
import cuis_preprocess
import matplotlib.pyplot as plt

if __name__ == "__main__":
    RFC = ensemble.RandomForestClassifier(random_state = 2018, criterion='gini')
    search_space = {"max_depth": [7, 8, 9, 10],
                    "criterion": ['gini','entropy'],
                    "n_estimators": list(range(10, 100, 1000)),
                    "max_features": ['log2','sqrt','auto'],
                    "min_samples_split": [2, 4, 6],
                    "min_samples_leaf": [1,2]}
    grid_search = GridSearchCV(estimator=RFC,
                               param_grid=search_space,
                               scoring="roc_auc",
                               cv=10)
    grid_search.fit(X=cuis_preprocess.X_train_matrix,
                    y=cuis_preprocess.y_train.values)
    print(grid_search.best_params_)
    RFC.set_params(**grid_search.best_params_)
    score = cross_validate(estimator=RFC,
                           X=cuis_preprocess.X_train_matrix,
                           y=cuis_preprocess.y_train,
                           scoring=["roc_auc", "precision","recall", "f1_micro", "f1_macro", "f1_weighted", "accuracy"],
                           cv=10,
                           return_train_score=False)
    ave_score = cuis_preprocess.dictMean(score)
    print(ave_score)
    
    predictions = grid_search.predict(cuis_preprocess.X_test_matrix)
    
    fpr, tpr, _ = metrics.roc_curve(cuis_preprocess.y_test , predictions)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
