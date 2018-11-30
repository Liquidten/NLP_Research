from sklearn.model_selection import GridSearchCV, cross_validate
from xgboost import XGBClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import cuis_preprocess

if __name__ == "__main__":
    xgboost = XGBClassifier(objective="binary:logistic", random_state=2018)
    search_space = {"max_depth": [7, 8, 9, 10],
                    "learning_rate": [0.0001, 0.001, 0.01, 1, 10],
                    "n_estimators": list(range(100, 1000, 100)),
                    "gamma": [0, 1, 5],
                    "min_child_weight": [4, 5, 7, 8],
                    "subsample": [0.8, 0.9, 1],
                    "colsample_bytree": [0.3, 0.5, 0.7, 0.8],
                    "reg_alpha": [0, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                    "reg_lambda": [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(estimator=xgboost,
                               param_grid=search_space,
                               scoring="roc_auc",
                               cv=10)
    grid_search.fit(X=cuis_preprocess.X_train_matrix,
                    y=cuis_preprocess.y_train.values)
    print(grid_search.best_params_)
    xgboost.set_params(**grid_search.best_params_)
    score = cross_validate(estimator=xgboost,
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