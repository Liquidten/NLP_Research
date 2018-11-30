import cuis_preprocess
from sklearn.svm import LinearSVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,cross_validate




if __name__ == "__main__":
    SVC = LinearSVC(class_weight = 'balanced', random_state = 2018, decision_function_shape = 'ovo')
    search_space = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"]}
    grid_search = GridSearchCV(estimator=SVC,
                               param_grid=search_space,
                               scoring="roc_auc",
                               cv=10)
    grid_search.fit(X=cuis_preprocess.X_train_matrix,
                    y=cuis_preprocess.y_train)
    print(grid_search.best_params_)
    SVC.set_params(**grid_search.best_params_)
    score = cross_validate(estimator=SVC,
                           X=cuis_preprocess.X_train_matrix,
                           y=cuis_preprocess.y_train.values,
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
    
