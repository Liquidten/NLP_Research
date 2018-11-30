import glob
import panadas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# untility


def dictMean(dictionary):
    average = {}
    for k, v in dictionary.iteritems():
        average[k] = sum(v)/float(len(v))
    return average


# read positive
positive_dic = "/home/xsu1/lab/Opioid/CUIS/yes/*.txt"
positive_files = glob.glob(positive_dic)
positive_examples = []
for file in positive_files:
    with open(file, "r") as myfile:
        data = myfile.read().replace("\n", "")
        positive_examples.append(data)
positive_df = pd.DataFrame()
positive_df["notes"] = positive_examples
positive_df["label"] = 1

# read negative data
negative_dic = "/home/xsu1/lab/Opioid/text/no/*.txt"
negative_files = glob.glob(negative_dic)
negative_examples = []
for file in negative_files:
    with open(file, "r") as myfile:
        data = myfile.read().replace("\n", "")
        negative_examples.append(data)
negative_df = pd.DataFrame()
negative_df["notes"] = negative_examples
negative_df["label"] = -1

# concat
examples = pd.concat(objs=[positive_df, negative_df],
                     axis=0)

# train test split
X_train, X_test, y_train, y_test = train_test_split(examples["notes"].values,
                                                    examples["label"].values,
                                                    test_size=0.2,
                                                    random_state=2018)

# bag of words
tfidf = TfidfVectorizer(analyzer='word',
                        norm='l2',
                        use_idf=True,
                        smooth_idf=True,
                        lowercase=False)
tfidf.fit(raw_documents=X_train)
X_train_matrix = tfidf.transform(raw_documents=X_train)

tfidf.fit(raw_documents=X_test)
X_test_matrix = tfidf.transform(raw_documents=X_test)




