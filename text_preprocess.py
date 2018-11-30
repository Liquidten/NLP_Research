# read positive data
positive_dic = "/home/xsu1/lab/Opioid/text/yes/*.txt"
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

# clean
examples["notes"] = examples["notes"].str.lower()
examples["notes"] = examples["notes"].str.strip()
examples["notes"] = examples["notes"].replace("[^\w\s]", "")
stop = stopwords.words('english')
examples["notes"] = examples["notes"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# split data
X_train, X_test, y_train, y_test = train_test_split(examples["notes"],
                                                    examples["label"],
                                                    test_size=0.2,
                                                    random_state=2018)

# tfidf
vectorizer = TfidfVectorizer(encoding="utf-8", stop_words="english", norm="l2", analyzer="word")
vectorizer.fit(raw_documents=X_train)
X_train_matrix = vectorizer.transform(raw_documents=X_train)