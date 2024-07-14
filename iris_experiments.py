import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import pickle

import mlflow
import mlflow.sklearn

if __name__ == "__main__":

    # Read the data
    data = pd.read_csv("./datasets/iris.csv")

    X = data.iloc[:, 0:4]#'sepal.length','sepal.width,','petal.length','petal.width']
    y = data.iloc[:, 4]

    random_seed = 17
    validation_size = 0.25

    train, test = train_test_split(data, train_size=0.7, random_state=random_seed)

    X_train = train.iloc[:, 0:4]
    y_train = train.iloc[:, 4]
    X_test = test.iloc[:, 0:4]
    y_test = test.iloc[:, 4]

    # Start mlflow logging
    exp = mlflow.set_experiment(experiment_name="iris")
    mlflow.start_run()

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("LC_stages: {}".format(exp.lifecycle_stage))
    print("Creation: {}".format(exp.creation_time))

    tol = 0.01
    lr = LogisticRegression(tol=tol, random_state=random_seed)
    lr.fit(X_train, y_train)

    b_acc = balanced_accuracy_score(lr.predict(X_test), y_test)
    acc = accuracy_score(lr.predict(X_test), y_test)

    params = {
        "tol": tol,
        "random_seed": random_seed
    }

    mlflow.log_params(params)

    metrics = {
        "accuracy_score": acc,
        "balanced_accuracy_score": b_acc 
    }

    mlflow.log_metrics(metrics)

    X_train.to_csv("data_runs/X_train.csv", index=False)
    X_test.to_csv("data_runs/X_test.csv", index=False)
    y_train.to_csv("data_runs/y_train.csv", index=False)
    y_test.to_csv("data_runs/y_test.csv", index=False)

    pickle.dump(lr, open("./data_runs/lr.model", "wb"))

    mlflow.log_artifacts("./data_runs/")
    mlflow.sklearn.log_model(lr, "lr")

    mlflow.end_run()

