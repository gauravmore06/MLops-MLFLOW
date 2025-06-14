import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# load data set
wine = load_wine()
X = wine.data
y = wine.target

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# difine param for RF model
max_depth = 10
n_estimators = 10

with mlflow.start_run(experiment_id=141069861402566730):
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # creatiing a confusion metrix plot
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Metrix')

    #save plot
    plt.savefig('Confusion-matrix.png')

    # log artifacts using mlflow
    mlflow.log_artifact('Confusion-matrix.png')
    mlflow.log_artifact(__file__)


    print(accuracy)