import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay,f1_score,accuracy_score, classification_report, RocCurveDisplay

def readf(file):
    data = pd.read_csv(file)
    data = data.drop(columns = ['Unnamed: 32','id'])
    data['diagnosis'] = [1 if each == "M" else 0 for each in data.diagnosis]
    t = data['diagnosis'].values
    X = (data.drop(columns = 'diagnosis'))
    X = X.drop(['smoothness_worst','perimeter_worst','radius_worst'],axis =1)

    M = data[data.diagnosis == 1]
    B = data[data.diagnosis == 0]
    plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Malign",alpha= 0.5)
    plt.scatter(B.radius_mean,B.texture_mean,color="green",label="Benign",alpha= 0.3)
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    plt.legend()
    plt.savefig('data.png')
    plt.show()

    return X, t

##'hyperparamenters'
alpha   = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001,0.0003, 0.0001]
K_range = range(1,26)
MaxIter = 3000

##'Spliting the data into test and training set'

file = "breast_cancer.csv"
X,t = readf(file)
X_train,X_test, t_train, t_test = train_test_split(X, t, test_size=0.25,random_state = 3)

X_train = StandardScaler().fit(X_train).transform(X_train)
X_test = StandardScaler().fit(X_test).transform(X_test)



##'Training Logistic Regeression Model'
LR_cross_val = []
stdlr = []
acc_max_alpha = -100

for i in alpha:

    LR = SGDClassifier(loss = 'log',max_iter = MaxIter,shuffle = False,
                   learning_rate = 'constant', eta0 = i)

    lr_cv = cross_val_score(LR, X_train, t_train ,scoring = 'f1', cv = 10).mean()
    LR_cross_val.append(lr_cv.mean())
    if lr_cv > acc_max_alpha:
        acc_max_alpha = lr_cv
        a = i

plt.plot(alpha,LR_cross_val)
plt.xscale('log')
plt.title('Cross Val Scores for Logistic regression on different Learning rates')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.savefig('Cross_val_log_alpha.png')
plt.show()

LR = LR.set_params(eta0 = a)
LR = LR.fit(X_train,t_train)
t_hat1 = LR.predict(X_test)

t1 =  LR.predict(X_train)
LR_train = f1_score(t_train, t1)
LR_report = classification_report(t_test , t_hat1)



RocCurveDisplay.from_predictions(t_test, t_hat1)
plt.savefig('ROC_log.png')
plt.show()

ConfusionMatrixDisplay.from_predictions(t_test, t_hat1,display_labels = ['Benign','Malignant'])
plt.savefig('Confusion_matrix_log.png')
plt.show()

print('-------------------------------------------------------')
print('Train F-1 Accuracy for Logistic Regression = '+str(LR_train))
print(LR_report)

##'Training K-Nearest Neighbours Model'
KNN_cross_val = []
acc_max_k = -100

for i in K_range:

    KNN = KNeighborsClassifier(n_neighbors = i)

    knn_cv = cross_val_score(KNN,  X_train, t_train,scoring = 'f1', cv = 10).mean()
    KNN_cross_val.append(knn_cv)
    if knn_cv > acc_max_k:
        acc_max_k = knn_cv
        K = i

plt.plot(K_range,KNN_cross_val)
plt.title('Cross Val Scores for different Neighbour values')
plt.xlabel('Number of Neighbous to consider')
plt.ylabel('Accuracy')
plt.savefig('Cross_Val_KNN_K.png')
plt.show()


KNN = KNN.set_params(n_neighbors = K)
KNN = KNN.fit(X_train,t_train)
t_hat2 = KNN.predict(X_test)

t2 =  KNN.predict(X_train)
KNN_train = f1_score(t_train, t2)
KNN_report = classification_report(t_test , t_hat2)


RocCurveDisplay.from_predictions(t_test, t_hat2)
plt.savefig('ROC_KNN.png')
plt.show()
ConfusionMatrixDisplay.from_predictions(t_test, t_hat2, display_labels = ['Benign','Malignant'])
plt.savefig('Confusion_matrix_KNN.png')
plt.show()

print('-------------------------------------------------------')
print('Train F-1 Accuracy for KNN= '+str(KNN_train))
print(KNN_report)


##'Training Neural Networks, Multi Layer Perceptron Model'

MLP_cross_val = []
acc_max_alpha = -100
for i in alpha:

    MLP = MLPClassifier(solver = 'sgd',hidden_layer_sizes = (50,1),
                    activation = 'tanh',max_iter= MaxIter,
                    learning_rate = 'constant',learning_rate_init = i, shuffle = False)
    
    MLP_cv = cross_val_score(MLP, X_train, t_train,scoring = 'f1', cv = 10).mean()
    MLP_cross_val.append(MLP_cv)
    if MLP_cv > acc_max_alpha:
        acc_max_alpha = MLP_cv
        a = i

plt.plot(alpha ,MLP_cross_val)
plt.xscale('log')
plt.title('Cross Val Scores for Multi Layer Perceptron on different Learning rates')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.savefig('Cross_Val_MLP_alpha.png')
plt.show()


MLP = MLP.set_params(learning_rate_init = a)
MLP = MLP.fit(X_train,t_train)
t_hat3 = MLP.predict(X_test)

t3 =  MLP.predict(X_train)
MLP_train = f1_score(t_train, t3)
MLP_report = classification_report(t_test , t_hat3)

RocCurveDisplay.from_predictions(t_test, t_hat3)
plt.savefig('ROC_MLP.png')
plt.show()

ConfusionMatrixDisplay.from_predictions(t_test, t_hat3,display_labels = ['Benign','Malignant'])
plt.savefig('Confusion_matrix_MLP.png')
plt.show()

print('-------------------------------------------------------')
print('Train F-1 Accuracy for MLP= '+str(MLP_train))
print(MLP_report)

