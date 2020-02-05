
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_1 = train.shape[0]
test_1 = test.shape[0]

p = train.describe()
corr = train.corr()

PassengerId = test['PassengerId']

class_survived = train['Survived'].copy()
train.drop(['Survived'], axis = 1, inplace = True)

#Junção dos dados
data = pd.concat(objs = [train, test], axis = 0).reset_index(drop = True)
data.info()

#Valores nulos
data.isnull().any()
data.isnull().sum()
    #Age
train_age_mediana = train['Age'].median()
train_age_media = np.round(train['Age'].mean())
data['Age'].fillna(train_age_mediana, inplace = True)
data['Age'] = data['Age'].astype(int)
    #Fare
train_fare_mediana = np.round(train['Fare'].median())
train_fare_media = np.round(train['Fare'].mean())
data['Fare'].fillna(train_fare_mediana, inplace = True)
data['Fare'] = data['Fare'].astype(int)
    #Embarked
train['Embarked'].describe()
valor_comum = 'S'
data['Embarked'].fillna(valor_comum, inplace = True)

#Exclusão das Variáveis que não serão utilizadas
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

#Tratar os dados para o modelo
        #Sex
data['Sex'] = data['Sex'].map({'male' : 0, 'female' : 1})
        #Age
           #Categorizando
pd.qcut(train['Age'], q = 6)
def Category_Age(a):
    if a <= 18: return 0
    elif 18 < a <= 23: return 1
    elif 23 < a <= 28: return 2
    elif 28 < a <= 34: return 3
    elif 34 < a <= 44: return 4
    elif 44 < a <= 80: return 5
    elif a > 80: return 5
data['Age_1'] = data['Age'].apply(Category_Age)
data.drop(['Age'], axis = 1, inplace = True)
        #Fare
            #Categorizando
pd.qcut(train['Fare'], q = 4)
def Category_Fare(f):
    if f <= 7.91: return 0
    elif 7.91 < f <= 14.454: return 1
    elif 14.454 < f <= 31: return 2
    elif 31 < f <= 100: return 3
    elif 100 < f <= 250: return 4
    elif f > 250: return 5
data['Fare_1'] = data['Fare'].apply(Category_Fare)
data.drop(['Fare'], axis = 1, inplace = True)
        #SibSp e Parch
            #Construindo Atributo
data['family'] = data['SibSp'] + data['Parch']
def Category_family(F):
    if F > 0: return 0
    elif F == 0: return 1  
data['not_alone'] = data['family'].apply(Category_family)
        #SibSp, Parch e Fare
            #Construindo Atributo
data['mean_ticket'] = data['Fare_1']/(data['family']+1)
data['mean_ticket'] = data['mean_ticket'].astype(int)
        #Age e Pclass 
            #Construindo Atributo
data['Age_Pclass'] = data['Age_1'] * data['Pclass']
        #Embarked
w = pd.value_counts(data['Embarked'])
data['Embarked'] = data['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2})


#Dados
train = data.iloc[:train_1]
test = data.iloc[train_1:]

train = pd.concat(objs = [train, class_survived], axis = 1).reset_index(drop = True)
previsores = train.iloc[:, 0:11].values
classe = train.iloc[:, 11].values

#Separação dos Dados
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.25,
                                                                  random_state = 0)
#Algoritmo - SVC - 81,61%
    #Criando o modelo
svm = SVC()
svm.fit(X_treinamento, y_treinamento)
    #Previsões
previsoes_SVC = svm.predict(X_teste)
    #Taxa de acerto
confusao_SVC = confusion_matrix(y_teste, previsoes_SVC)
taxa_acerto_SVC = accuracy_score(y_teste, previsoes_SVC)    

#Algoritmo - Decision Tree - 80,71%
    #Criando o modelo
tree = DecisionTreeClassifier(max_depth = 3)
tree.fit(X_treinamento, y_treinamento)
    #Previsões
previsoes_tree = tree.predict(X_teste)
    #Taxa de acerto
confusao_tree = confusion_matrix(y_teste, previsoes_tree)
taxa_acerto_tree = accuracy_score(y_teste, previsoes_tree)

#Algoritmo - Random Forest - 81,61%
    #Criando o modelo
forest = RandomForestClassifier(n_estimators = 100)
forest.fit(X_treinamento, y_treinamento)
    #Previsões
previsoes_forest = forest.predict(X_teste)
    #Taxa de acerto
confusao_forest = confusion_matrix(y_teste, previsoes_forest)
taxa_acerto_forest = accuracy_score(y_teste, previsoes_forest)


#Algoritmo - XGBoost - 83,85%
    #Criando o modelo
gmb = xgb.XGBClassifier(max_depth = 3, n_estimators = 300, learning_rate = 0.05)
gmb.fit(X_treinamento, y_treinamento)
    #Previsoes
previsoes_gmb = gmb.predict(X_teste)
    #Verificação do modelo - Taxa de Acerto do modelo
confusao_gmb = confusion_matrix(y_teste, previsoes_gmb)
taxa_acerto_gmb = accuracy_score(y_teste, previsoes_gmb)


#Algoritmo - Logistic Regression - 78.92
    #Criando o modelo
model = LogisticRegression()
model.fit(X_treinamento, y_treinamento)
    #Previsores
previsoes_logistic = model.predict(X_teste)
    #Verificação do modelo
confusao_logistic = confusion_matrix(y_teste, previsoes_logistic)
taxa_acerto_logistic = accuracy_score(y_teste, previsoes_logistic)

#ALgoritmo - KNN - 78,92%
    #Criando o modelo
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_treinamento, y_treinamento)
    #Previsores
previsoes_knn = knn.predict(X_teste)
    #Verificação do modelo
confusao_knn = confusion_matrix(y_teste, previsoes_knn)
taxa_acerto_knn = accuracy_score(y_teste, previsoes_knn)

#Submetendo a variável Test.csv
test.info()
test_submission = test.values
gmb = xgb.XGBClassifier(max_depth = 3, n_estimators = 300, learning_rate = 0.05)
gmb.fit(X_treinamento, y_treinamento)
previsoes_submission = gmb.predict(test_submission)
rferraz_submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': previsoes_submission })
rferraz_submission.to_csv('rferraz_submission', index = False)


