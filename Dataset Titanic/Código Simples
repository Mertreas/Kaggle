#importando as bibliotecas
import pandas as pd
import numpy as np

#carregar o dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#mostrando o dataset
train.head()

#importando o algoritmo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Atribuir binário ao sexo
def binario_sex(valor):
    if valor == 'male':
        return 0
    else:
        return 1
    
train['Binario'] = train['Sex'].map(binario_sex)
train.head()

#Atribuindo as variáveis selecionadas
var = ['Binario', 'Age']

#x recebendo "var" e y os sobreviventes
x = train[var]
y = train['Survived']

#Adicionando valores que estão faltando para não dar erro
x = x.fillna (-1)

#Transformando a variavel "test"
test['Binario'] = test['Sex'].map(binario_sex)

#Criando uma semente pra aletoriedade
np.random.seed(0)
x_treino, x_val, y_treino, y_val = train_test_split(x, y, test_size = 0.5)

#O tamanho dos shapes
x_treino.shape, x_val.shape, y_treino.shape, y_val.shape

#Treinar os modelos
model = RandomForestClassifier (n_estimators = 100, n_jobs = -1, random_state = 0)
model.fit(x_treino, y_treino)

#Comparando a predição com o y válido
pred = model.predict(x_val)
np.mean(y_val == pred)
