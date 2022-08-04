#!/usr/bin/env python
# coding: utf-8

# # IMPORTAR PACOTES

# In[122]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# # IMPORTAR DADOS

# In[123]:


df = pd.read_csv('train.csv', sep=',')


# In[124]:


df.info()
# Removemos o bloqueio de colunas para poder visualizar 
# todas as colunas do dataset
pd.set_option('display.max_columns', None)


# ### Descrever os dados
# 
# Fazemos uma descrição dos atributos do dataframe

# In[125]:


df.head()


# In[126]:


df.describe()


# ### Remoção de variáveis criptografadas
# 
# Removemos as variáveis que são chaves, senhas e ids criptografados
# 

# In[127]:


del df['023c68873b'], df['361f93f4d1'], df['8d0606b150'], df['91145d159d'], df['b835dfe10f'], df['e16e640635'], df['f1f0984934'], df['id']


# In[128]:


df.head()


# In[129]:


df.info()


# ## CRIANDO A REGRESSÃO LINEAR

# ### Selecionando variáveis para o modelo
# 
# Após a seleção do X e y, vamos dividir nossos dados em treino e teste para que possamos após a criação do modelo testar a performance do mesmo.

# Determinamos as variáveis de correlação

# In[130]:


import seaborn as sn


# In[131]:


pd.reset_option('^display.', silent=True)


# In[132]:


# análise de correlação
correlation = df.corr()


# In[133]:


# plot da matriz de correlação
plt.figure(figsize=(60, 60))
plot = sn.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
plot


# In[134]:


# X -> Vairiável com maior coeficiente de correlação
X = df[['96c30c7eef']]
y = df.target


# ### Dividindo os dados em treino e teste
# 
# O parâmetro **test_size** vai definir o tamanho dos nossos dados selecionados para teste, o tamanho dessa divisão, não existe uma regra para isso, vai depender de cada problema e principalmente do tamanho do conjunto de dados que temos para treino.

# In[135]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# ### Criando o modelo
# 
# Com nossos dados de treino dividido, podemos criar o modelo, que é uma tarefa relativamente simples, na prática o mais difícil é saber: Qual melhor modelo usar? Quais são os melhores hiperparâmetros?

# In[136]:


# Criando o modelo LinearRegression
regr = LinearRegression(copy_X=True,fit_intercept=True,n_jobs=None)

# Realizar treinamento do modelo
regr.fit(X_train, y_train)


# In[137]:


# Realizar predição com os dados separados para teste
y_pred = regr.predict(X_test)

# Visualização dos 20 primeiros resultados
y_pred[:20]


# ### Erro Médio Absoluto (Mean Absolute Error)
# 
# O erro médio absoluto (MAE) é a média da soma de todos os e do nosso gráfico de erros, as sua análise sofre uma interferência devido aos erros positivos e negativos se anularem.

# In[138]:


print('MAE: %.2f' % mean_absolute_error(y_test, y_pred))


# ### Erro Quadrado Médio (Mean Squared Error)
# 
# O erro quadrado médio (MSE) é a média da soma de todos os e elevados ao quadrado do nosso gráfico, o fato de ele ter as diferenças elevadas ao quadrados resolve o problema de os erros positivos e negativos se anulam, sendo mais preciso que o MAE.

# In[139]:


print('MAE: %.2f' % mean_absolute_error(y_test, y_pred))


# ### Erro Quadrado Médio (Mean Squared Error)
# 
# O erro quadrado médio (MSE) é a média da soma de todos os e elevados ao quadrado do nosso gráfico, o fato de ele ter as diferenças elevadas ao quadrados resolve o problema de os erros positivos e negativos se anulam, sendo mais preciso que o MAE.

# In[140]:


print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))


# ### Coeficiente de Determinação (R2 Score)
# 
# O coeficiente de Determinação (R²) varia entre 0 e 1 e expressa a quantidade da variância dos dados que é explicada pelo modelo linear. Explicando a variância da variável dependente a partir da variável independente.
# 
# No nosso exemplo o R² = 0,56 significa que o modelo linear explica 56% da variância da variável dependente a partir da variável independente.

# In[141]:


print('R2 Score: %.2f' % r2_score(y_test, y_pred))


# ## VISUALIZANDO OS RESULTADOS

# In[142]:


plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()


# ### Random Florest

# In[143]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import math


# In[157]:



#Dividir a base de treino e teste

### Peguei as variaveis que tem maior correlação com o target 
X = df.drop(df.columns.difference(["fe8cdd80ba","f66b98dd69","f013b60e50","ed7e658a27","ea0f4a32e3","e86a2190c1","e0a0772df0","d2c775fa99","aee1e4fc85","96c30c7eef","8a21502326","2719b72c0d"]),axis=1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Criaremos 30 arvores, e o tamanho mínimo delas(leaf) será 10, como a base é relativamente grande será bom para a predição.
#n_jobs = -1 significa que usarei todas as threads do meu processador

arvores = RandomForestRegressor(n_estimators=1000,min_samples_leaf=10,random_state=0,n_jobs=-1)
arvores.fit(X_train,y_train)

predict  = arvores.predict(X_test)

# Verificar qual o valor de erro do nosso modelo
print('Erro: ', metrics.mean_squared_error(y_test, predict))





# In[173]:


# Aqui vou salvar os dados preditados
df = pd.read_csv('train.csv', sep=',')
X_final = df.drop(df.columns.difference(["fe8cdd80ba","f66b98dd69","f013b60e50","ed7e658a27","ea0f4a32e3","e86a2190c1","e0a0772df0","d2c775fa99","aee1e4fc85","96c30c7eef","8a21502326","2719b72c0d"]),axis=1)

df["predict"] = arvores.predict(X_final)
df.to_csv("testFinal.csv")


# In[174]:


pd.set_option('display.max_columns', None)
df


# ### Conclusão
#  Como as variáveis não têm uma correlação muito boa com a target, acreditamos que o melhor caminho para preditar esse dataset, realmente seja o random florest com um erro de 5%, se aumentarmos as arvores para 1000, conseguimos chegar aos 4%
# 

# In[ ]:




