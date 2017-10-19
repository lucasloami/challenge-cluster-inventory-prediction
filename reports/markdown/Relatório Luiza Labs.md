
# Relatório - Predição de interesse em produtos

Este notebook tem dois objetivos principais: criar uma clusterização dos produtos vendidos pelo Magazine Luiza a partir do histórico de vendas e prever a venda de cada um dos seus produtos. Outros objetivos desse estudo incluem a análise da qualidade dos clusters e compreensão das principais características dos mesmos.

Neste projeto optou-se por utilizar o algoritmo KMeans para clusterização e o algoritmo ARIMA para a predição. 

Este documento é dividido da seguinte forma:

1. Análise exploratória
2. Implementação e análise do cluster
3. Implementação e análise do modelo preditivo
4. Conclusões
5. Referências

As linhas de código abaixo apenas carregam o dataset em memória e definem algumas funções úteis para esse projeto. 


```python
import pandas as pd
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.vq import kmeans,vq
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


sns.set(color_codes=True)

df = pd.read_csv("data/desafio.csv")
```


```python
# HELPER FUNCTION DEFINITIONS
def plot_elbow_curve(X, K):
    # scipy.cluster.vq.kmeans
    KM = [kmeans(X,k) for k in K]
    centroids = [cent for (cent,var) in KM]   # cluster centroids
    avgWithinSS = [var for (cent,var) in KM] # mean within-cluster sum of squares

    # alternative: scipy.cluster.vq.vq
    #Z = [vq(X,cent) for cent in centroids]
    #avgWithinSS = [sum(dist)/X.shape[0] for (cIdx,dist) in Z]

    # alternative: scipy.spatial.distance.cdist
#     D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
#     cIdx = [np.argmin(D,axis=1) for D in D_k]
#     dist = [np.min(D,axis=1) for D in D_k]
#     avgWithinSS = [sum(d)/X.shape[0] for d in dist]

    ##### plot ###
    kIdx = 3
    # elbow curve
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()
```

## Análise Exploratória 

Nesta seção exploraremos as características básicas do dataset, não apenas suas dimensões mas também as características de suas variáveis. 

O dataset possui `179149` linhas e `14` variáveis (um exemplo do mesmo é mostrado abaixo). Dessas variáveis, 8 são categóricas e as demais são contínuas. Percebe-se também que todas as variáveis tem preenchimento (densidade) completo dentro do dataset. Porém, a variável `process_date` contém células com o valor `0000-00-00`, que é um formato inválido de data. Esse valor precisará ser tratado para aplicação dos algoritmos que a utilizarem como feature. 


```python
print("Dimensões do dataset: {} rows e {} columns".format(df.shape[0], df.shape[1]))
num_cat_features = len( df.select_dtypes( include=['object'] ).columns )
print("Número de variáveis categóricas: {}".format( num_cat_features ) )
print("Número de variáveis contínuas: {}".format( len(df.columns) - num_cat_features ) )

def calculate_density(df):
  sparsity_dict = {}
  for col in df.columns:
    sparsity_dict[col] = df[col].count()/df.shape[0]
  return(sparsity_dict)

density_dict = calculate_density(df)
sorted_density_list = sorted(density_dict.items(), key=lambda x: x[1])

print("\n")
for l in sorted_density_list:
  print("{:25} => Density of {:4.2f}%".format(l[0], l[1]*100))
```

    Dimensões do dataset: 179149 rows e 14 columns
    Número de variáveis categóricas: 8
    Número de variáveis contínuas: 6
    
    
    category                  => Density of 100.00%
    code                      => Density of 100.00%
    order_id                  => Density of 100.00%
    tax_substitution          => Density of 100.00%
    price                     => Density of 100.00%
    capture_date              => Density of 100.00%
    process_status            => Density of 100.00%
    order_status              => Density of 100.00%
    icms                      => Density of 100.00%
    pis_cofins                => Density of 100.00%
    process_date              => Density of 100.00%
    liquid_cost               => Density of 100.00%
    source_channel            => Density of 100.00%
    quantity                  => Density of 100.00%



```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>code</th>
      <th>quantity</th>
      <th>price</th>
      <th>pis_cofins</th>
      <th>icms</th>
      <th>tax_substitution</th>
      <th>category</th>
      <th>liquid_cost</th>
      <th>order_status</th>
      <th>capture_date</th>
      <th>process_date</th>
      <th>process_status</th>
      <th>source_channel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bcb59c839e78b2601374cbad9239ca7b</td>
      <td>e6762ba2ffbca07ab6cee7551caeaad5</td>
      <td>1</td>
      <td>978.90</td>
      <td>90.5483</td>
      <td>0.0000</td>
      <td>191.8416</td>
      <td>4ece547755cba9e7fc14125bc895f31b</td>
      <td>542.7065</td>
      <td>entrega total</td>
      <td>2016-06-11</td>
      <td>2016-06-11</td>
      <td>processado</td>
      <td>b76eb9b8fc0f17098812da9117d3e500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4e91ee6b95895771dc9ee524e910a902</td>
      <td>e6762ba2ffbca07ab6cee7551caeaad5</td>
      <td>1</td>
      <td>1036.29</td>
      <td>95.8568</td>
      <td>176.1693</td>
      <td>0.0000</td>
      <td>4ece547755cba9e7fc14125bc895f31b</td>
      <td>542.7065</td>
      <td>em rota de entrega</td>
      <td>2016-06-11</td>
      <td>2016-06-11</td>
      <td>processado</td>
      <td>b76eb9b8fc0f17098812da9117d3e500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>88eb0ac86af1a521c0831298d22dea8b</td>
      <td>e6762ba2ffbca07ab6cee7551caeaad5</td>
      <td>1</td>
      <td>978.90</td>
      <td>90.5483</td>
      <td>0.0000</td>
      <td>191.8416</td>
      <td>4ece547755cba9e7fc14125bc895f31b</td>
      <td>542.7065</td>
      <td>entrega total</td>
      <td>2016-06-12</td>
      <td>2016-06-12</td>
      <td>processado</td>
      <td>b76eb9b8fc0f17098812da9117d3e500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dee418152a36314b4aee6ce9cf94fcbf</td>
      <td>e6762ba2ffbca07ab6cee7551caeaad5</td>
      <td>1</td>
      <td>978.90</td>
      <td>90.5483</td>
      <td>176.2020</td>
      <td>0.0000</td>
      <td>4ece547755cba9e7fc14125bc895f31b</td>
      <td>542.7065</td>
      <td>cancelado</td>
      <td>2016-06-13</td>
      <td>0000-00-00</td>
      <td>captado</td>
      <td>b76eb9b8fc0f17098812da9117d3e500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1c175bc61b9b659bbf011b2e5e3dcec6</td>
      <td>e6762ba2ffbca07ab6cee7551caeaad5</td>
      <td>1</td>
      <td>976.05</td>
      <td>90.2846</td>
      <td>0.0000</td>
      <td>192.3325</td>
      <td>4ece547755cba9e7fc14125bc895f31b</td>
      <td>542.7065</td>
      <td>entrega total</td>
      <td>2016-06-13</td>
      <td>2016-06-13</td>
      <td>processado</td>
      <td>b76eb9b8fc0f17098812da9117d3e500</td>
    </tr>
  </tbody>
</table>
</div>



Um dos objetivos desse estudo é prever a quantidade de vendas para cada um dos produtos. Dado isso, é importante ver como as vendas se comportam ao longo dos meses de maneiral total.

A partir do gráfico abaixo é possível ver que existem dois picos de vendas de produtos, um deles no dia `25 de novembro de 2016` e outro no dia `06 de janeiro de 2017`. Provavelmente o primeiro pico se deve ao evento da Black Friday e o segundo pico se deve ao evento de "Liquidação Fantástica do Magazine Luiza" [5]. Nos demais dias a curva de vendas permanece sem grandes oscilações.

Outro fator interessante é que, apesar de as duas data acima serem destaque em quantidades de produtos vendidos elas não representam os dias mais deram receita a empresa (`revenue = quantity * price`). O dia mais rentável foi `29 de setembro de 2016`. As duas datas acima ainda sim permanecem entre os top 5 dias mais rentáveis.  


```python
df['process_date'] = df['process_date'].apply(lambda x: np.nan if x == '0000-00-00' else x)
df['capture_date'] = pd.to_datetime(df['capture_date'], format="%Y/%m/%d")
df['process_date'] = pd.to_datetime(df['process_date'], format="%Y/%m/%d")

ts_quantity = df.groupby(pd.Grouper(key='capture_date', freq='D'))['quantity'].aggregate(np.sum)

df['revenue'] = df['quantity'] * df['price']

ts_revenue = df.groupby(pd.Grouper(key='capture_date', freq='D'))['revenue'].aggregate(np.sum)

plt.figure(figsize=(16,5))
plt.title("Sales x Day")
ts_quantity.plot()
plt.show()

plt.figure(figsize=(16,5))
plt.title("Revenue x Day")
ts_revenue.plot()
plt.show()
```


![png](output_7_0.png)



![png](output_7_1.png)


## Implementação e análise do cluster

Nesta seção construiremos o cluster. O algoritmo escolhido para cumprir essa atividade é o KMeans. Para isso, é preciso analisar como as características deste algoritmo  impactam na resolução deste problema. 

### Características do KMeans

De maneira geral, algoritmos de cluster não conseguem lidar tão bem com variáveis categóricas, pois é difícil achar uma função de distância que consiga representá-las. 

Por exemplo, dado a variável *COR* pertencente ao dataset D e que pode ter os valores *[vermelho, verde, azul, amarelo]*, é impossível dizer o quão próximo a cor `vermelho` está da cor `azul`. O KMeans se baseia na distância euclidiana entre as variáveis para montar seus clusters e esta medida, por definição, funciona apenas com variáveis contínuas. Isso nos permite dizer que deveremos tratar as variáveis categóricas presentes no dataset afim de clusterizar os produtos da melhor maneira possível. 

Após pesquisas realizadas, identificou-se duas formas principais de lidar com essa situação:

1. **1-of-K Coding** [3]: consiste em transformar todas as variáveis categóricas em _dummy vars_ e aplicar o KMeans considerando essas novas variáveis como contínuas. Esse método, apesar de ter resultados satisfatórios, aumenta drasticamente o número de variáveis presentes no dataset o que pode prejudicar a performance do algoritmo e também gerar overfitting (quando o cluster falha ao generalizar os agrupamentos de dados). 

2. **K-prototypes** [1][2][4]: esse algoritmo é baseado no KMeans e no KModes. É adaptado para lidar com um mix de variáveis categóricas e contínuas, usando medidas de distância adequadas para cada um desses tipos. Os trade-offs desse método se encontram na sua implementação em python [6], que possui baixa performance quando o dataset tem mais de dez mil linhas, e na falta de clareza se os métodos de validação do KMeans funcionam para ele também (nos artigos analisados essa informação não está totalmente clara). 

### Ajustes no dataset

O dataset utilizado neste projeto versa sobre o histórico de vendas e não apenas sobre as características dos produtos. Para montar a clusterização, é preciso que o dataset tenha unicidade por produto em cada uma das suas linhas e que as colunas versem sobre características dos mesmos. 

O código abaixo demonstra como o dataset de produtos foi montado. As premissas utilizadas nessa atividade foram:

1. Manter todas as variáveis numéricas já existentes no novo dataset
2. Remover as variáveis com hash, pois sua interpretação dentro do cluster não é viável, dado que não é possível identificar o significado da mesma
3. Contabilizar quantas vezes cada um dos possíveis valores das features categóricas aconteceram dentro do dataset

A partir dessas premissas é possível utilizar o KMeans sem problemas, tendo em vista que todas as variáveis serão contínuas. 


```python
df_order_status = pd.get_dummies(df['order_status'], prefix="order_status")
df_process_status = pd.get_dummies(df['process_status'], prefix="process_status")

df2 = pd.concat([df, df_order_status], axis=1)
df2 = pd.concat([df2, df_process_status], axis=1)


df_grouped = df2.groupby('code').agg({'quantity': np.mean, 'price': np.mean, 
                                     'pis_cofins': np.mean, 'tax_substitution': np.mean,
                                    'liquid_cost': np.mean, 'revenue': np.mean}).reset_index()

df_grouped2 = df2.groupby('code').agg(np.sum).reset_index()
df_grouped2 = df_grouped2[df_grouped2.columns.difference(['quantity', 'price', 'pis_cofins', 'icms',
                                                         'tax_substitution', 'liquid_cost', 'revenue'])]
# df_grouped2.head()
product_dataset = df_grouped.merge(df_grouped2, how='inner', on='code') 

product_dataset.head()

# EXPORT DATASET TO CSV
# product_dataset.to_csv("data/product_dataset.csv", index=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>revenue</th>
      <th>tax_substitution</th>
      <th>price</th>
      <th>pis_cofins</th>
      <th>liquid_cost</th>
      <th>quantity</th>
      <th>order_status_cancelado</th>
      <th>order_status_cancelado boleto não pago</th>
      <th>order_status_cancelado dados divergentes</th>
      <th>...</th>
      <th>order_status_entrega total</th>
      <th>order_status_fraude confirmada</th>
      <th>order_status_pendente processamento</th>
      <th>order_status_processado</th>
      <th>order_status_solicitação de cancelamento</th>
      <th>order_status_solicitação de troca</th>
      <th>order_status_suspeita de fraude</th>
      <th>order_status_suspenso barragem</th>
      <th>process_status_captado</th>
      <th>process_status_processado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0671c2b9132a3f5215a4212ce0691694</td>
      <td>225.445203</td>
      <td>13.551533</td>
      <td>213.213890</td>
      <td>19.673568</td>
      <td>117.0820</td>
      <td>1.020166</td>
      <td>182.0</td>
      <td>1751.0</td>
      <td>28.0</td>
      <td>...</td>
      <td>4151.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>32.0</td>
      <td>28.0</td>
      <td>93.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1970.0</td>
      <td>4526.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09f544ec2a74c89abeec7b0590fc2d11</td>
      <td>220.022618</td>
      <td>11.664623</td>
      <td>145.200961</td>
      <td>13.410406</td>
      <td>73.8002</td>
      <td>1.085890</td>
      <td>18.0</td>
      <td>89.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>809.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>107.0</td>
      <td>871.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0ad316f6b5cb5e81ebff73ae2490ccfe</td>
      <td>219.997219</td>
      <td>11.489706</td>
      <td>210.690798</td>
      <td>19.373709</td>
      <td>106.4842</td>
      <td>1.020450</td>
      <td>8.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>411.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>49.0</td>
      <td>440.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0bbe09e34a11e8e31cf49d6f8df2992d</td>
      <td>187.256657</td>
      <td>4.116640</td>
      <td>167.754106</td>
      <td>15.491333</td>
      <td>88.9639</td>
      <td>1.023460</td>
      <td>8.0</td>
      <td>40.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>266.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>51.0</td>
      <td>290.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0dca7ec6ba9b6e8f17f04f713a6be727</td>
      <td>149.058133</td>
      <td>2.273317</td>
      <td>68.569957</td>
      <td>6.341866</td>
      <td>27.2847</td>
      <td>1.419528</td>
      <td>22.0</td>
      <td>115.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>691.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>145.0</td>
      <td>787.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



### Implementação do KMeans

Um dos grandes desafios de implentar um algoritmo de clusterização é determinar qual o número ideal de clusters a ser montados. Após pesquisas realizadas, identificou-se duas formas principais de lidar com essa situação:

1. **Silhouette Análise** [7]: pode ser compreendida como o estudo da distância de separação entre os clusters resultantes do KMeans. O gráfico de silhueta exibe uma medida de quão próximo cada ponto em um cluster é para pontos nos clusters vizinhos e, portanto, fornece uma maneira de avaliar visualmente os parâmetros como número de clusters. Uma medida numérica utilizada é a _average silhouette_, que é a média dos valores das silhuetas dos clusters. Quanto maior o valor da _average silhouette_ melhor o cluster. 

2. **Elbow method** [8][9]: este método analisa a porcentagem de variância explicada como uma função do número de clusters: é preciso escolher uma série de clusters para que a adição de outro cluster não ofereça muito melhor modelagem dos dados. Mais precisamente, se uma parcela a porcentagem de variância explicada pelos clusters em relação ao número de clusters, os primeiros clusters adicionarão muita informação (explicam muita variância), mas em algum momento o ganho marginal cairá, dando um ângulo na gráfico.

Neste estudo utilizaremos o segundo método para identificação do melhor valor de `k` para o KMeans. A partir da implementação abaixo, é possível ver que os melhores valores para K são `4` e `5`. Vamos escolher arbitrariamente o valor `4` e implementar o algoritmo. Após isso, vamos associar as labels produzidas ao dataset de produtos montado anteriormente. 

É importante frisar que todas as varáiveis foram normalizadas utilizando o `MinMaxScaler` do scikit-learn. Isso permite que todas as features fiquem em um mesmo patamar de comparação. 


```python
# REMOVE PRODUCT ID COLUMN. IT IS NOT NECESSARY TO BUILD THE CLUSTER
df3 = product_dataset[product_dataset.columns.difference(['code'])]

min_max_scaler = MinMaxScaler()
# PERFORM FEATURE SCALING
df3 = min_max_scaler.fit_transform(df3)
plot_elbow_curve(df3, range(1,21))
```


![png](output_11_0.png)



```python
kmeans = KMeans(init='k-means++', n_clusters=4, random_state=30).fit(df3)
labels = kmeans.labels_
product_dataset['clusters'] = labels
product_dataset[product_dataset.clusters == 2].head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>revenue</th>
      <th>tax_substitution</th>
      <th>price</th>
      <th>pis_cofins</th>
      <th>liquid_cost</th>
      <th>quantity</th>
      <th>order_status_cancelado</th>
      <th>order_status_cancelado boleto não pago</th>
      <th>order_status_cancelado dados divergentes</th>
      <th>...</th>
      <th>order_status_fraude confirmada</th>
      <th>order_status_pendente processamento</th>
      <th>order_status_processado</th>
      <th>order_status_solicitação de cancelamento</th>
      <th>order_status_solicitação de troca</th>
      <th>order_status_suspeita de fraude</th>
      <th>order_status_suspenso barragem</th>
      <th>process_status_captado</th>
      <th>process_status_processado</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>09f544ec2a74c89abeec7b0590fc2d11</td>
      <td>220.022618</td>
      <td>11.664623</td>
      <td>145.200961</td>
      <td>13.410406</td>
      <td>73.8002</td>
      <td>1.085890</td>
      <td>18.0</td>
      <td>89.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>107.0</td>
      <td>871.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0ad316f6b5cb5e81ebff73ae2490ccfe</td>
      <td>219.997219</td>
      <td>11.489706</td>
      <td>210.690798</td>
      <td>19.373709</td>
      <td>106.4842</td>
      <td>1.020450</td>
      <td>8.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>49.0</td>
      <td>440.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0bbe09e34a11e8e31cf49d6f8df2992d</td>
      <td>187.256657</td>
      <td>4.116640</td>
      <td>167.754106</td>
      <td>15.491333</td>
      <td>88.9639</td>
      <td>1.023460</td>
      <td>8.0</td>
      <td>40.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>51.0</td>
      <td>290.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0dca7ec6ba9b6e8f17f04f713a6be727</td>
      <td>149.058133</td>
      <td>2.273317</td>
      <td>68.569957</td>
      <td>6.341866</td>
      <td>27.2847</td>
      <td>1.419528</td>
      <td>22.0</td>
      <td>115.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>145.0</td>
      <td>787.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0f38be2df6854b4374f06cae1bc38482</td>
      <td>232.526261</td>
      <td>13.661960</td>
      <td>214.635592</td>
      <td>19.817550</td>
      <td>117.0820</td>
      <td>1.026540</td>
      <td>62.0</td>
      <td>526.0</td>
      <td>9.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>32.0</td>
      <td>7.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>600.0</td>
      <td>1510.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



### Análise dos clusters resultantes

Para analisar as características que definem cada um dos clusters utilizaremos a função `describe` do scikit learn. Basicamente analisaremos as características estatísticas das variáveis. 

* **Cluster 1 (cluster_index = 0)**: é o cluster com maior receita média. Contém poucos produtos e que são comprados em sua maioria de forma unitária, ou seja, é o cluster com os produtos mais caros. 
* **Cluster 2 (cluster_index = 1)**: é o cluster identificado por cancelamentos. Possui médias elevadas de cancelamento que advém de três motivos principais: boleto não pago, dados divergentes e solicitações feitas pelo cliente para cancelamento de compra. 
* **Cluster 3 (cluster_index = 2)**: é o cluster dos produtos baratos. São produtos que tem poucos cancelamentos e muitos processamentos finalizados. 
* **Cluster 4 (cluster_index = 3)**: cluster mais balanceado em termos de caracteírsticas. Não tem nenhuma que se sobressaia em relção aos demais clusters, mas é importante ressaltar que há um número considerável de cancelamentos por boletos não pagos para esse tipo de produto. 


```python
product_dataset[product_dataset.clusters == 0].describe(percentiles=[])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue</th>
      <th>tax_substitution</th>
      <th>price</th>
      <th>pis_cofins</th>
      <th>liquid_cost</th>
      <th>quantity</th>
      <th>order_status_cancelado</th>
      <th>order_status_cancelado boleto não pago</th>
      <th>order_status_cancelado dados divergentes</th>
      <th>order_status_cancelado fraude confirmada</th>
      <th>...</th>
      <th>order_status_fraude confirmada</th>
      <th>order_status_pendente processamento</th>
      <th>order_status_processado</th>
      <th>order_status_solicitação de cancelamento</th>
      <th>order_status_solicitação de troca</th>
      <th>order_status_suspeita de fraude</th>
      <th>order_status_suspenso barragem</th>
      <th>process_status_captado</th>
      <th>process_status_processado</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>...</td>
      <td>11.000000</td>
      <td>11.0</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.000000</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1153.709375</td>
      <td>54.439964</td>
      <td>904.337302</td>
      <td>76.392197</td>
      <td>531.526464</td>
      <td>1.028387</td>
      <td>22.909091</td>
      <td>53.636364</td>
      <td>7.181818</td>
      <td>1.181818</td>
      <td>...</td>
      <td>2.636364</td>
      <td>0.0</td>
      <td>2.545455</td>
      <td>3.454545</td>
      <td>4.000000</td>
      <td>0.363636</td>
      <td>0.181818</td>
      <td>88.090909</td>
      <td>341.272727</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>411.061919</td>
      <td>42.177955</td>
      <td>300.073625</td>
      <td>37.588665</td>
      <td>167.658993</td>
      <td>0.040827</td>
      <td>24.764711</td>
      <td>56.593768</td>
      <td>12.040085</td>
      <td>3.600505</td>
      <td>...</td>
      <td>6.531045</td>
      <td>0.0</td>
      <td>3.908034</td>
      <td>3.445682</td>
      <td>4.358899</td>
      <td>0.674200</td>
      <td>0.404520</td>
      <td>93.088619</td>
      <td>347.645535</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>802.791669</td>
      <td>0.000000</td>
      <td>506.407299</td>
      <td>0.000000</td>
      <td>241.671500</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>947.955614</td>
      <td>56.324642</td>
      <td>870.067551</td>
      <td>77.003243</td>
      <td>542.706500</td>
      <td>1.005512</td>
      <td>18.000000</td>
      <td>32.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>57.000000</td>
      <td>340.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2018.831606</td>
      <td>138.374464</td>
      <td>1582.376667</td>
      <td>146.369861</td>
      <td>896.681400</td>
      <td>1.099270</td>
      <td>76.000000</td>
      <td>176.000000</td>
      <td>40.000000</td>
      <td>12.000000</td>
      <td>...</td>
      <td>22.000000</td>
      <td>0.0</td>
      <td>12.000000</td>
      <td>11.000000</td>
      <td>14.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>255.000000</td>
      <td>1015.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 26 columns</p>
</div>




```python
product_dataset[product_dataset.clusters == 1].describe(percentiles=[])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue</th>
      <th>tax_substitution</th>
      <th>price</th>
      <th>pis_cofins</th>
      <th>liquid_cost</th>
      <th>quantity</th>
      <th>order_status_cancelado</th>
      <th>order_status_cancelado boleto não pago</th>
      <th>order_status_cancelado dados divergentes</th>
      <th>order_status_cancelado fraude confirmada</th>
      <th>...</th>
      <th>order_status_fraude confirmada</th>
      <th>order_status_pendente processamento</th>
      <th>order_status_processado</th>
      <th>order_status_solicitação de cancelamento</th>
      <th>order_status_solicitação de troca</th>
      <th>order_status_suspeita de fraude</th>
      <th>order_status_suspenso barragem</th>
      <th>process_status_captado</th>
      <th>process_status_processado</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.0</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>345.227587</td>
      <td>20.398495</td>
      <td>247.749997</td>
      <td>16.682238</td>
      <td>146.100133</td>
      <td>1.061505</td>
      <td>483.333333</td>
      <td>1752.000000</td>
      <td>66.666667</td>
      <td>7.000000</td>
      <td>...</td>
      <td>18.000000</td>
      <td>1.333333</td>
      <td>77.666667</td>
      <td>81.333333</td>
      <td>170.666667</td>
      <td>2.0</td>
      <td>4.666667</td>
      <td>2337.000000</td>
      <td>13944.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>49.659422</td>
      <td>10.526943</td>
      <td>82.998012</td>
      <td>15.902662</td>
      <td>58.744129</td>
      <td>0.031440</td>
      <td>105.077749</td>
      <td>443.744296</td>
      <td>21.548395</td>
      <td>1.732051</td>
      <td>...</td>
      <td>3.464102</td>
      <td>2.309401</td>
      <td>51.013070</td>
      <td>39.119475</td>
      <td>51.325757</td>
      <td>2.0</td>
      <td>1.154701</td>
      <td>341.632844</td>
      <td>5538.598288</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>287.940286</td>
      <td>13.954603</td>
      <td>199.496193</td>
      <td>0.000000</td>
      <td>105.355700</td>
      <td>1.037194</td>
      <td>388.000000</td>
      <td>1317.000000</td>
      <td>46.000000</td>
      <td>6.000000</td>
      <td>...</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>26.000000</td>
      <td>37.000000</td>
      <td>123.000000</td>
      <td>0.0</td>
      <td>4.000000</td>
      <td>2001.000000</td>
      <td>7990.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>371.707449</td>
      <td>14.694428</td>
      <td>200.166734</td>
      <td>18.377166</td>
      <td>119.506500</td>
      <td>1.050309</td>
      <td>466.000000</td>
      <td>1735.000000</td>
      <td>65.000000</td>
      <td>6.000000</td>
      <td>...</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>79.000000</td>
      <td>96.000000</td>
      <td>164.000000</td>
      <td>2.0</td>
      <td>4.000000</td>
      <td>2326.000000</td>
      <td>14899.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>376.035028</td>
      <td>32.546455</td>
      <td>343.587064</td>
      <td>31.669548</td>
      <td>213.438200</td>
      <td>1.097010</td>
      <td>596.000000</td>
      <td>2204.000000</td>
      <td>89.000000</td>
      <td>9.000000</td>
      <td>...</td>
      <td>20.000000</td>
      <td>4.000000</td>
      <td>128.000000</td>
      <td>111.000000</td>
      <td>225.000000</td>
      <td>4.0</td>
      <td>6.000000</td>
      <td>2684.000000</td>
      <td>18943.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 26 columns</p>
</div>




```python
product_dataset[product_dataset.clusters == 2].describe(percentiles=[])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue</th>
      <th>tax_substitution</th>
      <th>price</th>
      <th>pis_cofins</th>
      <th>liquid_cost</th>
      <th>quantity</th>
      <th>order_status_cancelado</th>
      <th>order_status_cancelado boleto não pago</th>
      <th>order_status_cancelado dados divergentes</th>
      <th>order_status_cancelado fraude confirmada</th>
      <th>...</th>
      <th>order_status_fraude confirmada</th>
      <th>order_status_pendente processamento</th>
      <th>order_status_processado</th>
      <th>order_status_solicitação de cancelamento</th>
      <th>order_status_solicitação de troca</th>
      <th>order_status_suspeita de fraude</th>
      <th>order_status_suspenso barragem</th>
      <th>process_status_captado</th>
      <th>process_status_processado</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>...</td>
      <td>105.000000</td>
      <td>105.0</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.000000</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>220.781153</td>
      <td>11.917112</td>
      <td>180.310997</td>
      <td>16.638289</td>
      <td>92.321761</td>
      <td>1.060515</td>
      <td>16.914286</td>
      <td>54.285714</td>
      <td>1.685714</td>
      <td>0.238095</td>
      <td>...</td>
      <td>0.380952</td>
      <td>0.0</td>
      <td>3.695238</td>
      <td>3.752381</td>
      <td>9.133333</td>
      <td>0.076190</td>
      <td>0.095238</td>
      <td>73.790476</td>
      <td>548.200000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>143.196487</td>
      <td>9.592647</td>
      <td>107.343996</td>
      <td>9.911841</td>
      <td>61.010171</td>
      <td>0.082532</td>
      <td>16.686080</td>
      <td>65.294998</td>
      <td>2.127269</td>
      <td>0.546383</td>
      <td>...</td>
      <td>0.578143</td>
      <td>0.0</td>
      <td>4.863767</td>
      <td>3.738523</td>
      <td>7.809101</td>
      <td>0.358824</td>
      <td>0.325925</td>
      <td>81.175107</td>
      <td>463.143739</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.677929</td>
      <td>0.000000</td>
      <td>7.677929</td>
      <td>0.710229</td>
      <td>4.114100</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>180.088038</td>
      <td>9.695407</td>
      <td>153.447222</td>
      <td>14.150808</td>
      <td>78.862100</td>
      <td>1.029148</td>
      <td>12.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>410.000000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>819.984556</td>
      <td>43.523486</td>
      <td>505.618969</td>
      <td>46.478825</td>
      <td>289.356300</td>
      <td>1.419528</td>
      <td>102.000000</td>
      <td>526.000000</td>
      <td>13.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>0.0</td>
      <td>32.000000</td>
      <td>23.000000</td>
      <td>37.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>600.000000</td>
      <td>2566.000000</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 26 columns</p>
</div>




```python
product_dataset[product_dataset.clusters == 3].describe(percentiles=[])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue</th>
      <th>tax_substitution</th>
      <th>price</th>
      <th>pis_cofins</th>
      <th>liquid_cost</th>
      <th>quantity</th>
      <th>order_status_cancelado</th>
      <th>order_status_cancelado boleto não pago</th>
      <th>order_status_cancelado dados divergentes</th>
      <th>order_status_cancelado fraude confirmada</th>
      <th>...</th>
      <th>order_status_fraude confirmada</th>
      <th>order_status_pendente processamento</th>
      <th>order_status_processado</th>
      <th>order_status_solicitação de cancelamento</th>
      <th>order_status_solicitação de troca</th>
      <th>order_status_suspeita de fraude</th>
      <th>order_status_suspenso barragem</th>
      <th>process_status_captado</th>
      <th>process_status_processado</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.00000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>...</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.000000</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>318.858262</td>
      <td>14.605908</td>
      <td>222.413488</td>
      <td>20.528214</td>
      <td>135.812267</td>
      <td>1.042512</td>
      <td>149.666667</td>
      <td>573.25000</td>
      <td>18.166667</td>
      <td>1.333333</td>
      <td>...</td>
      <td>3.333333</td>
      <td>0.083333</td>
      <td>19.916667</td>
      <td>28.833333</td>
      <td>73.750000</td>
      <td>1.083333</td>
      <td>0.500000</td>
      <td>748.166667</td>
      <td>4274.666667</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>266.250027</td>
      <td>7.048851</td>
      <td>96.180490</td>
      <td>8.890202</td>
      <td>67.741172</td>
      <td>0.026064</td>
      <td>61.959420</td>
      <td>436.91046</td>
      <td>16.375333</td>
      <td>1.230915</td>
      <td>...</td>
      <td>2.015095</td>
      <td>0.288675</td>
      <td>10.748855</td>
      <td>11.223622</td>
      <td>27.200017</td>
      <td>0.792961</td>
      <td>0.522233</td>
      <td>484.537236</td>
      <td>1533.470178</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>83.059558</td>
      <td>5.796524</td>
      <td>73.954865</td>
      <td>6.823654</td>
      <td>36.530200</td>
      <td>1.020166</td>
      <td>63.000000</td>
      <td>144.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>219.000000</td>
      <td>1875.000000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>240.505700</td>
      <td>13.514785</td>
      <td>205.750719</td>
      <td>18.971197</td>
      <td>116.854650</td>
      <td>1.034884</td>
      <td>138.500000</td>
      <td>437.50000</td>
      <td>14.500000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>28.500000</td>
      <td>75.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>641.000000</td>
      <td>3840.500000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1086.781258</td>
      <td>30.877153</td>
      <td>346.855964</td>
      <td>32.048355</td>
      <td>229.524900</td>
      <td>1.104629</td>
      <td>246.000000</td>
      <td>1751.00000</td>
      <td>62.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>40.000000</td>
      <td>45.000000</td>
      <td>117.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1970.000000</td>
      <td>7864.000000</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 26 columns</p>
</div>



## Implementação e análise do modelo preditivo

O desafio de predizer as vendas dos meses de junho, julho e agosto de 2017 pode ser compreendido como _Timeseries Forecasting_. Para resolver esse tipo de problema, é possível utilizar modelos estatísticos ou modelos de machine learning [10]. Para este projeto utilizaremos o modelo estatístico ARIMA para realizar as predições.  

### Preparção do dataset

Para lidar com _Timeseries Forecasting_ precisamos agrupar os dados por ano-mês e somar os valores da variável `quantity`. Como o objetivo da predição é fazê-la relacionada a cada produto existente no dataset, então todo o modelo será criado e melhorado considerando um produto e o mesmo procedimento poderá ser executado para todos os demais produtos. 

O código abaixo mostra toda a preparação do dataset. 


```python
df2 = df[df.code == 'e6762ba2ffbca07ab6cee7551caeaad5']

arima_dataset = df2.groupby(pd.Grouper(key='capture_date', freq='M'))['quantity'].aggregate(np.sum)
arima_dataset = arima_dataset.fillna(value=0)
arima_dataset
```




    capture_date
    2016-06-30     7.0
    2016-07-31     0.0
    2016-08-31     0.0
    2016-09-30     3.0
    2016-10-31    13.0
    2016-11-30     3.0
    2016-12-31     8.0
    2017-01-31     8.0
    2017-02-28     1.0
    2017-03-31     3.0
    2017-04-30     5.0
    2017-05-31     6.0
    Freq: M, Name: quantity, dtype: float64



### Implementação e validação do modelo

O código abaixo mostra uma implementação do modelo ARIMA e sua parametrização a partir do método de _Grid Search_ [11]. As melhores configurações para o algoritmo são utilizadas para treinar um modelo final e predizer os valores dos meses seguintes. 

Como a base de treino possui dados apenas até janeiro de 2017, faremos 7 previsões com o modelo selecionado a fim de identificar a quantidade de produtos necessários. 


```python
def evaluate_arima_model(X, arima_order):
    size = int(len(X) * 0.70)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
#         print('predicted=%f, expected=%f' % (yhat, obs))
    mse = mean_squared_error(test, predictions)
    rmse = math.sqrt(mse)
    return(rmse)

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s RMSE=%.3f' % (order,mse))
                except NameError as e:
                    print e
                    continue
                except ValueError as e:
                    print e
                    continue
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    return(best_cfg)

# evaluate parameters
p_values = range(0, 2)
d_values = range(0, 2)
q_values = range(0, 2)
arima_cfg = evaluate_models(arima_dataset.values, p_values, d_values, q_values)
```

    ARIMA(0, 0, 0) RMSE=2.411
    ARIMA(0, 0, 1) RMSE=2.483
    ARIMA(0, 1, 0) RMSE=4.062
    ARIMA(0, 1, 1) RMSE=4.336
    ARIMA(1, 0, 0) RMSE=2.462
    ARIMA(1, 0, 1) RMSE=1.864
    ARIMA(1, 1, 0) RMSE=4.315
    ARIMA(1, 1, 1) RMSE=4.806
    Best ARIMA(1, 0, 1) RMSE=1.864



```python
def arima_predictions(X, arima_cfg, n_months=7, start_month=2):
    X = X.astype('float32')
    size = int(len(X) * .7)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    model = ARIMA(history, order=arima_cfg)
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(steps=n_months)[0]
    
    month=start_month
    for yhat in forecast:
        print("Month = {}: {}".format(month, yhat))
        month+=1
    
    
arima_predictions(arima_dataset.values, arima_cfg)

# arima_dataset[0:(int(len(arima_dataset)*.7))]
```

    Month = 2: 3.81794341454
    Month = 3: 4.31803956354
    Month = 4: 4.63326693882
    Month = 5: 4.83196532577
    Month = 6: 4.95721158314
    Month = 7: 5.03615849976
    Month = 8: 5.08592138908


## Conclusões

Neste projeto foram implementados dois modelos principais para estudo dos dados: um KMeans cluster que agrupou os protudos de acordo com suas características e um ARIMA Model que previu as vendas de cada um dos produtos. 

Foi utilizada o _elbow method_ para definir o melhor valor de K para o KMeans e foi feito um estudo sobre as principais características de cada um dos clusters. Em especial existe um cluster que possui um elevado índice de cancelamentos nas compras. Uma recomendação que daria é que para todo valor previsto de estoque para produtos desse cluster, que esse número fosse decrescido de algum percentual que evitasse custo de manter os produtos cancelados no estoque. 

O ARIMA foi validado usando RMSE como métrica de avaliação e seus parâmetros foram otimizados usando _Grid Search_.

Trabalhos futuros na parte do cluster:

* Considerar novas features, como quantidade total vendida, receita total gerada e se o produto esteve presente nas datas especiais citadas (ex: Black Friday)
* Determinar um percentual correto de decréscimo no valor de estoque previsto dos produtos do cluster que possui muitos cancelamentos. 

Trabalhos futuros na parte do modelo preditivo:

* Montar uma estrutura de código para que seja preciso apenas enviar o código do produto e o forecasting é feito automaticamente
* Aumentar o histórico de dados de cada produto a fim de diminuir a margem de erro do algoritmo
* Aplicar modelos de machine learning para realizar a previsão de estoque de tal modo que as características de cada cluster sejam utilizadas pelo modelo. 


## Referências

[1] Huang, Z.: Clustering large data sets with mixed numeric and categorical values, Proceedings of the First Pacific Asia Knowledge Discovery and Data Mining Conference, Singapore, pp. 21-34, 1997.

[2] Steinley, D.: K-means clustering: A half-century synthesis. British Journal of Math-
ematical and Statistical Psychology 59(1), 1{34 (2006)

[3] Wang, Fei & Franco, Hector & Pugh, John & Ross, Robert. (2016). Empirical Comparative Analysis of 1-of-K Coding and K-Prototypes in Categorical Clustering.

[4] https://yurongfan.wordpress.com/2017/02/04/a-summary-of-different-clustering-methods/ (uso da silhueta para K-prototype)

[5] https://thiagorodrigo.com.br/artigo/liquidacao-fantastica-2017-na-magazine-luiza-da-ate-70-de-desconto/

[6] https://github.com/nicodv/kmodes

[7] http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

[8] https://en.wikipedia.org/wiki/Elbow_method_(clustering)

[9] https://www.quora.com/How-can-we-choose-a-good-K-for-K-means-clustering

[10] Bontempi G., Ben Taieb S., Le Borgne YA. (2013) Machine Learning Strategies for Time Series Forecasting. In: Aufaure MA., Zimányi E. (eds) Business Intelligence. eBISS 2012. Lecture Notes in Business Information Processing, vol 138. Springer, Berlin, Heidelberg

[11] https://machinelearningmastery.com/time-series-forecast-study-python-monthly-sales-french-champagne/
