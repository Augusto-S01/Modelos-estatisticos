# Importação das bibliotecas


```python
#Importações de bibliotecas

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


```

# Configuração das bibliotecas


```python
pd.reset_option('display.max_colwidth')
```

# importação dos CSV e transformando em datasets pandas



```python
gruposDFnomeDasColunas = [
    "nome",
    "classificacao",
    "frequencia_feminina",
    "frequencia_masculina",
    "frequencial_total",
    "proporcao",
    "nomes_alternativos"    
]
gruposDF = pd.read_csv('grupos.csv', names=gruposDFnomeDasColunas, header=0)
gruposDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nome</th>
      <th>classificacao</th>
      <th>frequencia_feminina</th>
      <th>frequencia_masculina</th>
      <th>frequencial_total</th>
      <th>proporcao</th>
      <th>nomes_alternativos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALINE</td>
      <td>F</td>
      <td>528515</td>
      <td>2035</td>
      <td>530550</td>
      <td>0.996164</td>
      <td>|AALINE|AILINE|ALEINE|ALIINE|ALINE|ALINER|ALIN...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ARAO</td>
      <td>M</td>
      <td>0</td>
      <td>3526</td>
      <td>3526</td>
      <td>1.000000</td>
      <td>|AARAO|ARAAO|ARAO|</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARON</td>
      <td>M</td>
      <td>0</td>
      <td>3442</td>
      <td>3442</td>
      <td>1.000000</td>
      <td>|AARON|AHARON|AROM|ARON|ARYON|HARON|</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADA</td>
      <td>F</td>
      <td>5294</td>
      <td>289</td>
      <td>5583</td>
      <td>0.948236</td>
      <td>|ABA|ADA|ADAH|ADAR|ADHA|HADA|</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABADE</td>
      <td>M</td>
      <td>0</td>
      <td>57</td>
      <td>57</td>
      <td>1.000000</td>
      <td>|ABADE|</td>
    </tr>
  </tbody>
</table>
</div>




```python
nomesDSnomeDasColunas = [
    "nomes_alternativos",
    "classificacao",
    "primeiro_nome",
    "frequencia_feminina",
    "frequencia_masculina",
    "frequencia_total",
    "frequencia_grupo",
    "nome_grupo",
    "proporcao"
]
nomesDF= pd.read_csv("nomes.csv",names=nomesDSnomeDasColunas,header=0)
nomesDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nomes_alternativos</th>
      <th>classificacao</th>
      <th>primeiro_nome</th>
      <th>frequencia_feminina</th>
      <th>frequencia_masculina</th>
      <th>frequencia_total</th>
      <th>frequencia_grupo</th>
      <th>nome_grupo</th>
      <th>proporcao</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AILINE|ALEINE|ALIINE|ALINE|ALINER|ALINHE|ALINN...</td>
      <td>F</td>
      <td>AALINE</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>66</td>
      <td>530550</td>
      <td>ALINE</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ARAAO|ARAO</td>
      <td>M</td>
      <td>AARAO</td>
      <td>NaN</td>
      <td>281.0</td>
      <td>281</td>
      <td>3526</td>
      <td>ARAO</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AHARON|AROM|ARON|ARYON|HARON</td>
      <td>M</td>
      <td>AARON</td>
      <td>NaN</td>
      <td>676.0</td>
      <td>676</td>
      <td>3442</td>
      <td>ARON</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADA|ADAH|ADAR|ADHA|HADA</td>
      <td>F</td>
      <td>ABA</td>
      <td>82.0</td>
      <td>NaN</td>
      <td>82</td>
      <td>5583</td>
      <td>ADA</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>M</td>
      <td>ABADE</td>
      <td>NaN</td>
      <td>57.0</td>
      <td>57</td>
      <td>57</td>
      <td>ABADE</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



# Limpeza de nulos e n/a


```python
gruposDF.fillna(0, inplace=True)
nomesDF.fillna(0, inplace=True)

gruposDF.drop_duplicates(inplace=True)
nomesDF.drop_duplicates(inplace=True)
```

# Unir dataframes


```python
gruposSelecionado= gruposDF[['nome', 'frequencia_feminina', 'frequencia_masculina', 'classificacao']]
nomesSelecionado = nomesDF[['primeiro_nome', 'frequencia_feminina', 'frequencia_masculina', 'classificacao']]
```


```python
nomesSelecionado = nomesSelecionado.rename(columns={'primeiro_nome': 'nome'})
```


```python
nomesSelecionado.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nome</th>
      <th>frequencia_feminina</th>
      <th>frequencia_masculina</th>
      <th>classificacao</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AALINE</td>
      <td>66.0</td>
      <td>0.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AARAO</td>
      <td>0.0</td>
      <td>281.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AARON</td>
      <td>0.0</td>
      <td>676.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABA</td>
      <td>82.0</td>
      <td>0.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABADE</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
gruposSelecionado.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nome</th>
      <th>frequencia_feminina</th>
      <th>frequencia_masculina</th>
      <th>classificacao</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALINE</td>
      <td>528515</td>
      <td>2035</td>
      <td>F</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ARAO</td>
      <td>0</td>
      <td>3526</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARON</td>
      <td>0</td>
      <td>3442</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADA</td>
      <td>5294</td>
      <td>289</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABADE</td>
      <td>0</td>
      <td>57</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
data =  pd.merge(gruposSelecionado, nomesSelecionado, on='nome', how='outer')
```


```python
data.drop_duplicates(subset='nome', keep='first', inplace=True)
```

# Formatação e encoding


```python
data['frequenciaFeminina'] = data['frequencia_feminina_x'] + data['frequencia_feminina_y']
data['frequenciaMasculina'] = data['frequencia_masculina_x'] + data['frequencia_masculina_y']
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nome</th>
      <th>frequencia_feminina_x</th>
      <th>frequencia_masculina_x</th>
      <th>classificacao_x</th>
      <th>frequencia_feminina_y</th>
      <th>frequencia_masculina_y</th>
      <th>classificacao_y</th>
      <th>frequenciaFeminina</th>
      <th>frequenciaMasculina</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALINE</td>
      <td>528515.0</td>
      <td>2035.0</td>
      <td>F</td>
      <td>509869.0</td>
      <td>1868.0</td>
      <td>F</td>
      <td>1038384.0</td>
      <td>3903.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ARAO</td>
      <td>0.0</td>
      <td>3526.0</td>
      <td>M</td>
      <td>0.0</td>
      <td>3078.0</td>
      <td>M</td>
      <td>0.0</td>
      <td>6604.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARON</td>
      <td>0.0</td>
      <td>3442.0</td>
      <td>M</td>
      <td>0.0</td>
      <td>2269.0</td>
      <td>M</td>
      <td>0.0</td>
      <td>5711.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADA</td>
      <td>5294.0</td>
      <td>289.0</td>
      <td>F</td>
      <td>5029.0</td>
      <td>266.0</td>
      <td>F</td>
      <td>10323.0</td>
      <td>555.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABADE</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>M</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>M</td>
      <td>0.0</td>
      <td>114.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Manter apenas as colunas de interesse
data = data[['nome', 'frequenciaFeminina', 'frequenciaMasculina', 'classificacao_x']]
data.rename(columns={'classificacao_x': 'classificacao'}, inplace=True)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nome</th>
      <th>frequenciaFeminina</th>
      <th>frequenciaMasculina</th>
      <th>classificacao</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALINE</td>
      <td>1038384.0</td>
      <td>3903.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ARAO</td>
      <td>0.0</td>
      <td>6604.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARON</td>
      <td>0.0</td>
      <td>5711.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADA</td>
      <td>10323.0</td>
      <td>555.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABADE</td>
      <td>0.0</td>
      <td>114.0</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
labelEncoder = LabelEncoder()
data['classificacaoCodificada'] = labelEncoder.fit_transform(data['classificacao'])
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>nome</th>
      <th>frequenciaFeminina</th>
      <th>frequenciaMasculina</th>
      <th>classificacao</th>
      <th>classificacaoCodificada</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALINE</td>
      <td>1038384.0</td>
      <td>3903.0</td>
      <td>F</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ARAO</td>
      <td>0.0</td>
      <td>6604.0</td>
      <td>M</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ARON</td>
      <td>0.0</td>
      <td>5711.0</td>
      <td>M</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADA</td>
      <td>10323.0</td>
      <td>555.0</td>
      <td>F</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABADE</td>
      <td>0.0</td>
      <td>114.0</td>
      <td>M</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#F = 0
#M = 1
data.drop(columns=["classificacao"], inplace=True)
```

# Salvar CSV


```python
data.to_csv('data.csv', index=False)
```
