```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv("titanic.csv")
df.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



# Dados Numericos


```python
dfNumerico = df[["PassengerId","Age","Fare"]]
dfNumerico = dfNumerico.dropna()
dfNumerico.head()
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
      <th>PassengerId</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>22.0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>38.0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>26.0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>35.0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>35.0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>



## Importando pre processadores


```python
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, Normalizer, RobustScaler
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
      <th>PassengerId</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001122</td>
      <td>0.2750</td>
      <td>0.014151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.002245</td>
      <td>0.4750</td>
      <td>0.139136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.003367</td>
      <td>0.3250</td>
      <td>0.015469</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.004489</td>
      <td>0.4375</td>
      <td>0.103644</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005612</td>
      <td>0.4375</td>
      <td>0.015713</td>
    </tr>
  </tbody>
</table>
</div>



## Max abs Scaler


```python
max_abs_scaler = MaxAbsScaler()
dfMaxAbsScaler = pd.DataFrame(max_abs_scaler.fit_transform(dfNumerico), columns=dfNumerico.columns)
dfMaxAbsScaler.head()
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
      <th>PassengerId</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001122</td>
      <td>0.2750</td>
      <td>0.014151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.002245</td>
      <td>0.4750</td>
      <td>0.139136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.003367</td>
      <td>0.3250</td>
      <td>0.015469</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.004489</td>
      <td>0.4375</td>
      <td>0.103644</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005612</td>
      <td>0.4375</td>
      <td>0.015713</td>
    </tr>
  </tbody>
</table>
</div>



## MinMaxScaler


```python
min_max_scaler = MinMaxScaler()
df_min_max_scaled = pd.DataFrame(min_max_scaler.fit_transform(dfNumerico), columns=dfNumerico.columns)
df_min_max_scaled.head()
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
      <th>PassengerId</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.271174</td>
      <td>0.014151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001124</td>
      <td>0.472229</td>
      <td>0.139136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.002247</td>
      <td>0.321438</td>
      <td>0.015469</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.003371</td>
      <td>0.434531</td>
      <td>0.103644</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.004494</td>
      <td>0.434531</td>
      <td>0.015713</td>
    </tr>
  </tbody>
</table>
</div>



## standard_scaler


```python
standard_scaler = StandardScaler()
df_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(dfNumerico), columns=dfNumerico.columns)
df_standard_scaled.head()
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
      <th>PassengerId</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.728532</td>
      <td>-0.530377</td>
      <td>-0.518978</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.724670</td>
      <td>0.571831</td>
      <td>0.691897</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.720808</td>
      <td>-0.254825</td>
      <td>-0.506214</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.716946</td>
      <td>0.365167</td>
      <td>0.348049</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.713084</td>
      <td>0.365167</td>
      <td>-0.503850</td>
    </tr>
  </tbody>
</table>
</div>



## Normalizer


```python
normalizer = Normalizer()
df_normalized = pd.DataFrame(normalizer.fit_transform(dfNumerico), columns=dfNumerico.columns)
df_normalized.head()
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
      <th>PassengerId</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.043131</td>
      <td>0.948873</td>
      <td>0.312697</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.024751</td>
      <td>0.470273</td>
      <td>0.882174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.109705</td>
      <td>0.950778</td>
      <td>0.289804</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.062772</td>
      <td>0.549253</td>
      <td>0.833295</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.137892</td>
      <td>0.965245</td>
      <td>0.222006</td>
    </tr>
  </tbody>
</table>
</div>



## RobustScaler


```python
robust_scaler = RobustScaler()
df_robust_scaled = pd.DataFrame(robust_scaler.fit_transform(dfNumerico), columns=dfNumerico.columns)
df_robust_scaled.head()
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
      <th>PassengerId</th>
      <th>Age</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.974753</td>
      <td>-0.335664</td>
      <td>-0.335309</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.972558</td>
      <td>0.559441</td>
      <td>2.193153</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.970362</td>
      <td>-0.111888</td>
      <td>-0.308655</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.968167</td>
      <td>0.391608</td>
      <td>1.475155</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.965971</td>
      <td>0.391608</td>
      <td>-0.303720</td>
    </tr>
  </tbody>
</table>
</div>



# Dados Categoricos


```python
dfCategorico = df[["Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked","Cabin","Ticket"]]
dfCategorico.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Embarked</th>
      <th>Cabin</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>S</td>
      <td>NaN</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>C</td>
      <td>C85</td>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>S</td>
      <td>NaN</td>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>S</td>
      <td>C123</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>S</td>
      <td>NaN</td>
      <td>373450</td>
    </tr>
  </tbody>
</table>
</div>



## importando pre processadores


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
columnsToEncode = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
```

## OneHotEncoder


```python

encoder = OneHotEncoder()
encodedData = encoder.fit_transform(df[columnsToEncode])
encodedDf = pd.DataFrame(encodedData.toarray(), columns=encoder.get_feature_names_out(columnsToEncode))
encodedDf.head()
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
      <th>Survived_0</th>
      <th>Survived_1</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>SibSp_0</th>
      <th>SibSp_1</th>
      <th>SibSp_2</th>
      <th>...</th>
      <th>Parch_1</th>
      <th>Parch_2</th>
      <th>Parch_3</th>
      <th>Parch_4</th>
      <th>Parch_5</th>
      <th>Parch_6</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Embarked_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



## labelEncoder 


```python
labelEncoder = LabelEncoder()
```


```python
df[columnsToEncode] = df[columnsToEncode].apply(lambda col: labelEncoder.fit_transform(col))
df.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>Allen, Mr. William Henry</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
