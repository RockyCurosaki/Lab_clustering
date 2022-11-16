# Lab_clustering
## Кластеризация методом К-средних
Кластеризация — это процесс разделения различных частей данных на основе общих характеристик. Разрозненные отрасли, включая розничную торговлю, финансы и здравоохранение, используют методы кластеризации для различных аналитических задач. В розничной торговле кластеризация может помочь идентифицировать отдельные группы потребителей, что затем может позволить компании создавать таргетированную рекламу на основе демографических данных потребителей, которые могут быть слишком сложными для проверки вручную. В сфере финансов кластеризация может обнаруживать различные формы незаконной рыночной деятельности, такие как подделка книги заказов, при которой трейдеры обманным путем размещают крупные ордера, чтобы заставить других трейдеров купить или продать актив. В здравоохранении методы кластеризации использовались для определения структуры затрат пациентов, ранних неврологических расстройств и экспрессии раковых генов.
Python предлагает множество полезных инструментов для выполнения кластерного анализа. Выбор лучшего инструмента зависит от решаемой проблемы и типа имеющихся данных. Для относительно низкоразмерных задач (не более нескольких десятков входных данных), таких как определение отдельных групп потребителей, кластеризация K-средних является отличным выбором. 
Кластеризация K-средних – один из наиболее широко используемых алгоритмов неконтролируемого машинного обучения, который формирует кластеры данных на основе сходства между экземплярами данных. Для работы этого конкретного алгоритма необходимо заранее определить количество кластеров. K в K-средних означает количество кластеров. Алгоритм K-средних начинается со случайного выбора значения центроида для каждого кластера. После этого алгоритм итеративно выполняет три шага: 
- Уровень списка 1. Пункт 1.
Найти евклидово расстояние между каждым экземпляром данных и центроидами всех кластеров
- Уровень списка 1. Пункт 2.
Назначить экземпляры данных кластеру центроида с ближайшим расстоянием.
- Уровень списка 1. Пункт 3.
Вычислить новые значения центроидов на основе средних значений координат всех экземпляров данных из соответствующего кластера.
##Пример метода k-means.
Ниже приведён пример реализации модели Кластеризации методом К-средних на языке Python b библиотеки Scikit-Learn.
Для начала нам потребуются библиотеки matplotlib, pandas, numpy и scikit-learn.
```Python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
```
Далее мы осуществляем загрузку датасета.
```Python
dataset = pd.read_csv('D:\\Dataset.csv', encoding = 'ISO-8859-1') 
```
Далее мы извлекаем значения из столбцов и превращаем их в переменные
```Python
f1 = dataset['x'].values
f2 = dataset['y'].values
```
Затем создаём из данных массив
```Python
X = np.array(list(zip(f1, f2)))
```
Далее непосредственно осуществляется визуализация данных
```Python
plt.scatter(f1,f2)
plt.show()
```
Приведенный выше код просто отображает все значения в первом столбце массива X относительно всех значений во втором столбце. График будет выглядеть так:  
![Рисунок 1](https://github.com/RockyCurosaki/Lab_clustering/raw/main/1.png)  
Далее осуществляется создание кластеров и подгонка входных данных
```Python
kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
```
Затем осуществляются получение меток кластера и значений центроидов
```Python
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
```
Потом осуществляется вывод значений центроидов
```Python
print(centroids)
```
Следующим шагом снова визуализируем то, как данные были сгруппированы. На этот раз мы построим график данных вместе с присвоенными им метками и координатами центроида каждого кластера, чтобы увидеть, как положение центроида влияет на кластеризацию. 
```Python
plt.scatter(f1, f2, c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.show()
```  
![Рисунок 2](https://github.com/RockyCurosaki/Lab_clustering/raw/main/2.png)  
Кластеризация K-средних – это простой, но очень эффективный алгоритм неконтролируемого машинного обучения для кластеризации данных. Он группирует данные на основе евклидова расстояния между точками данных. Алгоритм кластеризации K-средних имеет множество применений для группировки текстовых документов, изображений, видео и многого другого.




