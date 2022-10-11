# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Proyecto Final - UCOM - Big Data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Para "Run all" se requiere la instalacion de estas librerias:

# COMMAND ----------

#!pip install geopandas
#!pip install folium
#!pip install nltk
#!pip install prophet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importamos librerias

# COMMAND ----------

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
import calendar

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Creamos la sesion de Spark

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col,array_contains

spark = SparkSession.builder.appName('UCOM_Final').getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Conectamos al Blob Storage y montamos

# COMMAND ----------

#storage_account_name = "blobucom"
#storage_account_key = "TByt2tyb7y2H9mnE/doJ90v69ic2gu/Dv3uXeCaT11vPkopo8/cRXFd8yplKl35wex8pH1Tfk9/b+AStQU5X5A=="
#container = "ucom-datos"

#dbutils.fs.mount(
# source = "wasbs://{0}@{1}.blob.core.windows.net".format(container, storage_account_name),
# mount_point = "/mnt/ucom-datos",
# extra_configs = {"fs.azure.account.key.{0}.blob.core.windows.net".format(storage_account_name): storage_account_key}
#)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Listado de archivos en el container

# COMMAND ----------

dbutils.fs.ls("dbfs:/mnt/ucom-datos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cargamos todos los archivos como Dataframes de Spark

# COMMAND ----------

spark_customers = spark.read.option("header",True).csv("dbfs:/mnt/ucom-datos/olist_customers_dataset.csv")
spark_geolocation = spark.read.option("header",True).csv("dbfs:/mnt/ucom-datos/olist_geolocation_dataset.csv")
spark_order_items = spark.read.option("header",True).csv("dbfs:/mnt/ucom-datos/olist_order_items_dataset.csv")
spark_payments = spark.read.option("header",True).csv("dbfs:/mnt/ucom-datos/olist_order_payments_dataset.csv")
spark_order_reviews = spark.read.option("header",True).csv("dbfs:/mnt/ucom-datos/olist_order_reviews_dataset.csv")
spark_orders = spark.read.option("header",True).csv("dbfs:/mnt/ucom-datos/olist_orders_dataset.csv")
spark_products = spark.read.option("header",True).csv("dbfs:/mnt/ucom-datos/olist_products_dataset.csv")
spark_sellers = spark.read.option("header",True).csv("dbfs:/mnt/ucom-datos/olist_sellers_dataset.csv")
spark_product_translation = spark.read.option("header",True).csv("dbfs:/mnt/ucom-datos/product_category_name_translation.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Paso a Dataframe de Pandas (porque me es mas facil)

# COMMAND ----------

customers = spark_customers.toPandas()
geolocation = spark_geolocation.toPandas()
order_items = spark_order_items.toPandas()
payments = spark_payments.toPandas()
order_reviews = spark_order_reviews.toPandas()
orders = spark_orders.toPandas()
products = spark_products.toPandas()
sellers = spark_sellers.toPandas()
product_translation = spark_product_translation.toPandas()

# COMMAND ----------

order_items.head()

# COMMAND ----------

geolocation.head(5)

# COMMAND ----------

orders.head(5)

# COMMAND ----------

customers.head(5)

# COMMAND ----------

temp1 = orders.merge(customers, how='left', on='customer_id')
temp1 = temp1[["order_id","customer_id","customer_unique_id","customer_zip_code_prefix"]]

geolocation.rename(columns={'geolocation_zip_code_prefix': 'customer_zip_code_prefix'}, inplace=True)

ubi = temp1.merge(geolocation, how='left', on='customer_zip_code_prefix')

ubi_price = ubi.merge(order_items, how='left', on='order_id') 

ubi_price = ubi_price[["order_id","customer_id","customer_unique_id","customer_zip_code_prefix", "geolocation_lat", "geolocation_lng", "price", "freight_value"]]

ubi_price.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificamos buscado Nulls en los DFs

# COMMAND ----------

lista = [ customers, geolocation, order_items, payments, order_reviews, orders, products, sellers, product_translation]
print ("#"*50);
for dfs in lista:
    print (dfs.isnull().sum());
    print ("#"*50);

# COMMAND ----------

# MAGIC %md
# MAGIC #### Hay algunos valores nulls en "order_reviews", "orders" y en "products"; entonces miramos los tipos/formatos de los datos

# COMMAND ----------

lista = [customers, geolocation, order_items, payments, order_reviews, orders, products, sellers, product_translation]
print ("#" * 50)
for dfs in lista:
    print (dfs.info())
    print ("#" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pasamos a lo de "orders" tipo date/hora para que pase

# COMMAND ----------

time = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for i in time:
    orders[i] = pd.to_datetime(orders[i])

# COMMAND ----------

orders.info()

# COMMAND ----------

order_reviews.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### En "order_review" es de esperarse los null, pero completamos igual con algo por higiene

# COMMAND ----------

order_reviews['review_comment_title'] = order_reviews['review_comment_title'].fillna('None')
order_reviews['review_comment_message'] = order_reviews['review_comment_message'].fillna('No Comment')

# COMMAND ----------

order_reviews.info()

# COMMAND ----------

orders.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Los datos faltantes son las fechas, en los comentarios y documentacion del dataset esta la hipotesis de que es una orden conjunta. Entonces completamos con el ultimo valor no null
# MAGIC #### podemos usar "ffill" para eso, god bless stackoverflow

# COMMAND ----------

orders = orders.fillna(method = 'ffill')

# COMMAND ----------

orders.info()

# COMMAND ----------

products.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### En productos, el tema de las columas es asi:
# MAGIC 
# MAGIC  - Categoria: Falta el dato en el dataset, entonces completamos nomas con "Otros"
# MAGIC  - product_name_lenght: Puede que el proveedor no pase el dato o que no califique el tipo de producto, entonces se decide rellenar con "Mediana" (segun buenas practicas)
# MAGIC  - product_description_lenght: Puede que el proveedor no pase el dato o que no califique el tipo de producto, entonces se decide rellenar con "Mediana" (segun buenas practicas)
# MAGIC  - product_photos_qty: Puede que el proveedor no pase el dato o que no califique el tipo de producto, entonces se decide rellenar con "Mediana" (segun buenas practicas)
# MAGIC  - product_weight_g: Puede que el proveedor no pase el dato o que no califique el tipo de producto, entonces se decide rellenar con "Mediana" (segun buenas practicas)
# MAGIC  - product_length_cm: Puede que el proveedor no pase el dato o que no califique el tipo de producto, entonces se decide rellenar con "Mediana" (segun buenas practicas)
# MAGIC  - product_height_cm: Puede que el proveedor no pase el dato o que no califique el tipo de producto, entonces se decide rellenar con "Mediana" (segun buenas practicas)
# MAGIC  - product_width_cm: Puede que el proveedor no pase el dato o que no califique el tipo de producto, entonces se decide rellenar con "Mediana" (segun buenas practicas)
# MAGIC  - product_weight_g: Puede que el proveedor no pase el dato o que no califique el tipo de producto, entonces se decide rellenar con "Mediana" (segun buenas practicas)

# COMMAND ----------

products['product_category_name'] = products['product_category_name'].fillna('otros')
products = products.fillna(products.median())

# COMMAND ----------

products.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creamos un ALL_DATA
# MAGIC #### Usamos merge y usamos el valor comun, tipo un vlookup/buscarv

# COMMAND ----------

geolocation.rename({"geolocation_zip_code_prefix":"customer_zip_code_prefix"}, axis = 1, inplace = True)

data1 = pd.merge(orders, order_items, left_on = 'order_id', right_on = 'order_id')
data2 = pd.merge(data1, sellers, left_on = 'seller_id', right_on = 'seller_id')
data3 = pd.merge(data2, products, left_on = 'product_id', right_on = 'product_id')
data4 = pd.merge(data3, order_reviews, left_on = 'order_id', right_on = 'order_id')
data5 = pd.merge(data4, product_translation, left_on = 'product_category_name', right_on = 'product_category_name')
#data6 = pd.merge(data5, customers, left_on = 'customer_id', right_on = 'customer_id')
#data7 = pd.merge(data6, geolocation, left_on = 'customer_zip_code_prefix', right_on = 'customer_zip_code_prefix')
all_data = pd.merge(data5, payments, how = 'left', left_on = 'order_id', right_on = 'order_id')

# COMMAND ----------

all_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analisis de los datos en ALL_DATA

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1- Mejores compradores
# MAGIC #### Contamos la cantidad de ordenes de cada id

# COMMAND ----------

count_ordered =all_data.groupby('customer_id')["order_id"].count().rename("Total Ordered").reset_index()
count_ordered.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Ordenamos y mostramos el Top 10

# COMMAND ----------

top_count_ordered =all_data.groupby('customer_id')["order_id"].count().rename("Total Ordered").reset_index().nlargest(10,'Total Ordered')
top_count_ordered.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2- Estado de las ventas

# COMMAND ----------

#review_categoria_count =all_data.groupby('order_status')["order_id"].count().rename("Cantidad").reset_index().nlargest(10,'Cantidad')

fig, ax = plt.subplots(figsize=(14, 6))
sns.countplot(data=all_data,x='order_status')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3- Satisfaccion del cliente (por cliente y por categoria)
# MAGIC #### review entro como str, entonces pasamos a float primero

# COMMAND ----------

all_data['review_score'] = all_data['review_score'].astype(float)
all_data.review_score.head()

# COMMAND ----------

def Clasificador (x):
    if x >= 4: return "Excelente!"
    if x == 3: return "Buena"
    else: return "Mala"
all_data['review_classification'] = all_data.review_score.apply(lambda row:Clasificador (row))
all_data.head(3)

# COMMAND ----------

review_class =all_data.groupby('review_classification')["order_id"].count().rename("Cantidad").reset_index().nlargest(3,'Cantidad')
review_class.index=['Excelente!', 'Bueno', 'Malo']

# COMMAND ----------

review_class.plot.pie(y='Cantidad', figsize=(5, 5))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculamos el score promedio que dejo cada cliente (promedio de todas sus ordenes)

# COMMAND ----------

promedio_review =all_data.groupby('customer_id')["review_score"].mean().rename("Score Promedio Reviews").reset_index()
promedio_review.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculamos el score promedio que tiene cada categoria de producto (promedio de todas las ordenes de la categoria)

# COMMAND ----------

score_product =all_data.groupby('product_category_name_english')["review_score"].mean().sort_values(ascending=False).rename("Score Promedio Reviews").reset_index()
score_product.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Podemos ver el top de categorias mejor calificadas por clientes

# COMMAND ----------

top_review = score_product[:10]
top_review

# COMMAND ----------

# MAGIC %md
# MAGIC #### Podemos plotear para ver mejor, cada categoria vs su cantidad de estrellas

# COMMAND ----------

plt.figure(figsize = (13,15))
plt.subplot(211)
base_color = sns.color_palette()[6]
sns.barplot(data = top_review, x = 'Score Promedio Reviews' , y = 'product_category_name_english', color = base_color)
plt.title('Top 10 - Categorias mejores calificadas')
plt.xlabel('Cant. Estrellas (promedio)')
plt.ylabel('Categoria');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3- Estado desde donde se realizo la compra
# MAGIC ### Cantidad de clientes en cada estado

# COMMAND ----------

state_customer = customers.groupby('customer_state')['customer_id'].count().sort_values(ascending = False).rename_axis('Estado'). reset_index(name='Cantidad de Clientes')
state_customer

# COMMAND ----------

# MAGIC %md
# MAGIC #### Poodemos plotear cada estado y su cantidad de clientes

# COMMAND ----------

plt.figure(figsize = (25,5))
plt.subplot(121)
base_color = sns.color_palette()[2]
sns.barplot(data = state_customer.sort_values('Cantidad de Clientes', ascending = False), x = 'Estado', y = 'Cantidad de Clientes', color = base_color)
plt.title('Cantidad de clientes en cada estado')
plt.xlabel('Estado')
plt.ylabel('Cantidad de clientes')

# COMMAND ----------

state_seller = sellers.groupby('seller_state')['seller_id'].count().sort_values(ascending = False).rename_axis('Estado').reset_index(name='Cantidad de Vendedores')
state_seller

# COMMAND ----------

plt.figure(figsize = (25,5))
plt.subplot(121)
base_color = sns.color_palette()[1]
sns.barplot(data = state_seller.sort_values('Cantidad de Vendedores', ascending = False), x = 'Estado', y = 'Cantidad de Vendedores', color = base_color)
plt.title('Cantidad de Vendedores en cada Estado')
plt.xlabel('Estado')
plt.ylabel('Cantidad de Vendedores')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### TOP de vendedores segun ventas

# COMMAND ----------

top10_seller = all_data.groupby('seller_id').agg(cantidad_vendida = ('order_item_id', 'count'))
sorted_data = top10_seller.sort_values(by='cantidad_vendida', ascending=False).head(10)

# COMMAND ----------

fig=plt.figure(figsize=(10,5))
sns.barplot(y=sorted_data.index, x=sorted_data['cantidad_vendida'])
plt.title('Top 10 Vendedores',fontsize=16);
plt.xlabel('Cantidad de Ventas',fontsize=12);
plt.ylabel('Seller_id',fontsize=12);

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4 - Cantidad de Productos vendidos por Categoria

# COMMAND ----------

product_counts = all_data.product_category_name_english.value_counts().sort_values(ascending = False).rename_axis('Categoria').reset_index(name='Cantidad de ventas')
product_counts.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Este es el top 10 de productos mas vendidos

# COMMAND ----------

product_count_top = all_data.product_category_name_english.value_counts()[:10].sort_values(ascending = False).rename_axis('Categoria').reset_index(name='Cantidad vendida')
product_count_top

# COMMAND ----------

plt.figure(figsize = (15,15))
plt.subplot(211)
base_color = sns.color_palette()[4]
sns.barplot(data = product_count_top, x = 'Cantidad vendida' , y = 'Categoria', color = base_color)
plt.title('Top 10 - Categorias mas ventidos')
plt.xlabel('Cantidad de ventas')
plt.ylabel('Categoria');

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5- Medio de pago mas utilizado

# COMMAND ----------

payment = payments['payment_type'].value_counts().rename_axis("Medio de pago").reset_index(name ="Cantidad")
payment

# COMMAND ----------

plt.figure(figsize = (15,5))
plt.subplot(121)
base_color = sns.color_palette()[3]
sns.barplot(data = payment, x = 'Medio de pago' , y = 'Cantidad', color = base_color)
plt.title('Medio de pagos vs Uso')
plt.xlabel('Medios de Pago')
plt.ylabel('Cantidad de ordenes');

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6 - Evolutivo de Ventas

# COMMAND ----------

all_data['Year'] = all_data['order_purchase_timestamp'].dt.year
all_data['Month'] = all_data['order_purchase_timestamp'].dt.month_name()

#evolutivo_count = [all_data['Year'], all_data['Month']]
#evolutivo_valor = [all_data['Year'], all_data['Month'], all_data['price']]


# COMMAND ----------

plt.figure(figsize=(15,6))
data_usar = all_data[(all_data['Year'] > 2016)]
sns.countplot(data=data_usar,x='Month',hue='Year', order=list(calendar.month_name));
plt.title('Ventas mensuales', fontsize=20);

# COMMAND ----------

all_data['Day'] = all_data['order_purchase_timestamp'].dt.day_name()
plt.figure(figsize=(10,5))
basecolor=sns.color_palette()[0]
sns.countplot(data=all_data,x='Day',order=list(calendar.day_name),color=basecolor);
plt.title('Vendas por dia de la Semana', fontsize=20);

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 7 - Hora y dia de Entrega

# COMMAND ----------

# Changing the data type for date columns
timestamp_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                  'order_estimated_delivery_date']
for col in timestamp_cols:
    orders[col] = pd.to_datetime(orders[col])
    
# Extracting attributes for purchase date - Year and Month
orders['order_purchase_year'] = orders['order_purchase_timestamp'].apply(lambda x: x.year)
orders['order_purchase_month'] = orders['order_purchase_timestamp'].apply(lambda x: x.month)
orders['order_purchase_month_name'] = orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%b'))
orders['order_purchase_year_month'] = orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m'))
orders['order_purchase_date'] = orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m%d'))

# Extracting attributes for purchase date - Day and Day of Week
orders['order_purchase_day'] = orders['order_purchase_timestamp'].apply(lambda x: x.day)
orders['order_purchase_dayofweek'] = orders['order_purchase_timestamp'].apply(lambda x: x.dayofweek)
orders['order_purchase_dayofweek_name'] = orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%a'))

# Extracting attributes for purchase date - Hour and Time of the Day
orders['order_purchase_hour'] = orders['order_purchase_timestamp'].apply(lambda x: x.hour)
hours_bins = [-0.1, 6, 12, 18, 23]
hours_labels = ['Madrugrada', 'Mañana', 'Tarde', 'Noche']
orders['order_purchase_time_day'] = pd.cut(orders['order_purchase_hour'], hours_bins, labels=hours_labels)

# New DataFrame after transformations
orders.head()

# COMMAND ----------

#tablas




%matplotlib inline
from matplotlib.gridspec import GridSpec

fig = plt.figure(constrained_layout=True, figsize=(13, 10))

# Axis definition
gs = GridSpec(2, 2, figure=fig)
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

sns.countplot(data=orders,x='order_purchase_dayofweek', ax=ax2, palette='YlGnBu')
weekday_label = ['Lun', 'Mar', 'Mier', 'Jue', 'Vier', 'Sab', 'Dom']
ax2.set_xticklabels(weekday_label)
ax2.set_title('Cantidad de Compras segun dia de la semana', size=14, color='dimgrey', pad=20)


day_color_list = ['darkslateblue', 'deepskyblue', 'darkorange', 'purple']
sns.countplot(data=orders, x='order_purchase_time_day', ax=ax3, palette=day_color_list)
ax3.set_title('Cantidad de Compras segun el horario', size=14, color='dimgrey', pad=20)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 8 - Tiempos de Entrega

# COMMAND ----------

all_data['delivery_time'] = all_data['order_delivered_customer_date'] - all_data['order_purchase_timestamp']
tiempo_entrega = pd.DataFrame(all_data['delivery_time'])
tiempo_entrega.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Grafico de tipo MAP

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import geopandas as gpd
from shapely.geometry import Point, Polygon, shape
import shapely.speedups
shapely.speedups.enable()

mapa = gpd.read_file("/dbfs/mnt/ucom-datos/bcim_2016_21_11_2018.gpkg", layer = "lim_unidade_federacao_a")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Por cantidad de clientes

# COMMAND ----------

mapa.rename({"sigla":"Estado"}, axis = 1, inplace = True)
brasil = mapa.merge(state_customer, on = "Estado", how = "left")

brasil.head()

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC 
# MAGIC brasil.plot(column = "Cantidad de Clientes", cmap= "Reds", figsize = (16,10), legend = True, edgecolor = "black")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Por cantidad de Vendedores

# COMMAND ----------

mapa.rename({"sigla":"Estado"}, axis = 1, inplace = True)
brasil2 = mapa.merge(state_seller, on = "Estado", how = "left")

brasil2.head()

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC 
# MAGIC brasil2.plot(column = "Cantidad de Vendedores", cmap= "Reds", figsize = (16,10), legend = True, edgecolor = "black")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mapas interactivos y heat maps

# COMMAND ----------

ubi.head()

# COMMAND ----------

import folium
from folium.plugins import FastMarkerCluster, Fullscreen, MiniMap, HeatMap, HeatMapWithTime, LocateControl
from branca.element import Figure

# COMMAND ----------

lats = list(ubi['geolocation_lat'].dropna().values)[:30000]
longs = list(ubi['geolocation_lng'].dropna().values)[:30000]

locations = list(zip(lats, longs))

fig = Figure(width=800, height=600)
map1 = folium.Map(location=[-15, -50], zoom_start=4.0)
FastMarkerCluster(data=locations).add_to(map1)
fig.add_child(map1)



# COMMAND ----------

heat_data = ubi.head(90000).groupby(by=['geolocation_lat', 'geolocation_lng'], as_index=False).count().iloc[:, :3]
fig = Figure(width=800, height=600)

map1 = folium.Map(
    location=[-15, -50], 
    zoom_start=4.0, 
    tiles='cartodbdark_matter'
)

HeatMap(
    name='Mapa de Calor',
    data=heat_data,
    radius=10,
    max_zoom=13
).add_to(map1)

fig.add_child(map1)

# COMMAND ----------

heat_data = ubi.head(90000).groupby(by=['geolocation_lat', 'geolocation_lng'], as_index=False).count().iloc[:, :3]
fig = Figure(width=800, height=600)

map1 = folium.Map(
    location=[-15, -50], 
    zoom_start=4.0, 
    tiles='cartodbdark_matter'
)

HeatMap(
    name='Mapa de Calor',
    data=heat_data,
    radius=10,
    max_zoom=13
).add_to(map1)

fig.add_child(map1)

# COMMAND ----------

ubi_price["price"] = pd.to_numeric(ubi_price["price"])

heat_data = ubi_price.head(90000).groupby(by=['geolocation_lat', 'geolocation_lng'], as_index=False).mean().dropna().iloc[:, :3]

fig = Figure(width=800, height=600)

map3 = folium.Map(
    location=[-15, -50], 
    zoom_start=4.0, 
    tiles='cartodbdark_matter'
)

HeatMap(
    name='Mapa de Calor',
    data=heat_data,
    radius=10,
    max_zoom=13
).add_to(map3)

fig.add_child(map3)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Analisis de sentimientos -  Reviews

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

# COMMAND ----------

order_reviews.head()

# COMMAND ----------

reviews = order_reviews[(order_reviews['review_comment_message'] != "No Comment")]
reviews.head()

# COMMAND ----------

import string
exclama = string.punctuation
print(exclama)

# COMMAND ----------



def remove_punctuation(text):
    no_punct=[words for words in text if words not in exclama]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct

reviews['review_comment_message']=reviews['review_comment_message'].apply(lambda x: remove_punctuation(x))
reviews.head()


# COMMAND ----------

comentarios = reviews['review_comment_message'].tolist()

# COMMAND ----------

pt_stopwords = stopwords.words('portuguese')

def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
    return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]

reviews_stopwords = [' '.join(stopwords_removal(review)) for review in comentarios]
reviews_stopwords[:10]

# COMMAND ----------

lista = []
for i in reviews_stopwords:
    lista.append(word_tokenize(i))
lista[:10]

# COMMAND ----------

reviews["tokens"] = lista
reviews.head()

# COMMAND ----------

reviews_5 = reviews[(reviews['review_score'] == "5")]

import itertools
ab = itertools.chain.from_iterable(reviews_5["tokens"])
lista_5 = list(ab)

wordCount_5 = pd.DataFrame(lista_5)
wordCount_5.columns = ['word']
wordCount_5.head()

wordCountsDF_5 = wordCount_5.value_counts().rename_axis("word").reset_index(name ="Cantidad")
wordCountsDF_5.head()
#reviews_4 = reviews[(reviews['review_score'] == "4")]
#reviews_malos = reviews[(reviews['review_score'] < 4)]

# COMMAND ----------

reviews_4 = reviews[(reviews['review_score'] == "4")]

import itertools
ab = itertools.chain.from_iterable(reviews_4["tokens"])
lista_4 = list(ab)

wordCount_4 = pd.DataFrame(lista_4)
wordCount_4.columns = ['word']
wordCount_4.head()

wordCountsDF_4 = wordCount_4.value_counts().rename_axis("word").reset_index(name ="Cantidad")
wordCountsDF_4.head()
#reviews_4 = reviews[(reviews['review_score'] == "4")]
#reviews_malos = reviews[(reviews['review_score'] < 4)]

# COMMAND ----------

reviews_1 = reviews[(reviews['review_score'] == "1")]

import itertools
ab = itertools.chain.from_iterable(reviews_1["tokens"])
lista_1 = list(ab)

wordCount_1 = pd.DataFrame(lista_1)
wordCount_1.columns = ['word']
wordCount_1.head()

wordCountsDF_1 = wordCount_1.value_counts().rename_axis("word").reset_index(name ="Cantidad")
wordCountsDF_1.head()
#reviews_4 = reviews[(reviews['review_score'] == "4")]
#reviews_malos = reviews[(reviews['review_score'] < 4)]

# COMMAND ----------

# MAGIC %md
# MAGIC ##Forecast de ventas:

# COMMAND ----------

order_items.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Podemos usar el modelo de PROPHET, es un modelo para forcasting de ventas creado por META aka Facebook.

# COMMAND ----------

from prophet import Prophet

order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date'])

order_items['Years'] = order_items['shipping_limit_date'].dt.year  
order_items['Month'], order_items['Days'] = (order_items['shipping_limit_date'].dt.month, order_items['shipping_limit_date'].dt.day)
order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date']).dt.date

df_order_items1 = order_items[['shipping_limit_date', 'Days']].rename(columns = {'shipping_limit_date': 'ds', 'Days': 'y'})


model = Prophet()
model.fit(df_order_items1)
future = model.make_future_dataframe(periods=900)  
forecast = model.predict(future)
model.plot(forecast, xlabel = 'Date', ylabel = 'Visit');

# COMMAND ----------

model.plot_components(forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Los graficos de plotly no funcionan en notebooks :/

# COMMAND ----------

from prophet.plot import plot_plotly, plot_components_plotly
from plotly.offline import plot
from plotly.graph_objs import *
import numpy as np

# COMMAND ----------

plot_plotly(model, forecast)

# COMMAND ----------

plot_components_plotly(model, forecast)
