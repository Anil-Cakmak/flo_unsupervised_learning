import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

df_ = pd.read_csv("datasets/flo_data_20K.csv")
df = df_.copy()


def check_df(dataframe, head=5, quantiles=[0.05, 0.50, 0.95, 0.99]):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    na_dataframe = dataframe.isnull().sum()
    print(na_dataframe.loc[na_dataframe != 0])
    print("##################### Index ####################")
    print(dataframe.index)
    print("##################### Quantiles #####################")
    print(dataframe.describe(quantiles).T)


check_df(df)

#####################################################################
# Müşterileri segmentlerlerken kullanılacak değişkenlerin seçilmesi.
#####################################################################

date_columns = df.columns[df.columns.str.contains("date")]
# Index(['first_order_date', 'last_order_date', 'last_order_date_online', 'last_order_date_offline'], dtype='object')

df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max()  # Timestamp('2021-05-30 00:00:00')
analysis_date = dt.datetime(2021, 6, 1)

df["tenure"] = (analysis_date - df["last_order_date"]).dt.days
df["recency"] = (df["last_order_date"] - df["first_order_date"]).dt.days

model_df = df[["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
               "customer_value_total_ever_online", "recency", "tenure"]]


###################################
# K-Means ile Müşteri Segmentasyonu
###################################

# Çarpıklık inceleme.


def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return


plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df, 'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df, 'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df, 'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df, 'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df, 'recency')
plt.subplot(6, 1, 6)
check_skew(model_df, 'tenure')
plt.tight_layout()
plt.show(block=True)

# Normal dağılımın sağlanması için log transformation uygulanması.
model_df = model_df.apply(lambda x: np.log1p(x), axis=1)

# Standartlaştırma.
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df = pd.DataFrame(model_scaling, columns=model_df.columns)

# Optimum küme sayısını belirlenmesi.
kmeans = KMeans(n_init=10)

elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
plt.show(block=True)
print(elbow.elbow_value_)  # 7

# Modeliniz oluşturulması ve müşterilerin segmentlenmesi.
k_means = KMeans(n_clusters=elbow.elbow_value_, n_init=10, random_state=42).fit(model_df)
segments = k_means.labels_

final_df = df[["master_id", "order_num_total_ever_online",
               "order_num_total_ever_offline", "customer_value_total_ever_offline",
               "customer_value_total_ever_online", "recency", "tenure"]]

final_df.loc[:, "kmeans_segment"] = segments

# 4. Her bir segmentin istatistiksel olarak incelenmesi.
final_df.groupby("kmeans_segment").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                        "order_num_total_ever_offline": ["mean", "min", "max"],
                                        "customer_value_total_ever_offline": ["mean", "min", "max"],
                                        "customer_value_total_ever_online": ["mean", "min", "max"],
                                        "recency": ["mean", "min", "max"],
                                        "tenure": ["mean", "min", "max", "count"]})

###############################################################
# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
###############################################################

# Optimum küme sayısının belirlenmesi.
hc_complete = linkage(model_scaling, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=10)
plt.axhline(y=1.3, color='r', linestyle='--')
plt.show(block=True)

# Modelin oluşturulması ve müşterilerin segmentlenmesi.
hc = AgglomerativeClustering(n_clusters=5)
hc_segments = hc.fit_predict(model_df)

final_df.loc[:, "hc_segment"] = hc_segments

# Her bir segmentin istatistiksel olarak incelenmesi.
final_df.groupby("hc_segment").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                    "order_num_total_ever_offline": ["mean", "min", "max"],
                                    "customer_value_total_ever_offline": ["mean", "min", "max"],
                                    "customer_value_total_ever_online": ["mean", "min", "max"],
                                    "recency": ["mean", "min", "max"],
                                    "tenure": ["mean", "min", "max", "count"]})
