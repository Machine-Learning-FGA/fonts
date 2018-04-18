import pandas as pd
import operator
import sys
import zipfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

with zipfile.ZipFile("fonts.zip") as fonts_zip:
    csv_files = fonts_zip.namelist()
    x = pd.DataFrame()
    fonts_list = {}
    for csv in csv_files:
        font = fonts_zip.open(csv)
        size = font.read()
        fonts_list[font] = sys.getsizeof(size)
        
    sorted_list = sorted(fonts_list.items(), key=operator.itemgetter(1), reverse=True)
    chosen_ones = sorted_list[:5]
    fonts_list = []
    
    for font in chosen_ones:
        font = fonts_zip.open(font[0].name)
        df = pd.read_csv(font, index_col=None, header=0, nrows=5000)
        fonts_list.append(df)

    x = pd.concat(fonts_list)

y_aux = x[['font']]
x.drop("h", axis=1, inplace=True)
x.drop("w", axis=1, inplace=True)
x.drop('font', axis=1, inplace=True)
x.drop('fontVariant', axis=1, inplace=True)


#transforma label em inteiro
y = y_aux.apply(LabelEncoder().fit_transform)


x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.5, random_state=42)

for i in range(4, 8):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_treino, y_treino.values.ravel())
    print("Knn with " + str(i) + " neighbors")
    print(knn.score(x_teste, y_teste))

