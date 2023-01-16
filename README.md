# PZ-5-
## Подключим необходимые библиотеки import numpy as np
import cv2
from keras.datasets import mnist from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop from keras.utils import np_utils;
import pandas as pd;
import matplotlib.pyplot as plt;
batch_size = 128; #размер партии данных, которая поступает на обучение, т.е. nr_classes = 10; # Количество классов
nr_iterations = 20; #Количество эпох (количество итераций) #Загрузка датасета
with np.load('mnist.npz') as data:
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test'];
x_train = x_train.reshape(60000, 784); # 60000 - это количество картинок, 784 - размер 1-й картинки (28х28=784)
x_test = x_test.reshape(10000, 784); # Уточняем тип данных
x_train = x_train.astype('float32') x_test = x_test.astype('float32')
# Нормируем входные значения, что бы получить числа от 0 до 1!!! x_train /= 255
x_test /= 255
# Делаем 10 бинарных столбцов (так как 10 цифр), формируем зависимые переменные, ответы. y_train = np_utils.to_categorical(y_train, nr_classes)
y_test = np_utils.to_categorical(y_test, nr_classes) a=pd.DataFrame(y_train)# Это не обязательный параметр. a.head() # Посмотрим, что содержится в данных Y_train
# Описываем сеть. Входной слой, один внутренний слой, выходной слой. Всего три слоя!!! model = Sequential()# Начало создания сети.
model.add(Dense(196, input_shape=(784,))) # входной (784 = 28х28 нейрона) и второй слой (196 нейронов).
model.add(Activation('relu')); model.add(Dropout(0.5));
model.add(Dense(10))# количество нейронов в выходном слое; model.add(Activation('softmax'));
model.summary() #Не обязательный параметр. # Параметры целевой функции потерь. model.compile(loss='categorical_crossentropy',
optimizer=Adam(), metrics=['accuracy'])
np.random.seed(1337) # для воспроизводимости сети # Обучаем сеть.
net_res_1 = model.fit(x_train, y_train,epochs=nr_iterations, batch_size = batch_size,
verbose = 1, validation_data = (x_test, y_test));
score = model.evaluate(x_test, y_test, verbose = 0)# Вычислим ошибки обучения. print(score)

# Визуализация работы программы.
 
plt.imshow(x_test[9].reshape(28,28))# Посмотрим на картинку №28. df=x_test[9].reshape(1,784); #Растянем картинку в вектор.
ans=model.predict(df);# Воспользуемся моделью для решения задачи классификации. ans=pd.DataFrame(ans);# Представим ответ в формате DataFrame для удобства round(ans,2)# Посмотрим на результат.


# Визуализация работы программы. img=cv2.imread("SYS.png",0) #Загружаем изображение.

newimg=cv2.resize(img, (28,28)) #Меняем размер изображения. dZ=newimg.reshape(-1,784); #Растянем картинку в вектор.
ans=model.predict(dZ);# Воспользуемся моделью для решения задачи классификации. ans=pd.DataFrame(ans);# Представим ответ в формате DataFrame для удобства просмотра. round(ans,2)# Посмотримd на результат.
plt.imshow(newimg) #Смотрим на картинку.
