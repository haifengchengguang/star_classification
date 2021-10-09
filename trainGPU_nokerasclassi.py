# -*- coding: utf8 -*-
import os
import json

from model1 import baseline_model
from pretreatment import subclass_amount, X_smotesampled, y_smotesampled, X_test, Y_test

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/cudnn/bin")
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from collections import Counter

tf.config.experimental.list_physical_devices('GPU')

# subclass_amount=21
#
# # 载入数据
# df = pd.read_csv(r"E:\学习资料\天文\作业五\normalize2021102\full_match_rizjhkw1_id_ra_dec.csv")
# X = np.expand_dims(df.values[1:, 22:67].astype(float), axis=2)
# Y = df.values[1:, 70]
#
# # 恒星分类编码为数字
# encoder = LabelEncoder()
# Y_encoded = encoder.fit_transform(Y)
# Y_onehot = np_utils.to_categorical(Y_encoded)
# classes_1=encoder.classes_
# a=list(range(subclass_amount))
# classes_1.tolist()
# d=zip(a,classes_1)
# c=dict(d)
# print(c)
# json_str = json.dumps(c)
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)
# # 划分训练集，测试集
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.1, random_state=0)
# print(type(X_test))
# print(type(Y_train))
# print(Counter(Y))
#
# # ytest_class=encoder.inverse_transform(Y_test)
# # print(ytest_class)
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(k_neighbors=1)
# X_train_=X_train.reshape(X_train.shape[0],-1)
# X_smotesampled, y_smotesampled = sm.fit_resample(X_train_,Y_train)
# X_smotesampled=X_smotesampled.reshape(X_smotesampled.shape[0], 45,1)
# print(type(X_smotesampled.shape))
#print(X_smotesampled)

# 定义神经网络
# def baseline_model(subclass_am):
#     model = Sequential()
#     # model.add(ZeroPadding1D((3,3),input_shape=(45, 1)))
#     model.add(Conv1D(48, kernel_size=3, strides=4, activation='relu', input_shape=(45, 1)))
#     model.add(MaxPooling1D(pool_size=3, strides=2))
#     model.add(Conv1D(128, kernel_size=3, padding="same", activation='relu'))
#     model.add(MaxPooling1D(pool_size=3, strides=2))
#     model.add(Conv1D(192, kernel_size=3, padding="same", activation='relu'))
#     model.add(Conv1D(192, kernel_size=3, padding="same", activation='relu'))
#     model.add(Conv1D(128, kernel_size=3, padding="same", activation='relu'))
#     model.add(MaxPooling1D(pool_size=1, strides=2))
#     model.add(Flatten())
#     model.add(Dropout(0.5))
#     model.add(Dense(2048, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(2048, activation='relu'))
#     model.add(Dense(subclass_am, activation='softmax'))
#     # plot_model(model, to_file='./model_classifier.png', show_shapes=True)
#     print(model.summary())
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model


model = baseline_model(subclass_amount=subclass_amount)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#                   metrics=["accuracy"])

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights_extinc_1009/myAlex_{epoch}.h5',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                monitor='val_loss')]
#
epochs = 200
BATCH_SIZE = 32
# tensorflow2.1 recommend to using           fit
# history = model.fit(x=train_dataset,
#                         steps_per_epoch=train_num // batch_size,
#                         epochs=epochs,
#                         validation_data=val_dataset,
#                         validation_steps=val_num // batch_size,
#                         callbacks=callbacks)
history = model.fit(X_smotesampled,
                    y_smotesampled,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, Y_test),
                    validation_freq=1,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=1)
# plot loss and accuracy image
print(history.history.keys())
history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["accuracy"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_accuracy"]

# figure 1
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# figure 2
plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

#混淆矩阵定义
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    print(len(classes))
    #plt.xticks(tick_marks, ('F0', 'F4', 'K0', 'K1', 'k2', 'K3', 'K4', 'K5', 'L','M0','M1','M2','M3','M4','M5','M6','M7','M8','M9','gM9','sdm'))
    #plt.yticks(tick_marks, ('F0', 'F4', 'K0', 'K1', 'k2', 'K3', 'K4', 'K5', 'L','M0','M1','M2','M3','M4','M5','M6','M7','M8','M9','gM9','sdm'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=4)
    plt.tight_layout()
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.savefig('test_xx.png', dpi=400, bbox_inches='tight', transparent=False)
    plt.show()
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    predictions_unique = np.unique(predictions)
    predictions_list=predictions_unique.tolist()
    print(predictions_list)
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    truelabel_unique=np.unique(truelabel)
    #print(truelabel)
    print(truelabel_unique)
    truelabel_list=truelabel_unique.tolist()
    print(truelabel_list)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))

# # 加载模型用做预测
# json_file = open(r"model.json", "r")
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model.h5")
# print("loaded model from disk")
# loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # 分类准确率
# print("The accuracy of the classification model:")
# scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
# print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# # 输出预测类别
# predicted = loaded_model.predict(X)
# predicted_label = loaded_model.predict_classes(X)
# print("predicted label:\n " + str(predicted_label))

# 显示混淆矩阵
plot_confuse(model, X_test, Y_test)
