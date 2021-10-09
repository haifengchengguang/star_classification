import itertools
import json

from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from collections import Counter
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"E:\学习资料\天文\作业五\normalize2021102\full_match_rizjhkw1_id_ra_dec_distance_extinc_1009_45.csv")
X = np.expand_dims(df.values[1:, 22:67].astype(float), axis=2)
Y = df.values[1:, 70]
#subclass_amount=21
# 恒星分类编码为数字
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)
classes_1=encoder.classes_
classes_1_list=classes_1.tolist()

subclass_amount=len(classes_1_list)
print(subclass_amount)
a=list(range(subclass_amount))

d=zip(a,classes_1)
c=dict(d)
print(c)
json_str = json.dumps(c)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)
# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.1, random_state=0)
print(type(X_test))
print(type(Y_train))
print(Counter(Y))

# ytest_class=encoder.inverse_transform(Y_test)
# print(ytest_class)

sm = SMOTE(k_neighbors=1)
X_train_=X_train.reshape(X_train.shape[0],-1)
X_smotesampled, y_smotesampled = sm.fit_resample(X_train_,Y_train)
X_smotesampled=X_smotesampled.reshape(X_smotesampled.shape[0], 45,1)
print(type(X_smotesampled.shape))

#混淆矩阵定义
def plot_confusion_matrix(cm, classes,i_1, title='Confusion matrix', cmap=plt.cm.jet):
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
                 color="white" if cm[i, j] > thresh else "black",fontsize=3)
    plt.tight_layout()
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.savefig(f'./save_weights_extinc_1009_png/test_{i_1}.png', dpi=400, bbox_inches='tight', transparent=False)
    plt.show()
def plot_confuse(model, x_val, y_val,i):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    truelabel_unique=np.unique(truelabel)
    print(truelabel_unique)
    #truelabel_list=truelabel_unique.tolist()
    #print(truelabel_list)
    predictions_unique=np.unique(predictions)
    print(predictions_unique)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1),i)

# 卷积网络可视化
def visual(model, data, num_layer=1):
    # data:图像array数据
    # layer:第n层的输出
    layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
    f1 = layer([data])[0]
    print(f1.shape)
    num = f1.shape[-1]
    print(num)
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i+1)
        plt.imshow(f1[:, :, i] * 255, cmap='gray')
        plt.axis('off')
    plt.show()
