import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/cudnn/bin")
import pandas as pd
from model1 import baseline_model


import json

import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv(r"E:\学习资料\天文\作业五\desktop\lowmass_0neten.csv")
    X = np.expand_dims(df.values[1:, 21:66].astype(float), axis=2)
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model=baseline_model(subclass_amount=21)
    weighs_path = "./save_weights/myAlex_45.h5"
    #assert os.path.exists(img_path), "file: '{}' dose not exist.".format(weighs_path)
    model.load_weights(weighs_path)
    result = np.squeeze(model.predict(X))
    # print(result.shape)
    # print(type(result))
    # np.savetxt('testdataresult.csv',result,delimiter=',')
    predict_class = np.argmax(result,axis=0)
    print(predict_class.shape)
    print(type(predict_class))
    np.savetxt('testdataresult2.csv', predict_class, delimiter=',')
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
    #                                              result[predict_class])
    # plt.title(print_res)
    # print(print_res)
    # plt.show()

if __name__ == '__main__':
    print('main')
    main()