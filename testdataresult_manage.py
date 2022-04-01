import numpy as np
import pandas as pd
from tqdm import tqdm
# df = pd.read_csv(r"testdataresult_sql.csv")
# X = np.expand_dims(df.values[0:, 0:24].astype(float), axis=1)
# print(X)
# xlist=X.tolist()
# print(xlist)
# result=np.argmax(xlist,1)
# print(result)
# print(result.shape)
# #np.savetxt('testdataresult_sql_max.csv', result, delimiter=',')
# import csv
# csv_reader = csv.reader(open("./testdataresult_sql.csv"))
# result=np
# #sep 指定分隔符
data = pd.read_table("gaia_color_cut/20220324/gaia_bp_cut_predit.csv",sep=",")
max_value=data.max(axis=1)

#print(max_value)
max_index=data.idxmax(axis=1)
#print(max_index)
#print(data)
#lenth=len(data)
# lenth=800000
# result=np.empty(shape=[0,2],dtype=float)
# #print(result)
# result2=np.empty(shape=[0,2],dtype=float)
# for i in tqdm(range(0, lenth)):
#     a=data.iloc[i]
#     #print(type(a))
#     b=a.nlargest(2)
#     index=b.index.tolist()
#     value=b.tolist()
#     #print(value[0])
#     #print(a)
#     #array1=pd.to_numeric(a.str.get(''),errors='coerce').nlargest(2,keep='all')
#     # maxvalue=a.max()
#     # max_index=a[a.values==maxvalue]
#     # max_index_str=str(max_index)
#     # #print(type(max_index))
#     # indexlist=max_index.index.tolist()
#     # print(indexlist[0])
#     # print(max_index.iloc[0])
#     result=np.append(result,[[index[0],value[0]]],axis=0)
#     result2 = np.append(result2, [[index[1], value[1]]], axis=0)
#     #print(i)
#     # #result=np.insert()
#     # print(i)
# # print(result.shape)
# # print("------------------------------------------------------------------------------------")
# # print(result2.shape)
# print("best")
np.savetxt('gaia_color_cut/20220324/gaia_bp_cut_predict_index.csv', max_index, fmt="%s", delimiter=',')
# print("second")
np.savetxt('gaia_color_cut/20220324/gaia_bp_cut_predict_value.csv', max_value,fmt="%s",delimiter=',')
print("end")

    # row_max_value=row.max()
    # row_max_index=row[row.values==row_max_value]
    # row_indexlist = row_max_index.index.tolist()
    # print(row_indexlist[0])
    # print(row_max_index.iloc[0])

# for line in csv_reader:
#     print(line)
#     print(type(line))
#     value=max(line)
#     index=line.index(value)
#     line[index]=0
#     second_value=max(line)
#     second_inex=line.index(second_value)
#     result=np.add([index,value,second_inex,second_value])
# print(result)
