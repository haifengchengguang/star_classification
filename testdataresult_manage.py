import numpy as np
import pandas as pd
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
data = pd.read_table("testdataresult_sql.csv",sep=",")
#print(data)
#lenth=len(data)
lenth=200
result=np.empty(shape=[lenth,2],dtype=float)
for i in range(0, lenth):
    a=data.iloc[i]
    maxvalue=a.max()
    max_index=a[a.values==maxvalue]
    max_index_str=str(max_index)
    #print(type(max_index))
    indexlist=max_index.index.tolist()
    print(indexlist[0])
    print(max_index.iloc[0])
    result=np.append(result,[[indexlist[0],max_index.iloc[0]]],axis=0)
    print(i)
np.savetxt('testdataresult_subclass.csv', result, fmt="%.5f",delimiter=',')
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
