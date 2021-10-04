import numpy as np
import pandas as pd
df = pd.read_csv(r"E:\学习资料\天文\作业五\normalize2021102\full_match_rizjhkw1_id_ra_dec.csv")
#X = np.expand_dims(df.values[1:, 22:67].astype(float), axis=2)
# df_num=df.iloc[:,22:67]
# #print(df_num)
# df_normal=df_num/df_num.max()
# #print(df_normal)
# df_normal.to_csv("full_match_normalize_extinc.csv",float_format='%.3f')
df_num=df.iloc[:,2:12]
#print(df_num)
df_normal=df_num/df_num.max()
#print(df_normal)
df_normal.to_csv("full_match_normalize_extinc_rizjhkw.csv",float_format='%.5f')