import pandas as pd
import glob


file_pattern = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\community\communityScores_compare_32_*.csv'
file_list = glob.glob(file_pattern)


file_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))


combined_df = pd.DataFrame()


for i, file in enumerate(file_list):
  
    df = pd.read_csv(file, usecols=[0], header=None)  
 
    combined_df[i] = df.iloc[:, 0]


combined_df.to_csv(r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\community\correct_combined02.csv', index=False, header=False)
