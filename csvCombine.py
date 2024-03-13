import pandas as pd
import glob

# 定义文件匹配模式，假定这些文件位于当前工作目录下
#file_pattern = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\community\communityScores_compare_32_*.csv'
file_pattern = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\community\communityScores_compare45_*.csv'

file_list = glob.glob(file_pattern)
#communityScores_compare26_3_19.csv
# 按文件名中的数字排序，确保列的顺序正确
file_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# 初始化一个空的DataFrame，用于存储合并后的数据
combined_df = pd.DataFrame()

# 逐个读取文件
for i, file in enumerate(file_list):
    # 读取每个文件的第一列数据
    df = pd.read_csv(file, usecols=[0], header=None)  # 假设我们只关心每个文件的第一列
    # 将该列数据添加到合并后的DataFrame中，不使用'File_'前缀
    combined_df[i] = df.iloc[:, 0]

# 保存合并后的DataFrame到CSV文件
combined_df.to_csv(r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\community\correct_combined45.csv', index=False, header=False)
