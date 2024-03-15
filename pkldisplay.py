import pickle
import os
from hyperopt import Trials

directory_path = r'C:\Users\gklizh\Documents\Workspace\code_and_data12\data\python_related\result\test_para\39_beta'
output_file_path = os.path.join(directory_path, 'combined39_beta.pkl')
output_log_file_path = os.path.join(directory_path, 'log_combined39_beta.txt')  # 新建一个用于记录日志的文件

# 重定向标准输出到日志文件
with open(output_log_file_path, 'w') as log_file:
    #comm_trials_binary_test39_beta20.pkl
    # 直接将每个文件的内容作为一个单独的对象写入
    with open(output_file_path, 'wb') as output:
        for i in range(21):  # 假定有21个文件
            file_name = f'comm_trials_binary_test39_beta{i}.pkl'
            file_path = os.path.join(directory_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    pickle.dump(data, output)  # 将整个对象写入combined文件

    #fname = 'combined.pkl'
    input_file = open(output_file_path, 'rb')

    try:
        while True:
            log_file.write("---------------------------------------------\n")
            trials = pickle.load(input_file)
            # 这里处理每个读取到的trials对象
            #print(trials.trials)  # 举例输出，实际中根据需要处理
            for trial in trials.trials:
                log_file.write(str(trial) + '\n')
            log_file.write("The best para is :\n")
            best = trials.best_trial['result']['params']
            log_file.write(str(best) + '\n')

    except EOFError:
        # 当读取到文件末尾时，会抛出EOFError，此时可以关闭文件
        input_file.close()
