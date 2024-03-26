import re
import os

dataset = 'twitter'
dataset = 'rest'
# result_dir = '/nas2/lishengping/caiyun_projects/rgat_absa/'
result_dir = './'

# for dataset in ['twitter', 'rest', 'laptop']:
for dataset in ['rest']:

    path = f'data/output-gcn-{dataset}/eval_results.txt'
    result_path = os.path.join(result_dir, path)

    text = open(result_path).read()
    accs = re.findall('acc \= (.*)\n', text)
    f1s = re.findall('f1 \= (.*)\n', text)
    accs = [round(float(a), 4) for a in accs]
    f1s = [round(float(a), 4) for a in f1s]

    max_f1 = max(f1s)
    max_index = f1s.index(max_f1)
    max_acc = accs[max_index]
    print(f'dataset: {dataset} max f1: {max_f1} acc: {max_acc}')

