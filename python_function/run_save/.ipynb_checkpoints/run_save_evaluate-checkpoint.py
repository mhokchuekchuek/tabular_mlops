import sys
sys.path.append("/code/notebook/")
from script.run_save.save_evaluate import to_csv_eval

# load target file
f = open("/code/notebook/script/task.txt", 'r')
task = [x for x in f.readlines()]

to_csv_eval(task[-1])