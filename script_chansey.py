import os
import copy

# Sept 23
code1 = "python3 chansey.py "
code2 = "--train_case "
code3 = "--test_case "
code4 = "--id "

# 1->2
# 2->1
# 3->4
# 4->3
# 1->3
# 4->2

target_pair = []
target_pair.append([1,2])
target_pair.append([2,1])
target_pair.append([3,4])
target_pair.append([4,3])
target_pair.append([1,3])
target_pair.append([4,2])
print(target_pair)

for train_case, test_case in target_pair:
    model_para = "chansey_{:01d}--{:01d}".format(train_case, test_case)
    code = copy.deepcopy(code1)
    code += code2 + str(train_case) + ' '
    code += code3 + str(test_case) + ' '
    code += code4 + model_para
    print('----------------------------------------------')
    print(code)
    print('----------------------------------------------')
    os.system(code)
