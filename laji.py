# from memory_profiler import profile

# class aa:
    
#     def __init__(self,x):
#         b=[]
#         for i in range(30):
#             b.append([1] * (10**6))

#     #@profile()    
#     def __iter__(self):
#         b=[]
#         for i in range(3):
#             b.append([1] * (10**6))
#         return iter(b)
#             #yield b



# import time


# def main():
#     a=aa(100)
#     @profile()
#     def func(a):
#         for i in a:
#             print(len(i))
#         print(1)
#     while True:
#         time.sleep(1)
#         func(a)
        

# main()


import json
import os
import ipdb
from tqdm import tqdm
# t_dir= "/home/zhongguokexueyuanzidonghuayanjiusuo/xinxin.zhu/3/howto_feature/howto100m_data_json/"
# id = json.load(open('/home/zhongguokexueyuanzidonghuayanjiusuo/xinxin.zhu/3/howto_feature/val_2modal.json'))
# valid_id =[]
# for i in id:
#     try:
#         #ipdb.set_trace()
#         aa = json.load(open(os.path.join(t_dir,i[:2],i+'.json')))
#         valid_id.append(i)
#     except:
#         pass
# print(len(valid_id))
# json.dump(valid_id,open('/home/zhongguokexueyuanzidonghuayanjiusuo/xinxin.zhu/3/howto_feature/val_2modal.json','w'))


import threading

invalid=[]

def func(start,end):
    global invalid
    txt_dir = '/home/zhongguokexueyuanzidonghuayanjiusuo/xinxin.zhu/3/howto_feature/howto100m_data_json/'
    subdir_list = os.listdir(txt_dir)
    subdir_list = subdir_list[start:end]
    
    for s in tqdm(subdir_list):
        subdir = os.path.join(txt_dir,s)
        for ss in os.listdir(subdir):
            fp = os.path.join(subdir,ss)
            try:
                aa=json.load(open(fp))
            except:
                invalid.append(fp)
                print(fp)

                
import time

txt_dir = '/home/zhongguokexueyuanzidonghuayanjiusuo/xinxin.zhu/3/howto_feature/howto100m_data_json/'
subdir_list = os.listdir(txt_dir)
stride = len(subdir_list)//100 + 1
threads=[]

for i in range(100):
    thread =  threading.Thread(target= func,args=(i*stride,(i+1)*stride))
    thread.daemon = True
    thread.start()
    time.sleep(1)
    threads.append(thread)
for thread in threads:
    thread.join()
# try:
#     while True: 
#         print(1)
# except:
#     print('2')
print(invalid)
json.dump(invalid,open('invalid.json','w'))