import os
import random

img_dir = "./output/"
img_path = os.walk(img_dir)
img_dic = {}
img_list = []
img_path_list = []
filenames_filename = "./filenames_filename.txt"
triplets_file_name = "./triplets_file_name.txt"
i = 0
for path,dir_list,file_list in img_path:  
    for file_name in file_list:  
        file_path = os.path.join(path, file_name)
        id_no = file_name.split("_")[0]
        if id_no in img_dic:
            img_dic[id_no] += 1
        else:
            img_dic[id_no] = 1
total_num = 0
for key, val in img_dic.items():
    if val == 2:
        total_num += 1
        img_list.append(key)
    if val > 2:
        print(key, val)
print(total_num)

ff = open(filenames_filename, "w")
tf = open(triplets_file_name, "w")


for i in range(len(img_list)):
    neg = random.randint(0,total_num-1)
    while(neg == i):
        neg = random.randint(0,total_num-1)
    ff.write(img_dir+img_list[i]+"_bottom.jpg"+"\n")
    tf.write(str(i*2)+" "+str(neg*2+1)+" "+str(i*2+1)+"\n")
    ff.write(img_dir+img_list[i]+"_top.jpg"+"\n")
    tf.write(str(i*2+1)+" "+str(neg*2)+" "+str(i*2)+"\n")

ff.close()
tf.close()