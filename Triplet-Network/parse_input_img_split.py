import os
import random

img_dir = "./output/"
img_path = os.walk(img_dir)
img_dic = {}
img_list = []
img_path_list = []
filenames_filename = "./filenames_filename.txt"
triplets_train_name = "./triplets_train_name.txt"
triplets_valid_name = "./triplets_valid_name.txt"
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
print(total_num-1)

fftrain = open(filenames_filename, "w")
ffvalid = open(filenames_filename, "a")
tftrain = open(triplets_train_name, "w")
tfvalid = open(triplets_valid_name, "w")

# shuffle img_id
random.shuffle(img_list)

# split percentage
p = 0.9

# split
train_id_list = img_list[:int(len(img_list) * p)]
valid_id_list = img_list[int(len(img_list) * p):]
n_train = len(train_id_list)
n_valid = len(valid_id_list)
print(n_train)
print(n_valid)

# write files
for i in range(len(train_id_list)):
    neg = random.randint(0, n_train-1)
    while(neg == i):
        neg = random.randint(0, n_train-1)
    fftrain.write(img_dir+train_id_list[i]+"_bottom.jpg"+"\n")
    tftrain.write(str(i*2)+" "+str(neg*2+1)+" "+str(i*2+1)+"\n")
    fftrain.write(img_dir+train_id_list[i]+"_top.jpg"+"\n")
    tftrain.write(str(i*2+1)+" "+str(neg*2)+" "+str(i*2)+"\n")
fftrain.close()
tftrain.close()

for i in range(len(valid_id_list)):
    neg = random.randint(0, n_valid-1)
    while(neg == i):
        neg = random.randint(0, n_valid-1)
    ffvalid.write(img_dir+valid_id_list[i]+"_bottom.jpg"+"\n")
    tfvalid.write(str(n_train*2+i*2)+" "+str(n_train*2+neg*2+1)+" "+str(n_train*2+i*2+1)+"\n")
    ffvalid.write(img_dir+valid_id_list[i]+"_top.jpg"+"\n")
    tfvalid.write(str(n_train*2+i*2+1)+" "+str(n_train*2+neg*2)+" "+str(n_train*2+i*2)+"\n")

ffvalid.close()
tfvalid.close()