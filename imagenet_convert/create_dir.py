import os
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(path + "---OK---")
    else:
        print(path + "---There is this folder!---")

if __name__ == '__main__':
    data = []
    # 设置文件对象并读取每一行文件，这个文件夹中保存了对应的验证数据集图像名和图像的标签
    for line in open("/home/ma-user/modelarts/work/jjw/SFT/lqf/1d-tokenizer/imagenet_convert/txt/classes.txt", "r"):  
        data.append(line)
    for a in data:
            # folder=file+line
        a = a.strip()
        folder = a[9:]
            # strip()方法移除字符串头尾指定的字符
        #folder = folder.strip()
        mkdir('/home/ma-user/modelarts/work/jjw/SFT/lqf/ILSVRC/Data/CLS-LOC/val/{}'.format(folder))
