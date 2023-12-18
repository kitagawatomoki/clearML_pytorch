import glob2
import os

def get_cifar_10():
    train_list = []
    test_list = []
    class_list = ["airplane", "automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    for c in class_list:
        train_list += sorted(glob2.glob(os.path.join("/Corpus3/CIFAR-10-images/train", c, "*.jpg")))
        test_list += sorted(glob2.glob(os.path.join("/Corpus3/CIFAR-10-images/test", c, "*.jpg")))

    return train_list,test_list