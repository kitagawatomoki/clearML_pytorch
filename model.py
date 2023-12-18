import timm

def get_model(name, num_classes,pretrained=True):
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    if name == "convnext_base":
        batch_size = 32*2
    elif name == "darknet17":
        batch_size = 128*2
    elif name == "densenet121":
        batch_size = 32*2
    elif name == "efficientnet_b0":
        batch_size = 32*2
    elif name == "mobilenetv3_small_050":
        batch_size = 512*2
    elif name == "resnet18":
        batch_size = 256*2
    elif name == "vgg11_bn":
        batch_size = 96*2
    elif name == "vit_base_patch8_224":
        batch_size = 4*2
    else:
        batch_size = 32

    return model, batch_size
