# SENet_Segmentation


- 2d_unetse_model

'''
model = torch.nn.DataParallel(
    Dilated_UNet(inChannel=2, num_classes=2, init_features=32, network_depth=3, bottleneck_layers=3))
model.cuda()
'''

- 2d_densenetse_model

'''
model=torch.nn.DataParallel(FCDenseNet103(2))
model.cuda()
'''

- 3d_unetse_model

'''
model = torch.nn.DataParallel(
    UNet3D_SE(inChannel=2, num_classes=2, init_features=32, network_depth=3, bottleneck_layers=2))
model.cuda()
'''