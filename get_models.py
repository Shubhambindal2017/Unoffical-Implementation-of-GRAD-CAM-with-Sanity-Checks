import torch
import torchvision
import numpy as np 

def get_default_model(modelname):

    if modelname == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        model.eval()
        model.cuda()
        model_imsize = 224,224

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)

        
    '''
    elif modelname == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.eval()
        model.cuda()
        model_imsize = 224,224

        

    elif modelname == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        model.eval()
        model.cuda()
        model_imsize = 227,227
    '''
        
    model_mean,model_std = vgg_mean,vgg_std 
    preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(model_imsize),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean = model_mean,std=model_std)
                                           ])
    
    return model, model_imsize, preprocess


def cascade_randomization(modelname = None, num_layers_from_last = None):  

  if modelname == 'vgg19':
    model = torchvision.models.vgg19(pretrained=True)
    model_imsize = 224,224

    vgg_mean = (0.485, 0.456, 0.406)
    vgg_std = (0.229, 0.224, 0.225)

  num = -1*num_layers_from_last

  conv2d_keys = []
  for key in model.features._modules.keys():

    if(isinstance(model.features._modules[key],torch.nn.modules.Conv2d)):
      conv2d_keys.append(key)

  for conv2d_key in conv2d_keys[num:]:

    layer = model.features._modules[conv2d_key]
    #print(layer)
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    kernel_size = layer.kernel_size
    stride = layer.stride
    padding = layer.padding

    model.features._modules[conv2d_key] = torch.nn.modules.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

  model.eval()
  model.cuda()

  model_mean,model_std = vgg_mean,vgg_std 
  preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(model_imsize),
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(mean = model_mean,std=model_std)
                                          ])
  
  return model, model_imsize, preprocess


def independent_randomization(modelname = None, layer = None):  

  if modelname == 'vgg19':
    model = torchvision.models.vgg19(pretrained=True)
    model_imsize = 224,224

    vgg_mean = (0.485, 0.456, 0.406)
    vgg_std = (0.229, 0.224, 0.225)

 
  conv2d_keys = []
  for key in model.features._modules.keys():

    if(isinstance(model.features._modules[key],torch.nn.modules.Conv2d)):
      conv2d_keys.append(key)

  conv2d_key = conv2d_keys[-1*layer]

  layer = model.features._modules[conv2d_key]
  #print(layer)
  in_channels = layer.in_channels
  out_channels = layer.out_channels
  kernel_size = layer.kernel_size
  stride = layer.stride
  padding = layer.padding

  model.features._modules[conv2d_key] = torch.nn.modules.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

  model.eval()
  model.cuda()

  model_mean,model_std = vgg_mean,vgg_std 
  preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(model_imsize),
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(mean = model_mean,std=model_std)
                                          ])
    
  return model, model_imsize, preprocess





