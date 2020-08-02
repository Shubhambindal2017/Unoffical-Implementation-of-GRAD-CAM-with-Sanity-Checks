def prepare_model_for_gradcam(modelname,model):
    if modelname == 'vgg19':
        last_spatial_layer = model.features._modules['36']
        pass

    '''
    elif modelname == 'resnet18':
        last_spatial_layer = model._modules['layer4']._modules['1']._modules['bn2']
        pass
    elif modelname == 'alexnet':
        last_spatial_layer = model._modules['features']._modules['12']
        pass
    '''
    
    def fwdhook(self,input,output):
        self.our_feats = output
    hooked_last_spatial = last_spatial_layer.register_forward_hook(fwdhook)

    def bwdhook(self,grad_in,grad_out):
        self.our_grad_out = grad_out[0]
    bwd_hooked_last_spatial = last_spatial_layer.register_backward_hook(bwdhook)
    
    return last_spatial_layer,hooked_last_spatial,bwd_hooked_last_spatial