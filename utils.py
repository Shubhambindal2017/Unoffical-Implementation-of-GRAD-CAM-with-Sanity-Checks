import torch
import numpy as np
from matplotlib import cm
from PIL import Image
from skimage import transform
from skimage import io
import skimage
import sys

tensor_to_numpy = lambda t:t.detach().cpu().numpy()

def get_saliency(model,ref,class_of_interest,model_imsize,last_spatial_layer):
    ref_scores = model(ref)
    gradcam_loss = torch.sum(ref_scores[:,class_of_interest])
    #print(gradcam_loss)
    gradcam_loss.backward()

    Z = 1.

    if True:
        alpha_c_k = (1/Z) * torch.sum(last_spatial_layer.our_grad_out,(2,3))


    #print(last_spatial_layer.our_grad_out.shape)
    #print(alpha_c_k.shape)

    alpha_into_A = alpha_c_k.unsqueeze(-1).unsqueeze(-1) * last_spatial_layer.our_feats
    #print(alpha_c_k.unsqueeze(-1).unsqueeze(-1).shape)
    #print(last_spatial_layer.our_feats.shape)
    #print(alpha_into_A.shape)
    
    alpha_into_A_channelsum = torch.sum(alpha_into_A,1)
    #print(alpha_into_A_channelsum.shape)

    L_c = torch.nn.functional.relu(alpha_into_A_channelsum)
#     print(L_c.shape)

    L_c_np = tensor_to_numpy(L_c)

    #import pdb; pdb.set_trace() 
    print(f'L_c_np max : {L_c_np.max()}')

    if L_c_np.max() != 0:
      L_c_np = L_c_np/(L_c_np.max((1,2))[:, np.newaxis, np.newaxis]) # + sys.float_info.epsilon)   #[:,None,None]
    
    #import pdb; pdb.set_trace()


    heat_map = list(map(lambda t:transform.resize((t*255.).astype(np.uint8),model_imsize),L_c_np))
    heat_map = np.array(heat_map)
    #TODO why did i transpose
    #print(heat_map.shape)

    
    return L_c_np,heat_map,ref_scores

def batch_overlay(heat_map,im, model_imsize):
    #TODO Write code for converting a batch of heat mapps into cm.jet images, and overlay them onto the reference image
    heat_map_jet = list(map(lambda h:cm.jet(h),heat_map))
    heat_map_jet = np.array(heat_map_jet)[:,:,:,:3]
    # heat_map_jet = np.transpose(heat_map_jet, (2,0,1,3))
    heat_map_jet_pil = list(map(lambda t: Image.fromarray(np.uint8(t*255)), heat_map_jet))

    #heat_map_jet_pil = Image.fromarray(np.uint8(heat_map))
    #heat_map_jet.shape
    
    saliency_overlayed = []
    for im_i,h in zip(im,heat_map_jet_pil):
        ref_i_pil = Image.fromarray((transform.resize(im_i,model_imsize)*255).astype(np.uint8))
        s = Image.blend(ref_i_pil,h,alpha=0.5)
#         s = np.array(s)
        saliency_overlayed.append(s)
#     saliency_overlayed = np.array(saliency)
    return saliency_overlayed,heat_map_jet_pil

