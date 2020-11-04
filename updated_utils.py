import torch
import numpy as np
from matplotlib import cm
from PIL import Image
from skimage import transform
from skimage import io
import skimage
import sys

tensor_to_numpy = lambda t:t.detach().cpu().numpy()

def get_saliency(model, refs, categories, model_imsize, last_spatial_layer):
    
    refs_scores = model(refs)

    #import pdb; pdb.set_trace()
    gradcam_loss = None

    img_num = 0

    for class_of_interest in categories:

      if gradcam_loss == None:
        gradcam_loss = refs_scores[img_num ,class_of_interest]
    
      else:

        gradcam_loss = torch.add(gradcam_loss, refs_scores[img_num ,class_of_interest])

      img_num += 1

    gradcam_loss.backward()

    Z = 1.

    if True:
        alpha_c_k = (1/Z) * torch.sum(last_spatial_layer.our_grad_out,(2,3))

    alpha_into_A = alpha_c_k.unsqueeze(-1).unsqueeze(-1) * last_spatial_layer.our_feats

    alpha_into_A_channelsum = torch.sum(alpha_into_A,1)
    L_c = torch.nn.functional.relu(alpha_into_A_channelsum)

    L_c_np = tensor_to_numpy(L_c)

    L_c_np = L_c_np/(L_c_np.max((1,2))[:, np.newaxis, np.newaxis] + sys.float_info.epsilon)  #[:,None,None]

    heat_map = list(map(lambda t:transform.resize((t*255.).astype(np.uint8),model_imsize),L_c_np))
    heat_map = np.array(heat_map)


    
    return L_c_np,heat_map,refs_scores


def batch_overlay(heat_maps,ims, model_imsize):
    #TODO Write code for converting a batch of heat mapps into cm.jet images, and overlay them onto the reference image
    heat_map_jet = list(map(lambda h:cm.jet(h),heat_maps))
    heat_map_jet = np.array(heat_map_jet)[:,:,:,:3]
    # heat_map_jet = np.transpose(heat_map_jet, (2,0,1,3))
    heat_map_jet_pil = list(map(lambda t: Image.fromarray(np.uint8(t*255)), heat_map_jet))

    saliency_overlayed = []
    for im_i,h in zip(ims,heat_map_jet_pil):
        ref_i_pil = Image.fromarray((transform.resize(im_i,model_imsize)*255).astype(np.uint8))
        s = Image.blend(ref_i_pil,h,alpha=0.5)
#         s = np.array(s)
        saliency_overlayed.append(s)
#     saliency_overlayed = np.array(saliency)
    return saliency_overlayed,heat_map_jet_pil
