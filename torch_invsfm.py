'''
    Last Edit Date  : 22.08.09
    Editor          : Chanhyuk Yun
'''
import os
import numpy as np
from PIL import Image
import argparse
import load_data as ld
import torch
# import torchvision
import models_torch as models
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# tf.disable_v2_behavior()
def load_data(db_dir, pts3d_dir, cam_dir, img_dir, num_samples, scale_size, crop_size):
    # Load point cloud with per-point sift descriptors and rgb features from
    # colmap database and points3D.bin file from colmap sparse reconstruction
    print('Loading point cloud...')
    pcl_xyz, pcl_rgb, pcl_sift = ld.load_points_colmap(db_dir,pts3d_dir)
    print('Done!')

    # Load camera matrices and from images.bin and cameras.bin files from
    # colmap sparse reconstruction
    print('Loading cameras...')
    K,R,T,h,w,_ = ld.load_cameras_colmap(img_dir,cam_dir)
    print('Done!')

    # Generate projections
    proj_depth = []
    proj_sift = [] 
    proj_rgb = []
    for i in range(len(K))[::(len(K)//num_samples)]:
        proj_mat = K[i].dot(np.hstack((R[i],T[i])))
        pdepth, prgb, psift = ld.project_points(pcl_xyz, pcl_rgb, pcl_sift,
                                                proj_mat, h[i], w[i], scale_size, crop_size)    
        proj_depth.append((pdepth)[None,...])
        proj_sift.append((psift)[None,...])
        proj_rgb.append((prgb)[None,...])
        
    proj_depth = np.vstack(proj_depth)
    proj_sift = np.vstack(proj_sift)
    proj_rgb = np.vstack(proj_rgb)

    # Pytorch input
    pdepth = torch.tensor(proj_depth, dtype=torch.float32)
    psift = torch.tensor(proj_sift, dtype=torch.uint8)
    prgb = torch.tensor(proj_rgb, dtype=torch.uint8)

    pdepth = pdepth.to(device)
    psift = psift.to(device)
    prgb = prgb.to(device)

    psift = psift.to(torch.float32)
    prgb = prgb.to(torch.float32)
    valid = torch.greater(pdepth, 0.)

    return pdepth, psift, prgb, valid

def load_vinp(db_dir, pts3d_dir, cam_dir, img_dir, prm):
    input_attr = prm.input_attr
    pdepth, psift, prgb, valid = load_data(db_dir, pts3d_dir, cam_dir, img_dir, prm.num_samples, prm.scale_size, prm.crop_size)
    # set up visibnet
    if input_attr=='depth':
        vinp = pdepth
        inp_ch = 1
    elif input_attr=='depth_rgb':
        vinp = torch.concat((pdepth, prgb/127.5-1.), dim=3)
        inp_ch = 4
    elif input_attr=='depth_sift':
        vinp = torch.concat((pdepth, psift/127.5-1.), dim=3)
        inp_ch = 129
    elif input_attr=='depth_sift_rgb':
        vinp = torch.concat((pdepth, psift/127.5-1., prgb/127.5-1.), dim=3)
        inp_ch = 132

    vinp_ch = inp_ch
    valid = valid.permute(0,3,1,2)
    vinp = vinp.permute(0,3,1,2)

    return vinp[:prm.num_samples], vinp_ch, valid[:prm.num_samples], pdepth[:prm.num_samples], psift[:prm.num_samples], prgb[:prm.num_samples]

def set_vnet(vinp_ch, vwts_dir):
    print('Loading VisibNet...')
    vnet = models.VisibNet(vinp_ch)

    d = np.load(vwts_dir)
    weights = [val[1] for val in d.items()]
    wts = []
    bias = []
    mn = []
    var = []
    for i in range(int(len(weights)/4)):
        wts.append(weights[4*i])
        bias.append(weights[4*i+3])
        mn.append(weights[4*i+1])
        var.append(weights[4*i+2])
    wts.append(weights[int(len(weights)/4)*4])
    bias.append(weights[int(len(weights)/4)*4+1])
    #---------------------------------------------------------------------
    # Parameter transfer from tensorflow to pytorch
    #---------------------------------------------------------------------
    vnet.down1[0].weight = nn.Parameter(torch.from_numpy(wts[0]).to(torch.float32).permute((3,2,0,1)))
    vnet.down2[0].weight = nn.Parameter(torch.from_numpy(wts[1]).to(torch.float32).permute((3,2,0,1)))
    vnet.down3[0].weight = nn.Parameter(torch.from_numpy(wts[2]).to(torch.float32).permute((3,2,0,1)))
    vnet.down4[0].weight = nn.Parameter(torch.from_numpy(wts[3]).to(torch.float32).permute((3,2,0,1)))
    vnet.down5[0].weight = nn.Parameter(torch.from_numpy(wts[4]).to(torch.float32).permute((3,2,0,1)))
    vnet.down6[0].weight = nn.Parameter(torch.from_numpy(wts[5]).to(torch.float32).permute((3,2,0,1)))

    vnet.down1[1].running_mean = (torch.from_numpy(mn[0]).to(torch.float32))
    vnet.down2[1].running_mean = (torch.from_numpy(mn[1]).to(torch.float32))
    vnet.down3[1].running_mean = (torch.from_numpy(mn[2]).to(torch.float32))
    vnet.down4[1].running_mean = (torch.from_numpy(mn[3]).to(torch.float32))
    vnet.down5[1].running_mean = (torch.from_numpy(mn[4]).to(torch.float32))
    vnet.down6[1].running_mean = (torch.from_numpy(mn[5]).to(torch.float32))

    vnet.down1[1].running_var = (torch.from_numpy(var[0]).to(torch.float32))
    vnet.down2[1].running_var = (torch.from_numpy(var[1]).to(torch.float32))
    vnet.down3[1].running_var = (torch.from_numpy(var[2]).to(torch.float32))
    vnet.down4[1].running_var = (torch.from_numpy(var[3]).to(torch.float32))
    vnet.down5[1].running_var = (torch.from_numpy(var[4]).to(torch.float32))
    vnet.down6[1].running_var = (torch.from_numpy(var[5]).to(torch.float32))

    vnet.down1[1].bias = nn.Parameter(torch.from_numpy(bias[0]).to(torch.float32))
    vnet.down2[1].bias = nn.Parameter(torch.from_numpy(bias[1]).to(torch.float32))
    vnet.down3[1].bias = nn.Parameter(torch.from_numpy(bias[2]).to(torch.float32))
    vnet.down4[1].bias = nn.Parameter(torch.from_numpy(bias[3]).to(torch.float32))
    vnet.down5[1].bias = nn.Parameter(torch.from_numpy(bias[4]).to(torch.float32))
    vnet.down6[1].bias = nn.Parameter(torch.from_numpy(bias[5]).to(torch.float32))

    vnet.up1[0].weight = nn.Parameter(torch.from_numpy(wts[6]).to(torch.float32).permute((3,2,0,1)))
    vnet.up2[0].weight = nn.Parameter(torch.from_numpy(wts[7]).to(torch.float32).permute((3,2,0,1)))
    vnet.up3[0].weight = nn.Parameter(torch.from_numpy(wts[8]).to(torch.float32).permute((3,2,0,1)))
    vnet.up4[0].weight = nn.Parameter(torch.from_numpy(wts[9]).to(torch.float32).permute((3,2,0,1)))
    vnet.up5[0].weight = nn.Parameter(torch.from_numpy(wts[10]).to(torch.float32).permute((3,2,0,1)))
    vnet.up6[0].weight = nn.Parameter(torch.from_numpy(wts[11]).to(torch.float32).permute((3,2,0,1)))
    vnet.up7[0].weight = nn.Parameter(torch.from_numpy(wts[12]).to(torch.float32).permute((3,2,0,1)))
    vnet.up8[0].weight = nn.Parameter(torch.from_numpy(wts[13]).to(torch.float32).permute((3,2,0,1)))
    vnet.up9[0].weight = nn.Parameter(torch.from_numpy(wts[14]).to(torch.float32).permute((3,2,0,1)))
    vnet.up10[0].weight = nn.Parameter(torch.from_numpy(wts[15]).to(torch.float32).permute((3,2,0,1)))

    vnet.up1[1].running_mean = (torch.from_numpy(mn[6]).to(torch.float32))
    vnet.up2[1].running_mean = (torch.from_numpy(mn[7]).to(torch.float32))
    vnet.up3[1].running_mean = (torch.from_numpy(mn[8]).to(torch.float32))
    vnet.up4[1].running_mean = (torch.from_numpy(mn[9]).to(torch.float32))
    vnet.up5[1].running_mean = torch.from_numpy(mn[10]).to(torch.float32)
    vnet.up6[1].running_mean = torch.from_numpy(mn[11]).to(torch.float32)
    vnet.up7[1].running_mean = torch.from_numpy(mn[12]).to(torch.float32)
    vnet.up8[1].running_mean = torch.from_numpy(mn[13]).to(torch.float32)
    vnet.up9[1].running_mean = torch.from_numpy(mn[14]).to(torch.float32)

    vnet.up1[1].running_var = torch.from_numpy(var[6]).to(torch.float32)
    vnet.up2[1].running_var = torch.from_numpy(var[7]).to(torch.float32)
    vnet.up3[1].running_var = torch.from_numpy(var[8]).to(torch.float32)
    vnet.up4[1].running_var = torch.from_numpy(var[9]).to(torch.float32)
    vnet.up5[1].running_var = torch.from_numpy(var[10]).to(torch.float32)
    vnet.up6[1].running_var = torch.from_numpy(var[11]).to(torch.float32)
    vnet.up7[1].running_var = torch.from_numpy(var[12]).to(torch.float32)
    vnet.up8[1].running_var = torch.from_numpy(var[13]).to(torch.float32)
    vnet.up9[1].running_var = torch.from_numpy(var[14]).to(torch.float32)

    vnet.up1[1].bias = nn.Parameter(torch.from_numpy(bias[6]).to(torch.float32))
    vnet.up2[1].bias = nn.Parameter(torch.from_numpy(bias[7]).to(torch.float32))
    vnet.up3[1].bias = nn.Parameter(torch.from_numpy(bias[8]).to(torch.float32))
    vnet.up4[1].bias = nn.Parameter(torch.from_numpy(bias[9]).to(torch.float32))
    vnet.up5[1].bias = nn.Parameter(torch.from_numpy(bias[10]).to(torch.float32))
    vnet.up6[1].bias = nn.Parameter(torch.from_numpy(bias[11]).to(torch.float32))
    vnet.up7[1].bias = nn.Parameter(torch.from_numpy(bias[12]).to(torch.float32))
    vnet.up8[1].bias = nn.Parameter(torch.from_numpy(bias[13]).to(torch.float32))
    vnet.up9[1].bias = nn.Parameter(torch.from_numpy(bias[14]).to(torch.float32))
    vnet.up10[0].bias = nn.Parameter(torch.from_numpy(bias[15]).to(torch.float32))

    vnet = vnet.to(device)
    return vnet

def set_cnet(cinp_ch, cwts_dir):
    print('Loading CoarseNet...')
    # 2. CoarseNet
    cnet = models.CoarseNet(cinp_ch)

    d = np.load(cwts_dir)
    weights = [val[1] for val in d.items()]
    wts = []
    bias = []
    mn = []
    var = []
    for i in range(int(len(weights)/4)):
        wts.append(weights[4*i])
        bias.append(weights[4*i+3])
        mn.append(weights[4*i+1])
        var.append(weights[4*i+2])
    wts.append(weights[int(len(weights)/4)*4])
    bias.append(weights[int(len(weights)/4)*4+1])
    #---------------------------------------------------------------------
    # Parameter transfer from tensorflow to pytorch
    #---------------------------------------------------------------------
    cnet.down1[0].weight = nn.Parameter(torch.from_numpy(wts[0]).to(torch.float32).permute((3,2,0,1)))
    cnet.down2[0].weight = nn.Parameter(torch.from_numpy(wts[1]).to(torch.float32).permute((3,2,0,1)))
    cnet.down3[0].weight = nn.Parameter(torch.from_numpy(wts[2]).to(torch.float32).permute((3,2,0,1)))
    cnet.down4[0].weight = nn.Parameter(torch.from_numpy(wts[3]).to(torch.float32).permute((3,2,0,1)))
    cnet.down5[0].weight = nn.Parameter(torch.from_numpy(wts[4]).to(torch.float32).permute((3,2,0,1)))
    cnet.down6[0].weight = nn.Parameter(torch.from_numpy(wts[5]).to(torch.float32).permute((3,2,0,1)))

    cnet.down1[1].running_mean = (torch.from_numpy(mn[0]).to(torch.float32))
    cnet.down2[1].running_mean = (torch.from_numpy(mn[1]).to(torch.float32))
    cnet.down3[1].running_mean = (torch.from_numpy(mn[2]).to(torch.float32))
    cnet.down4[1].running_mean = (torch.from_numpy(mn[3]).to(torch.float32))
    cnet.down5[1].running_mean = (torch.from_numpy(mn[4]).to(torch.float32))
    cnet.down6[1].running_mean = (torch.from_numpy(mn[5]).to(torch.float32))

    cnet.down1[1].running_var = (torch.from_numpy(var[0]).to(torch.float32))
    cnet.down2[1].running_var = (torch.from_numpy(var[1]).to(torch.float32))
    cnet.down3[1].running_var = (torch.from_numpy(var[2]).to(torch.float32))
    cnet.down4[1].running_var = (torch.from_numpy(var[3]).to(torch.float32))
    cnet.down5[1].running_var = (torch.from_numpy(var[4]).to(torch.float32))
    cnet.down6[1].running_var = (torch.from_numpy(var[5]).to(torch.float32))

    cnet.down1[1].bias = nn.Parameter(torch.from_numpy(bias[0]).to(torch.float32))
    cnet.down2[1].bias = nn.Parameter(torch.from_numpy(bias[1]).to(torch.float32))
    cnet.down3[1].bias = nn.Parameter(torch.from_numpy(bias[2]).to(torch.float32))
    cnet.down4[1].bias = nn.Parameter(torch.from_numpy(bias[3]).to(torch.float32))
    cnet.down5[1].bias = nn.Parameter(torch.from_numpy(bias[4]).to(torch.float32))
    cnet.down6[1].bias = nn.Parameter(torch.from_numpy(bias[5]).to(torch.float32))

    cnet.up1[0].weight = nn.Parameter(torch.from_numpy(wts[6]).to(torch.float32).permute((3,2,0,1)))
    cnet.up2[0].weight = nn.Parameter(torch.from_numpy(wts[7]).to(torch.float32).permute((3,2,0,1)))
    cnet.up3[0].weight = nn.Parameter(torch.from_numpy(wts[8]).to(torch.float32).permute((3,2,0,1)))
    cnet.up4[0].weight = nn.Parameter(torch.from_numpy(wts[9]).to(torch.float32).permute((3,2,0,1)))
    cnet.up5[0].weight = nn.Parameter(torch.from_numpy(wts[10]).to(torch.float32).permute((3,2,0,1)))
    cnet.up6[0].weight = nn.Parameter(torch.from_numpy(wts[11]).to(torch.float32).permute((3,2,0,1)))
    cnet.up7[0].weight = nn.Parameter(torch.from_numpy(wts[12]).to(torch.float32).permute((3,2,0,1)))
    cnet.up8[0].weight = nn.Parameter(torch.from_numpy(wts[13]).to(torch.float32).permute((3,2,0,1)))
    cnet.up9[0].weight = nn.Parameter(torch.from_numpy(wts[14]).to(torch.float32).permute((3,2,0,1)))
    cnet.up10[0].weight = nn.Parameter(torch.from_numpy(wts[15]).to(torch.float32).permute((3,2,0,1)))

    cnet.up1[1].running_mean = (torch.from_numpy(mn[6]).to(torch.float32))
    cnet.up2[1].running_mean = (torch.from_numpy(mn[7]).to(torch.float32))
    cnet.up3[1].running_mean = (torch.from_numpy(mn[8]).to(torch.float32))
    cnet.up4[1].running_mean = (torch.from_numpy(mn[9]).to(torch.float32))
    cnet.up5[1].running_mean = torch.from_numpy(mn[10]).to(torch.float32)
    cnet.up6[1].running_mean = torch.from_numpy(mn[11]).to(torch.float32)
    cnet.up7[1].running_mean = torch.from_numpy(mn[12]).to(torch.float32)
    cnet.up8[1].running_mean = torch.from_numpy(mn[13]).to(torch.float32)
    cnet.up9[1].running_mean = torch.from_numpy(mn[14]).to(torch.float32)

    cnet.up1[1].running_var = torch.from_numpy(var[6]).to(torch.float32)
    cnet.up2[1].running_var = torch.from_numpy(var[7]).to(torch.float32)
    cnet.up3[1].running_var = torch.from_numpy(var[8]).to(torch.float32)
    cnet.up4[1].running_var = torch.from_numpy(var[9]).to(torch.float32)
    cnet.up5[1].running_var = torch.from_numpy(var[10]).to(torch.float32)
    cnet.up6[1].running_var = torch.from_numpy(var[11]).to(torch.float32)
    cnet.up7[1].running_var = torch.from_numpy(var[12]).to(torch.float32)
    cnet.up8[1].running_var = torch.from_numpy(var[13]).to(torch.float32)
    cnet.up9[1].running_var = torch.from_numpy(var[14]).to(torch.float32)

    cnet.up1[1].bias = nn.Parameter(torch.from_numpy(bias[6]).to(torch.float32))
    cnet.up2[1].bias = nn.Parameter(torch.from_numpy(bias[7]).to(torch.float32))
    cnet.up3[1].bias = nn.Parameter(torch.from_numpy(bias[8]).to(torch.float32))
    cnet.up4[1].bias = nn.Parameter(torch.from_numpy(bias[9]).to(torch.float32))
    cnet.up5[1].bias = nn.Parameter(torch.from_numpy(bias[10]).to(torch.float32))
    cnet.up6[1].bias = nn.Parameter(torch.from_numpy(bias[11]).to(torch.float32))
    cnet.up7[1].bias = nn.Parameter(torch.from_numpy(bias[12]).to(torch.float32))
    cnet.up8[1].bias = nn.Parameter(torch.from_numpy(bias[13]).to(torch.float32))
    cnet.up9[1].bias = nn.Parameter(torch.from_numpy(bias[14]).to(torch.float32))
    cnet.up10[0].bias = nn.Parameter(torch.from_numpy(bias[15]).to(torch.float32))

    cnet = cnet.to(device)
    return cnet

def set_rnet(rinp_ch, rwts_dir, input_attr):
    print('Loading RefineNet...')
    d = np.load(rwts_dir)

    weights = [val[1] for val in d.items()]
    wts = []
    bias = []
    for i in range(int(len(weights)/2)):
        wts.append(weights[i*2])
        bias.append(weights[i*2+1])
    #---------------------------------------------------------------------
    # Parameter transfer from tensorflow to pytorch
    #---------------------------------------------------------------------
    rnet = models.RefineNet(rinp_ch)

    rnet.down1[0].weight = nn.Parameter(torch.from_numpy(wts[0]).to(torch.float32).permute((3,2,0,1)))
    rnet.down2[0].weight = nn.Parameter(torch.from_numpy(wts[1]).to(torch.float32).permute((3,2,0,1)))
    rnet.down3[0].weight = nn.Parameter(torch.from_numpy(wts[2]).to(torch.float32).permute((3,2,0,1)))
    rnet.down4[0].weight = nn.Parameter(torch.from_numpy(wts[3]).to(torch.float32).permute((3,2,0,1)))
    rnet.down5[0].weight = nn.Parameter(torch.from_numpy(wts[4]).to(torch.float32).permute((3,2,0,1)))
    rnet.down6[0].weight = nn.Parameter(torch.from_numpy(wts[5]).to(torch.float32).permute((3,2,0,1)))

    rnet.down1[1].bias = nn.Parameter(torch.from_numpy(bias[0]).to(torch.float32))
    rnet.down2[1].bias = nn.Parameter(torch.from_numpy(bias[1]).to(torch.float32))
    rnet.down3[1].bias = nn.Parameter(torch.from_numpy(bias[2]).to(torch.float32))
    rnet.down4[1].bias = nn.Parameter(torch.from_numpy(bias[3]).to(torch.float32))
    rnet.down5[1].bias = nn.Parameter(torch.from_numpy(bias[4]).to(torch.float32))
    rnet.down6[1].bias = nn.Parameter(torch.from_numpy(bias[5]).to(torch.float32))

    rnet.up1[0].weight = nn.Parameter(torch.from_numpy(wts[6]).to(torch.float32).permute((3,2,0,1)))
    rnet.up2[0].weight = nn.Parameter(torch.from_numpy(wts[7]).to(torch.float32).permute((3,2,0,1)))
    rnet.up3[0].weight = nn.Parameter(torch.from_numpy(wts[8]).to(torch.float32).permute((3,2,0,1)))
    rnet.up4[0].weight = nn.Parameter(torch.from_numpy(wts[9]).to(torch.float32).permute((3,2,0,1)))
    rnet.up5[0].weight = nn.Parameter(torch.from_numpy(wts[10]).to(torch.float32).permute((3,2,0,1)))
    rnet.up6[0].weight = nn.Parameter(torch.from_numpy(wts[11]).to(torch.float32).permute((3,2,0,1)))
    rnet.up7[0].weight = nn.Parameter(torch.from_numpy(wts[12]).to(torch.float32).permute((3,2,0,1)))
    rnet.up8[0].weight = nn.Parameter(torch.from_numpy(wts[13]).to(torch.float32).permute((3,2,0,1)))
    rnet.up9[0].weight = nn.Parameter(torch.from_numpy(wts[14]).to(torch.float32).permute((3,2,0,1)))
    rnet.up10[0].weight = nn.Parameter(torch.from_numpy(wts[15]).to(torch.float32).permute((3,2,0,1)))

    rnet.up1[1].bias = nn.Parameter(torch.from_numpy(bias[6]).to(torch.float32))
    rnet.up2[1].bias = nn.Parameter(torch.from_numpy(bias[7]).to(torch.float32))
    rnet.up3[1].bias = nn.Parameter(torch.from_numpy(bias[8]).to(torch.float32))
    rnet.up4[1].bias = nn.Parameter(torch.from_numpy(bias[9]).to(torch.float32))
    rnet.up5[1].bias = nn.Parameter(torch.from_numpy(bias[10]).to(torch.float32))
    rnet.up6[1].bias = nn.Parameter(torch.from_numpy(bias[11]).to(torch.float32))
    rnet.up7[1].bias = nn.Parameter(torch.from_numpy(bias[12]).to(torch.float32))
    rnet.up8[1].bias = nn.Parameter(torch.from_numpy(bias[13]).to(torch.float32))
    rnet.up9[1].bias = nn.Parameter(torch.from_numpy(bias[14]).to(torch.float32))
    rnet.up10[0].bias = nn.Parameter(torch.from_numpy(bias[15]).to(torch.float32))

    rnet = rnet.to(device)
    return rnet

def eval_vnet(vinp, vnet, valid):
    with torch.no_grad():
        vnet.eval()
        vout = vnet(vinp)
    vpred = torch.logical_and(torch.gt(vout,.5),valid)
    vpredf = vpred.to(torch.float32)*0.+1.
    vpred = vpred.cpu()
    vpredf = vpredf.permute((0,2,3,1))
    return vpred, vpredf
def eval_cnet(cinp, cnet):
    with torch.no_grad():
        cnet.eval()
        cpred = cnet(cinp)
    cpred_ = (cpred+1.)*127.5
    cpred_ = cpred_.cpu()
    return cpred_, cpred
def eval_rnet(rinp, rnet):
    with torch.no_grad():
        rnet.eval()
        rpred = rnet(rinp)
    rpred = (rpred+1.)*127.5
    rpred_ = rpred.cpu()
    return rpred_

def v_eval(inp_ch, vinp, vwts_dir):
    vnet = set_vnet(inp_ch, vwts_dir)
    vpred_ = torch.zeros([1,1,vinp.shape[2],vinp.shape[3]])
    vpredf_ = torch.zeros([1,vinp.shape[2],vinp.shape[3],1]).to(device)
    for i in range(vinp.shape[0]):
        vpred, vpredf = eval_vnet(vinp[i:i+1], vnet, valid[i:i+1])
        vpred_ = torch.cat([vpred_, vpred], dim=0)
        vpredf_ = torch.cat([vpredf_, vpredf], dim=0)
    vpred = vpred_[1:]
    vpredf = vpredf_[1:].to(device)
    return vpred, vpredf
def c_eval(inp_ch, cinp, cwts_dir):
    cnet = set_cnet(inp_ch, cwts_dir)
    cpred_ = torch.zeros([1,3,cinp.shape[2],cinp.shape[3]])
    cpredf_ = torch.zeros([1,3,cinp.shape[2],cinp.shape[3]]).to(device)
    for i in range(cinp.shape[0]):
        cpred, cpredf = eval_cnet(cinp[i:i+1], cnet)
        cpred_ = torch.cat([cpred_, cpred], dim=0)
        cpredf_ = torch.cat([cpredf_, cpredf], dim=0)
    cpred = cpred_[1:]
    cpredf = cpredf_[1:]
    return cpred, cpredf
def r_eval(inp_ch, rinp, rwts_dir, prm):
    rnet = set_rnet(inp_ch+3, rwts_dir, prm.input_attr)
    rpred_ = torch.zeros([1,3,rinp.shape[2],rinp.shape[3]])
    for i in range(rinp.shape[0]):
        rpred = eval_rnet(rinp[i:i+1], rnet)
        rpred_ = torch.cat([rpred_, rpred], dim=0)
    rpred = rpred_[1:]
    return rpred

#-------------------------------------------------------------------------------
# Codes from InvSfM(tf.v1) for data loading
#-------------------------------------------------------------------------------
################################################################################
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Configure')
    parser.add_argument("--input_attr", type=str, default='depth_sift_rgb',
                        choices=['depth','depth_sift','depth_rgb','depth_sift_rgb'],
                        help="%(type)s: Per-point attributes to inlcude in input tensor (default: %(default)s)")
    parser.add_argument("--pct_3D_points", type=float, default=100., choices=[20,60,100],
                        help="%(type)s: Percent of available 3D points to include in input tensor (default: %(default)s)")
    parser.add_argument("--dataset", type=str, default='nyu', choices=['nyu','medadepth'],
                        help="%(type)s: Dataset to use for demo (default: %(default)s)")
    parser.add_argument("--crop_size", type=int, default=512, choices=[256,512],
                        help="%(type)s: Size to crop images to (default: %(default)s)")
    parser.add_argument("--scale_size", type=int, default=512, choices=[256,394,512],
                        help="%(type)s: Size to scale images to before crop (default: %(default)s)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="%(type)s: Number of samples to process/visualize (default: %(default)s)")
    prm = parser.parse_args()

    if prm.scale_size < prm.crop_size: parser.error("SCALE_SIZE must be >= CROP_SIZE")
    if prm.num_samples <= 0: parser.error("NUM_SAMPLES must be > 0")

    prm_str = 'Parameters:\n'+'\n'.join(['{} {}'.format(k.upper(),v) for k,v in vars(prm).items()])
    print(prm_str+'\n')

    # set paths for model wts
    vnet_wts_fp = 'wts/pretrained/{}/visibnet.model.npz'.format(prm.input_attr)
    cnet_wts_fp = 'wts/pretrained/{}/coarsenet.model.npz'.format(prm.input_attr)
    rnet_wts_fp = 'wts/pretrained/{}/refinenet.model.npz'.format(prm.input_attr)

    # set paths for colmap files
    scene = 'nyu_0000'
    cmap_database_fp = 'data/{}/database.db'.format(scene)
    cmap_points3D_fp = 'data/{}/points3D.txt'.format(scene)
    cmap_cameras_fp = 'data/{}/cameras.txt'.format(scene)
    cmap_images_fp = 'data/{}/images.txt'.format(scene)
    ################################################################################


    ################################################################################

    #-------------------------------------------------------------------------------
    # Codes for pytorch : preparing data
    #-------------------------------------------------------------------------------
    '''
    Data in tensorflow  : NHWC
    Data in pytorch     : NCHW
    data_torch = np.transpose( data_tf.numpy() , (0,3,1,2) )
    '''
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device : {}\n'.format(device))

    #-------------------------------------------------------------------------------
    # Codes for pytorch : Get pretrained weights trained in tensorflow for pytorch
    #-------------------------------------------------------------------------------
    # 1. VisibNet
    vinp, inp_ch, valid, pdepth, psift, prgb = load_vinp(cmap_database_fp, cmap_points3D_fp, cmap_cameras_fp, cmap_images_fp, prm)
    vpred, vpredf = v_eval(inp_ch, vinp, vnet_wts_fp)

    # 2. CoarseNet
    # set up coarsenet 
    if prm.input_attr=='depth':
        cinp = pdepth*vpredf
    elif prm.input_attr=='depth_rgb':
        cinp = torch.concat((pdepth*vpredf, prgb*vpredf/127.5-1.), dim=3)
    elif prm.input_attr=='depth_sift':
        cinp = torch.concat((pdepth*vpredf, psift*vpredf/127.5-1.), dim=3)
    elif prm.input_attr=='depth_sift_rgb':
        cinp = torch.concat((pdepth*vpredf, psift*vpredf/127.5-1., prgb*vpredf/127.5-1.), dim=3)
    cinp = cinp.permute((0,3,1,2))
    cpred, cpredf = c_eval(inp_ch, cinp, cnet_wts_fp)

    # 3. RefineNet
    # set up refinenet
    rinp = torch.concat((cpredf, cinp), dim=1)
    rpred = r_eval(inp_ch, rinp, rnet_wts_fp, prm)

    # # Save as images
    vpred_img = []
    cpred_img = []
    rpred_img = []
    for i in range(prm.num_samples):
        vpred_img.append(vpred[i])
        cpred_img.append(cpred[i])
        rpred_img.append(rpred[i])

    vpred_img = np.vstack(vpred_img)
    cpred_img = np.hstack(cpred_img)
    rpred_img = np.hstack(rpred_img)
    vpred = np.vstack(vpred_img)

    valid = np.vstack(valid.cpu())
    valid = np.vstack(valid)
    zero = np.zeros(valid.shape, dtype=bool)
    # Prepare an empty image; vpred_img : HWC
    vpred_img = np.ones([vpred.shape[0],prm.crop_size,3])*255.
    # set all valid pixels 0
    vpred_img[np.dstack((valid,valid,valid))] = 0.
    # set valid but invisible pixels red
    vpred_img[np.dstack((np.logical_and(valid,np.logical_not(vpred)),zero,zero))] = 255.
    # set valid and visible pixels blue
    vpred_img[np.dstack((zero,zero,np.logical_and(valid,vpred)))] = 255.

    cpred_img = np.transpose(cpred_img, (1,2,0))
    rpred_img = np.transpose(rpred_img, (1,2,0))

    mntg = np.hstack((vpred_img.astype(np.uint8),
                    cpred_img.astype(np.uint8),
                    rpred_img.astype(np.uint8)))
    # save images
    img = Image.fromarray(mntg.astype(np.uint8))
    os.makedirs('viz/invSfM/', exist_ok=True)
    fp = 'viz/invSfM/{}_{}_torch.png'.format(scene, prm.input_attr)
    print('Saving visualization to {}...'.format(fp))
    img.save(fp)
    print('Done!')

