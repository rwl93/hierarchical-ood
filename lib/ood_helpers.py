import numpy as np
import sys
import os
import random
import sklearn.covariance

import torch
import torch.nn as nn
import torch.utils.data as utilsdata
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as tvdatasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_model_hook(activations_model, full_model, hook, device):
    """
    Hook the intermediate layer activations of a sample model
    """
    handle = None
    if "sample" in activations_model:
        parts = activations_model.split("_")
        layer_num = int(parts[-1])
        if layer_num == 1:
            handle = full_model.features[0].register_forward_hook(hook)
        elif layer_num == 2:
            handle = full_model.features[3].register_forward_hook(hook)
        elif layer_num == 3:
            handle = full_model.features[6].register_forward_hook(hook)
        elif layer_num == 4:
            handle = full_model.features[9].register_forward_hook(hook)
        elif layer_num == 5:
            handle = full_model.classifier[1].register_forward_hook(hook)
        elif layer_num == 6:
            handle = full_model.classifier[4].register_forward_hook(hook)
        elif layer_num == 7:
            handle = full_model.classifier[7].register_forward_hook(hook)
        elif layer_num == 8:
            handle = full_model.classifier[10].register_forward_hook(hook)
        else:
            print("cannot set hook into sample model")
            exit()
    elif ("resnet" in activations_model):
        parts = activations_model.split("_")
        cfg = [int(x) for x in parts[1:]]
        if len(cfg) == 5:
            handle = full_model.fc.register_forward_hook(hook)
        elif len(cfg) == 4:
            #handle = full_model.layer4[cfg[-1]-1].register_forward_hook(hook) # Post-ReLU
            handle = full_model.layer4[cfg[-1]-1].identity.register_forward_hook(hook) # Pre-ReLU
        elif len(cfg) == 3:
            #handle = full_model.layer3[cfg[-1]-1].register_forward_hook(hook)
            handle = full_model.layer3[cfg[-1]-1].identity.register_forward_hook(hook)
        elif len(cfg) == 2:
            #handle = full_model.layer2[cfg[-1]-1].register_forward_hook(hook)
            handle = full_model.layer2[cfg[-1]-1].identity.register_forward_hook(hook)
        elif len(cfg) == 1:
            #handle = full_model.layer1[cfg[-1]-1].register_forward_hook(hook)
            handle = full_model.layer1[cfg[-1]-1].identity.register_forward_hook(hook)
        else:
            exit("Invalid resnet hook")
    else:
        print("UNDEFINED ACTIVATIONS MODEL")
        exit()
    return handle


def get_ood_dataloader(dsetname, dsize=256, cropsize=224, batch_size=32, shuf=False):
    """
    Create the out-of-distribution dataloader
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(dsize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        normalize,
        ])
    returnloader = None
    if dsetname == "random":
        returnloader = torch.utils.data.DataLoader(
            tvdatasets.FakeData(size=1000, image_size=(3,dsize,dsize),
                                transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=shuf, num_workers=2,
        )
    elif dsetname == "mnist":
        returnloader = torch.utils.data.DataLoader(
            tvdatasets.MNIST("./data", train=False, download=True, transform=transform_test),
            batch_size=batch_size, shuffle=shuf, num_workers=2,
        )
    elif dsetname == "fmnist":
        returnloader = torch.utils.data.DataLoader(
            tvdatasets.FashionMNIST("./data",train=False, download=True, transform=transform_test),
            batch_size=batch_size, shuffle=shuf, num_workers=2,
        )
    elif dsetname == "c10":
        returnloader = torch.utils.data.DataLoader(
            tvdatasets.CIFAR10("./data",train=False, download=True, transform=transform_test),
            batch_size=batch_size, shuffle=shuf, num_workers=2,
        )
    elif dsetname == "svhn":
        returnloader = torch.utils.data.DataLoader(
            tvdatasets.SVHN("./data",split='test', download=True, transform=transform_test),
            batch_size=batch_size, shuffle=shuf, num_workers=2,
        )
    else:
        exit("Unknown OOD dataset")

    return returnloader

def generate_ood_scores_ODIN_and_BASELINE(base_model, loader, T, IPP=False, eps=0.):
    """
    Generate OOD scores with the ODIN detector and the baseline detector.

    Parameters
    ----------
    model : nn.Module
        base classifier network to test
    loader : torch.utils.data.DataLoader
        dataloader to compute ood scores over
    T : int
        temperature level to divide logits by
    IPP : bool
        whether or not to do the input preprocessing step
    eps : float
        epsilon of perturbation if IPP is set
    """
    base_model.eval()
    baseline_scores = []
    odin_scores = []
    odin_ipp_scores = []

    if IPP:
        for dat, lbl in loader:
            dat = dat.to(device); lbl = lbl.to(device)
            dat.requires_grad = True

            # Initial forward pass
            initial_outs = base_model(dat)

            # Get predicted labels
            _,initial_preds = initial_outs.max(1)

            # Compute gradient w.r.t. data using predicted labels
            tmploss = F.cross_entropy(initial_outs,initial_preds)
            base_model.zero_grad()
            tmploss.backward()
            data_grad = dat.grad.data.detach()
            with torch.no_grad():
                # Perturb data in gradient direction
                pert_dat = torch.clamp(dat - eps*data_grad.sign(), 0., 1.)
                # Recompute prediction scores
                new_outs = base_model(pert_dat)
                # Save new confidence values as ODIN scores
                baseline_scores.extend(
                    list(torch.max(
                        F.softmax(initial_outs.data.clone().detach(), dim=1),
                        dim=1)[0].cpu().numpy())
                )
                odin_scores.extend(
                    list(torch.max(
                        F.softmax(initial_outs.data.clone().detach()/T, dim=1),
                        dim=1)[0].cpu().numpy())
                )
                odin_ipp_scores.extend(
                    list(torch.max(
                        F.softmax(new_outs.data.clone().detach()/T, dim=1),
                        dim=1)[0].cpu().numpy())
                )
    else:
        with torch.no_grad():
            for dat, _ in loader:
                dat = dat.to(device)
                initial_outs = base_model(dat)
                baseline_scores.extend(
                    list(torch.max(
                        F.softmax(initial_outs.data.clone().detach(), dim=1),
                        dim=1)[0].cpu().numpy())
                )
                odin_scores.extend(
                    list(torch.max(
                        F.softmax(initial_outs.data.clone().detach()/T, dim=1),
                        dim=1)[0].cpu().numpy())
                )
    return baseline_scores, odin_scores, odin_ipp_scores


############################################################################################################
#### MAHALANOBIS OOD Stuff

def compute_empirical_means_and_precision(net, activations, layers, loader, num_classes):
    layer_reps = {}
    layer_lbls = {}
    layer_class_means = {}
    layer_precision_matrix = {}

    with torch.no_grad():
        
        ### COMPUTE SAMPLE MATRIX
        print("Computing sample matrix...")
        batch_idx = -1
        for x,y,_ in loader:
            batch_idx += 1
            x = x.to(device); y = y.to(device)
            # Forward pass data through model to prime the activations dictionary
            net(x)
            # Treat each layer separately
            for lyr in layers:
                # Compute feature representation and save it into layer_reps
                act = F.relu(activations[lyr]) # N,C,H,W
                if len(act.shape) == 4: # If activations are from a conv layer must average out spatial dims to get [N,C] shaped representation
                    act = act.view(act.shape[0],act.shape[1], -1) # N,C,H*W
                    sample_mean = torch.mean(act, dim=2) # N,C
                else: # If activations are from a FC layer, dont have to do anything because already [N,C]
                    sample_mean = act
                # Append this representation onto layer_reps
                if lyr not in layer_reps.keys():
                    layer_reps[lyr] = sample_mean.data
                    layer_lbls[lyr] = y.data
                else:
                    layer_reps[lyr] = torch.cat((layer_reps[lyr],sample_mean.data), 0)
                    layer_lbls[lyr] = torch.cat((layer_lbls[lyr],y.data))

        ### COMPUTE CLASS MEANS FOR EACH LAYER
        print("Computing class means...")
        for lyr in layers:
            class_means = []
            for cls in range(num_classes):
                cls_dat = layer_reps[lyr][layer_lbls[lyr] == cls]
                class_means.append(torch.mean(cls_dat,0).unsqueeze_(0))
            class_means = torch.cat(class_means,0)
            layer_class_means[lyr] = class_means.data   
    
        ### COMPUTE PRECISION MATRIX FOR EACH LAYER
        print("Computing precision matrices...")
        for lyr in layers:
            centered_data = None
            for cls in range(num_classes):
                for jj in range(layer_reps[lyr][layer_lbls[lyr] == cls].shape[0]):
                    meandiff = layer_reps[lyr][layer_lbls[lyr]==cls][jj] - layer_class_means[lyr][cls]
                    meandiff.unsqueeze_(0)
                    if centered_data is None:
                        centered_data = meandiff.data
                    else:
                        centered_data = torch.cat((centered_data,meandiff.data),0)
            group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
            layer_precision_matrix[lyr] = torch.from_numpy(group_lasso.precision_).float().to(device)

        for lyr in layers:
            print("Layer {}. Reps.size: {}. Lbls.size: {}. ClassMean.size: {}. PrecisionMat.size: {}".format(lyr,layer_reps[lyr].shape,layer_lbls[lyr].shape,layer_class_means[lyr].shape, layer_precision_matrix[lyr].shape ))
            
    return layer_class_means, layer_precision_matrix


def compute_mahalanobis_score_batch_preIPP(feat, mu, prec):
    num_classes = mu.shape[0]
    class_scores = torch.zeros((feat.shape[0],num_classes)).float().to(device)
    for cls in range(num_classes):
        zc_tensor = feat - mu[cls].unsqueeze_(0)
        Mx_PT = -0.5*torch.matmul(torch.matmul(zc_tensor, prec), zc_tensor.t()).diag()
        class_scores[:,cls] = Mx_PT
    class_scores = torch.max(class_scores,dim=1)[0]
    return class_scores # Shape = [N] = the mahalanobis score for each element of the batch

def compute_mahalanobis_score_batch(feat, mu, prec):
    with torch.no_grad():
        num_classes = mu.shape[0]
        class_scores = torch.zeros((feat.shape[0],num_classes)).float().to(device)
        for cls in range(num_classes):
            zc_tensor = feat - mu[cls].unsqueeze_(0)
            Mx_PT = -0.5*torch.matmul(torch.matmul(zc_tensor, prec), zc_tensor.t()).diag()
            class_scores[:,cls] = Mx_PT
        class_scores = torch.max(class_scores,dim=1)[0]
        return class_scores # Shape = [N] = the mahalanobis score for each element of the batch

## Generate OOD scores with the MAHALANOBIS detector
#   - model = base classifier network to test
#   - loader = dataloader to compute ood scores over
#   - norm_mean/std = normalization constants of base_model
#   - IPP = boolean for whether or not to do the input preprocessing step
#   - eps = epsilon of perturbation if IPP is set
def generate_ood_scores_MAHALANOBIS(
        base_model, loader, activation_dictionary, layer_names_list,
        layer_means_dict, layer_precisions_dict, norm_mean, norm_std,
        IPP=False, eps=0.):
    mahalanobis_scores = []
    mahalanobis_ipp_scores = []

    if IPP:
        #for dat,lbl,_ in loader:
        for pkg in loader:
            dat = pkg[0]; lbl=pkg[1]
            dat = dat.to(device); lbl = lbl.to(device)
            dat.requires_grad = True
            # Initial forward pass & populate activation_dictionary
            initial_outs = base_model((dat-norm_mean)/norm_std)
            # Compute mahalanobis scores over each layer and sum
            lyr_scores = torch.zeros((initial_outs.shape[0])).to(device)
            for lyr in layer_names_list:
                # Compute feature representation and save it into layer_reps
                act = F.relu(activation_dictionary[lyr]) # N,C,H,W
                sample_feature = None
                if len(act.shape) == 4: # If activations are from a conv layer must average out spatial dims to get [N,C] shaped representation
                    act = act.view(act.shape[0],act.shape[1], -1) # N,C,H*W
                    sample_feature = torch.mean(act, dim=2) # N,C
                else: # Otherwise, FC layer so shape is already [N,C]
                    sample_feature = act
                lyr_scores += compute_mahalanobis_score_batch_preIPP(sample_feature, layer_means_dict[lyr].clone().detach(), layer_precisions_dict[lyr].clone().detach()) # should be N scores, one for each img in batch   
            #print(lyr_scores)
            #print(lyr_scores.shape)
            # Compute gradient of M(x) w.r.t input images
            base_model.zero_grad()
            lyr_scores.mean().backward()
            data_grad = dat.grad.data.detach()
            with torch.no_grad():
                # Perturb data in gradient direction
                pert_dat = torch.clamp(dat + eps*data_grad.sign(), 0., 1.)
                # Recompute predictions and re-populate activation dictionary
                new_outs = base_model((pert_dat-norm_mean)/norm_std)
                # Compute mahalanobis scores over each layer and sum
                final_lyr_scores = torch.zeros((initial_outs.shape[0])).to(device)
                for lyr in layer_names_list:
                    # Compute feature representation and save it into layer_reps
                    act = F.relu(activation_dictionary[lyr]) # N,C,H,W
                    sample_feature = None
                    if len(act.shape) == 4: # If activations are from a conv layer must average out spatial dims to get [N,C] shaped representation
                        act = act.view(act.shape[0],act.shape[1], -1) # N,C,H*W
                        sample_feature = torch.mean(act, dim=2) # N,C
                    else: # Otherwise, FC layer so shape is already [N,C]
                        sample_feature = act
                    final_lyr_scores += compute_mahalanobis_score_batch(
                        sample_feature, layer_means_dict[lyr].clone().detach(),
                        layer_precisions_dict[lyr].clone().detach()) # should be N scores, one for each img in batch    

                mahalanobis_scores.extend(list(lyr_scores.cpu().numpy()))   
                mahalanobis_ipp_scores.extend(list(final_lyr_scores.cpu().numpy())) 
    else:
        with torch.no_grad():
            #for dat,lbl,_ in loader:
            for pkg in loader:
                dat = pkg[0]
                dat = dat.to(device) #; lbl = lbl.to(device)
                # Initial forward pass & populate activation_dictionary
                initial_outs = base_model((dat-norm_mean)/norm_std)
                # Compute mahalanobis scores over each layer and sum
                lyr_scores = torch.zeros((initial_outs.shape[0])).to(device)
                for lyr in layer_names_list:
                    # Compute feature representation and save it into layer_reps
                    act = F.relu(activation_dictionary[lyr]) # N,C,H,W
                    sample_feature = None
                    if len(act.shape) == 4: # If activations are from a conv layer must average out spatial dims to get [N,C] shaped representation
                        act = act.view(act.shape[0],act.shape[1], -1) # N,C,H*W
                        sample_feature = torch.mean(act, dim=2) # N,C
                    else: # Otherwise, FC layer so shape is already [N,C]
                        sample_feature = act
                    lyr_scores += compute_mahalanobis_score_batch(sample_feature, layer_means_dict[lyr].clone().detach(), layer_precisions_dict[lyr].clone().detach()) # should be N scores, one for each img in batch  

                mahalanobis_scores.extend(list(lyr_scores.cpu().numpy()))   

    return mahalanobis_scores,mahalanobis_ipp_scores


## Generate OOD scores with the MAHALANOBIS detector
#   - model = base classifier network to test
#   - loader = dataloader to compute ood scores over
#   - norm_mean/std = normalization constants of base_model
#   - IPP = boolean for whether or not to do the input preprocessing step
#   - eps = epsilon of perturbation if IPP is set
def generate_ood_scores_MAHALANOBIS_Update(
        base_model, loader, activation_dictionary, layer_names_list,
        layer_means_dict, layer_precisions_dict,
        ):
    mahalanobis_scores = []

    with torch.no_grad():
        for pkg in loader:
            dat = pkg[0]
            dat = dat.to(device) #; lbl = lbl.to(device)
            # Initial forward pass & populate activation_dictionary
            initial_outs = base_model(dat)
            # Compute mahalanobis scores over each layer and sum
            lyr_scores = torch.zeros((initial_outs.shape[0])).to(device)
            for lyr in layer_names_list:
                # Compute feature representation and save it into layer_reps
                act = F.relu(activation_dictionary[lyr]) # N,C,H,W
                sample_feature = None
                if len(act.shape) == 4: # If activations are from a conv layer must average out spatial dims to get [N,C] shaped representation
                    act = act.view(act.shape[0],act.shape[1], -1) # N,C,H*W
                    sample_feature = torch.mean(act, dim=2) # N,C
                else: # Otherwise, FC layer so shape is already [N,C]
                    sample_feature = act
                lyr_scores += compute_mahalanobis_score_batch(sample_feature, layer_means_dict[lyr].clone().detach(), layer_precisions_dict[lyr].clone().detach()) # should be N scores, one for each img in batch  

            mahalanobis_scores.extend(list(lyr_scores.cpu().numpy()))   
    return mahalanobis_scores
