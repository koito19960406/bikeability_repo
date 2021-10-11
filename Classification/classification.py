#!/usr/bin/env python
# coding: utf-8

# In[29]:


# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)
get_ipython().system('pip install torch')
get_ipython().system('pip install torchvision')

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import numpy as np
import glob
import tqdm
import pandas
import boto3
import io
import matplotlib.pyplot as plt 
import pandas as pd



# In[63]:
def classify_image(model_folder,input_img_folder,output_folder,arch='resnet50',s3_upload_bucket=False):
    import torch
    from torch.autograd import Variable as V
    import torchvision.models as models
    from torchvision import transforms as trn
    from torch.nn import functional as F
    import os
    from PIL import Image
    import numpy as np
    import glob
    import tqdm
    import pandas
    import boto3
    import io
    import matplotlib.pyplot as plt 
    import pandas as pd
    import sagemaker
    from sagemaker import get_execution_role

    # load the pre-trained weights
    model_file = '{}_places365.pth.tar'.format(arch)
    model_file_path=os.path.join(model_folder,model_file)
    if not os.access(model_file_path, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()


    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = os.path.join(model_folder,'categories_places365.txt')
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # result dictionary
    result_dict=dict()

    # run through img list

    for img_name in tqdm.tqdm(os.listdir(input_img_folder)):
        # open img
        img = Image.open(os.path.join(input_img_folder,img_name))
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # convert to np array
        probs_array=probs.numpy()
        classes_list=[classes[idx[i]] for i in range(len(idx))]
        classes_array=np.array(classes_list)

        # store the result in dictionaries
        temp_dict=dict(zip(classes_array,probs_array))
        result_dict[img_name.split('/')[-1].replace('.jpg','')]= temp_dict
        
    # convert dict to df
    result_df=pd.DataFrame.from_dict({(i): result_dict[i]
                            for i in result_dict.keys()},
                        orient='index',
                        columns=list(classes))
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    result_df.to_csv(os.path.join(output_folder,'classification.csv'))

    if s3_upload_bucket!=False:
        role = get_execution_role()
        sess = sagemaker.Session()
        sess.upload_data(path=os.path.join(output_folder,'classification.csv'), bucket=s3_upload_bucket, key_prefix='classification')



