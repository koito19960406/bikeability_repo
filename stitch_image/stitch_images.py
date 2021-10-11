#!/usr/bin/env python
# coding: utf-8

# In[2]:


from PIL import Image


# In[20]:


def stitchImg(inputCSV,inputImgFolder,outputImgFolder):
    '''
    This function is used to stitch GSV images of the same pano ID into a single image
    Parameters: 
        inputCSV: a CSV file that contains the pano ID
        inputImgFolder: a folder of the input images
        outputImgFolder: a folder to which output images should be saved
    '''
    from PIL import Image
    import glob
    import pandas as pd
    import tqdm
    import os, os.path

    if not os.path.exists(outputImgFolder):
        os.makedirs(outputImgFolder)

    input_pano_id=pd.read_csv(inputCSV)
    # go through the list of pano id
    for index, row in tqdm.tqdm(input_pano_id.iterrows()):
        pano=row['panoId']
        img_list=[img for img in glob.glob(os.path.join(inputImgFolder,'pano={}*.jpg'.format(pano)))]
        img_list=sorted(img_list, key=lambda x: (len(x),x))
        first_loop=True
        for img in img_list:
            image = Image.open(img)
            image_size = image.size
            if first_loop:
                new_image = Image.new('RGB',(4*image_size[0], image_size[1]), (250,250,250))
                img_count=0
                first_loop=False
            new_image.paste(image,(img_count*image_size[0],0))
            img_count+=1

        new_image.save(os.path.join(outputImgFolder,'pano={}.jpg'.format(pano)),"JPEG")

# ------------Main Function -------------------    
if __name__ == "__main__":
    import os, os.path
    
    root = "/Users/koichiito/Documents/NUS/Academic Matter/2021 Spring/Thesis/Trial"
    inputCSV = os.path.join(root,'data/Pnt_start0_end125.csv')
    inputImgFolder = os.path.join(root,'img_test')
    outputImgFolder = os.path.join(root,'img_stitched')
    stitchImg(inputCSV,inputImgFolder,outputImgFolder)


# In[ ]:




