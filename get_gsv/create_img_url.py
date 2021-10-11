#!/usr/bin/env python
# coding: utf-8

# In[10]:


def createAwsUrl(inputCSV,bucketName,outputFolder):
    '''
    This function is used to create a CSV file that contains a list of urls of panorama images
    that we use for surveying.
    Parameters: 
        inputCSV: a CSV file that contains the pano ID
        bucketName: a name of the aws s3 bucket name
        outputFolder: a folder to which the output CSV file should be saved to
    '''
    import pandas as pd
    import tqdm
    import os, os.path

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    result_list=[]
    input_pano_id=pd.read_csv(inputCSV)
    # go through the list of pano id
    for index, row in tqdm.tqdm(input_pano_id.iterrows()):
        pano=row['panoId']
        url='https://{BUCKET_NAME}.s3-ap-southeast-1.amazonaws.com/pano%3D{PANO_ID}.jpg'.            format(BUCKET_NAME=bucketName,PANO_ID=pano)
        result_list.append(url)
    result_df=pd.DataFrame(result_list,columns=['url'])
    result_df.to_csv(os.path.join(outputFolder,'aws_url_{}.csv'.format(bucketName.replace('-','_'))), index=False)

# ------------Main Function -------------------    
if __name__ == "__main__":
    import os, os.path
    
    root = "/Users/koichiito/Documents/NUS/Academic Matter/2021 Spring/Thesis/Trial"
    inputCSV = os.path.join(root,'data/Pnt_start0_end125.csv')
    bucketName='gsv-beauty-score-test-100'
    outputFolder = os.path.join(root,'data')
    createAwsUrl(inputCSV,bucketName,outputFolder)


# In[ ]:




