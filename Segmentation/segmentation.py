
# In[22]:
def segment_images(model_data,input_img_folder,resized_img_folder,output_txt_folder,output_img_folder,class_json_file,s3_upload_bucket=False):
    """
    This function is used to segment images with a model based on mapillary vistas. 
    Pixel level-accuracy is around 83%
    Input:
        model_data: a tar.gz file created on AWS s3
        input_img_folder: a path to the folder of input images
        resized_img_folder: a path to the folder for input images that will be resized before the segmentation.
        output_txt_folder: a path to the folder for the output CSV file that will contain the ratio of pixels by each category
        class_json_file: a path to the json file that contains category information
        s3_upload_bucket: a string name of s3 bukect you want to upload the output csv to. default is False.  
    """
    
    import sagemaker
    from sagemaker import get_execution_role
    import PIL
    from PIL import Image
    import numpy as np
    import os
    from matplotlib import pyplot as plt
    import tqdm
    import json 
    import pandas as pd 
    from pandas.io.json import json_normalize #package for flattening json in pandas df

    # create directories if the input directories don't exist
    for variable in [resized_img_folder,output_txt_folder,output_img_folder]:
        if not os.path.exists(variable):
            os.makedirs(variable)
        
    # set sagermaker session and image
    role = get_execution_role()
    sess = sagemaker.Session()
    training_image = sagemaker.image_uris.retrieve('semantic-segmentation', sess.boto_region_name)

    # create a model and predictor
    trainedmodel = sagemaker.model.Model(
        model_data=model_data,
        image_uri=training_image,  # example path for the semantic segmentation in eu-west-1
        role=role,
        predictor_cls=sagemaker.predictor.Predictor)  # your role here; could be different name

    ss_predictor=trainedmodel.deploy(initial_instance_count=1,session=sess,instance_type='ml.c5.4xlarge',endpoint_name='segmentation')

    # serialize and deserialize images
    class ImageDeserializer(sagemaker.deserializers.BaseDeserializer):
        """Deserialize a PIL-compatible stream of Image bytes into a numpy pixel array"""
        def __init__(self, accept="image/png"):
            self.accept = accept

        @property
        def ACCEPT(self):
            return (self.accept,)

        def deserialize(self, stream, content_type):
            """Read a stream of bytes returned from an inference endpoint.
            Args:
                stream (botocore.response.StreamingBody): A stream of bytes.
                content_type (str): The MIME type of the data.
            Returns:
                mask: The numpy array of class labels per pixel
            """
            try:
                return np.array(Image.open(stream))
            finally:
                stream.close()
    ss_predictor.deserializer = ImageDeserializer(accept="image/png")
    ss_predictor.serializer = sagemaker.serializers.IdentitySerializer('image/jpeg')

    # create a function for pixel calculation
    def calculate_pixel_ratio(x,array):
        total_num=array.shape[0]*array.shape[1]
        x_num=np.count_nonzero(array==x)
        return x_num/total_num

    # creating variables to be used for extracting the pixl values
    result_dict={}

    # go through each image in the img_data folder
    for file in tqdm.tqdm(os.listdir(input_img_folder)):
        temp_dict={}
        filename_raw = os.path.join(input_img_folder,file)
        width = 800
        im = PIL.Image.open(filename_raw)
        aspect = im.size[0] / im.size[1]
        im.thumbnail([width, int(width / aspect)], PIL.Image.ANTIALIAS)
        filename=os.path.join(resized_img_folder,file)
        im.save(filename, "JPEG")
        with open(filename, 'rb') as imfile:
            imbytes = imfile.read()
        cls_mask = ss_predictor.predict(imbytes)
        
        # save as image
        im_from_array = Image.fromarray(cls_mask)
        im_from_array.save(os.path.join(output_img_folder,file))
        
        for i in range(124):
            ratio=calculate_pixel_ratio(i,cls_mask)
            temp_dict[i]=ratio
        result_dict[file.replace('.jpg','')]= temp_dict
    result_df=pd.DataFrame(result_dict)
    result_df=result_df.transpose()
    
    # change the column names
    # import classes as a dataframe
    with open(class_json_file) as f:
        d = json.load(f)
    classes_df = json_normalize(d['labels'],max_level=3)
    result_df.columns=classes_df['name'].tolist()
    result_df.to_csv(os.path.join(output_txt_folder,'segmentation.csv'))
    if s3_upload_bucket!=False:
        sess.upload_data(path=os.path.join(output_txt_folder,'segmentation.csv'), bucket=s3_upload_bucket, key_prefix='segmentation')
    sagemaker.Session().delete_endpoint(ss_predictor.endpoint_name)

# In[57]:

if __name__=='__main__':
    model_data='s3://sagemaker-ap-southeast-1-428024436188/semantic-segmentation-demo/output/mapillary-segmentation-3/output/model.tar.gz'
