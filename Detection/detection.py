def detect_images(input_img_folder,output_csv_folder,output_img_folder,model='yolo3_darknet53_coco'):
    """
    This function is used to detect objects in images.
    Input:
        input_img_folder: a folder that contains the input images
        output_csv_folder: a folder to save the output csv file in
        output_img_folder: a folder to save the output img file in
        model: a string of the detection model to use. Default is 'yolo3_darknet53_coco'
    """
    from gluoncv import model_zoo, data, utils
    import os
    import glob
    import numpy as np
    import pandas as pd
    import tqdm
    from matplotlib import pyplot as plt

    # create output folders
    if not os.path.exists(output_csv_folder):
        os.makedirs(output_csv_folder)
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)
    
    # get a model
    net = model_zoo.get_model(model, pretrained=True)

    # transform the input images
    im_fnames=glob.glob(os.path.join(input_img_folder,'*.jpg'))
    x, img = data.transforms.presets.yolo.load_test(im_fnames, short=512)


    # ## Inference and display
    # The forward function will return all detected bounding boxes, and the
    # corresponding predicted class IDs and confidence scores. Their shapes are
    # `(batch_size, num_bboxes, 1)`, `(batch_size, num_bboxes, 1)`, and
    # `(batch_size, num_bboxes, 4)`, respectively.
    # 
    # We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
    # results. We slice the results for the first image and feed them into `plot_bbox`:
    
    # the dictionary to store the result
    result_dict=dict()
    for i in tqdm.tqdm(range(len(x))):
#         # continue if the output image already exists
#         if os.path.isfile(os.path.join(output_img_folder,'{}'.format(im_fnames[i].split('/')[-1]))):
#             continue
            
        # temporary dictionary to store data for this loop
        temp_dict={}
        
        class_IDs, scores, bounding_boxs = net(x[i])
#         # visualize the detected images
#         ax = utils.viz.plot_bbox(img[i], bounding_boxs[0], scores[0],
#                             class_IDs[0], class_names=net.classes)
        high_scores=scores[0][:,0]>0.5
        indices=np.where(high_scores>0)[0]
        if indices.size > 0:
            classIDs_high=class_IDs[0][:,0][indices].astype(int).asnumpy()
            for j in range(len(net.classes)):
                count=np.count_nonzero(classIDs_high==j)
                temp_dict[j]=count
            result_dict[im_fnames[i].split('/')[-1].replace('.jpg','')]=temp_dict
        else:
            result_dict[im_fnames[i].split('/')[-1].replace('.jpg','')]= ''
#         # save the figure
#         plt.savefig(os.path.join(output_img_folder,'{}'.format(im_fnames[i].split('/')[-1])),dpi=400)
#         plt.close()
        
    # convert the result to a dataframe
    result_df=pd.DataFrame(result_dict)
    result_df=result_df.transpose()
    result_df.columns=net.classes
    result_df = result_df.replace("", np.nan, regex=True)
    result_df=result_df.fillna(0)
    result_df=result_df.astype(int)

    # save the data to output_folder
    result_df.to_csv(os.path.join(output_csv_folder,'detection.csv'))




