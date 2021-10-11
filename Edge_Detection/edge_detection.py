def edge_detect_image(input_img_folder,output_img_folder,output_csv_folder):
    """
    This function is used to detect edges in images.
        input_img_folder: a folder that contains input image files (takes only .jpg file)
        output_img_folder: a folder to save the output images in
        output_csv_folder: a folder to save the output csv file in
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import tqdm
    import pandas as pd
    import glob
    
    # create output folders
    if not os.path.exists(output_csv_folder):
        os.makedirs(output_csv_folder)
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)

    # a dataframe to store the result
    result_df=pd.DataFrame(columns=['pano_id','edge_ratio'])
    # a list of input images
    img_list=glob.glob(os.path.join(input_img_folder,'*.jpg'))
    
    for img_file in tqdm.tqdm(img_list):
#         # continue if the output image already exists
#         if os.path.isfile(os.path.join(output_img_folder,'{}'.format(img_file.split('/')[-1]))):
#             continue
        img = cv2.imread(img_file)
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(imggray, (5, 5), 0)
        canny = cv2.Canny(blurred,20,50)

#         # plot the result
#         plt.axis("off")
#         plt.imshow(canny)
#         plt.savefig(os.path.join(output_img_folder,'{}'.format(os.path.split(img_file)[1])))
#         # plt.show()
#         plt.close()

        # save it to result_df
        edge_ratio=np.count_nonzero(canny)/(canny.shape[0]*canny.shape[1])
        df_length = len(result_df)
        result_df.loc[df_length]=[os.path.split(img_file)[1].replace('.jpg',''),edge_ratio]
    result_df.to_csv(os.path.join(output_csv_folder,'edge_detection.csv'))




