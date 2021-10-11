def blob_detect_image(input_img_folder,output_csv_folder,output_img_folder):
    """
    This function is used to detect objects in images.
    Input:
        input_img_folder: a folder that contains the input images
        output_csv_folder: a folder to save the output csv file in
        output_img_folder: a folder to save the output img file in
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

    # create the final dataframe 
    result_df=pd.DataFrame(columns=['pano_id','blob_num'])
    # a list of input images
    img_list=glob.glob(os.path.join(input_img_folder,'*.jpg'))
    
    for img_file in tqdm.tqdm(img_list):
        # Read image
        im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create()

        # Detect blobs and save it to result_df
        keypoints = detector.detect(im)
        df_length = len(result_df)
        result_df.loc[df_length]=[os.path.split(img_file)[1].replace('.jpg',''),len(keypoints)]

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(os.path.join(output_img_folder,'{}'.format(os.path.split(img_file)[1])), im_with_keypoints)
        # # Show keypoints
        # cv2.imshow("Keypoints", im_with_keypoints)
        # # cv2.waitKey(0)
        # plt.axis("off")
        # plt.imshow(im_with_keypoints)
        # plt.savefig(os.path.join(output_img_folder,'{}'.format(os.path.split(img_file)[1])))
        # plt.show()

    result_df.to_csv(os.path.join(output_csv_folder,'blob_detection.csv'))



