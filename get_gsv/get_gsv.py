#!/usr/bin/env python
# coding: utf-8

def getStreetNetwork(city_list, out_folder):
    """
    This function is used to get street network shapefile from OSM.
    parameters:
        city_list: a list of names of cities
    """
    import osmnx as ox
    import os

    for city in city_list:
        G = ox.graph_from_place(city, network_type='all')
        ox.plot_graph(G)
        if not os.path.exists(os.path.join(out_folder,'{}_street_network'.format(city))):
            ox.save_graph_shapefile(G,os.path.join(out_folder,'{}_street_network'.format(city)))
        
# ------------Main Function -------------------    
if __name__ == '__main__':
    city_list=['Singapore','Tokyo']
    root = "/Users/koichiito/Documents/NUS/Academic Matter/2021 Spring/Thesis/Trial"
    out_folder=os.path.join(root,'data')
    getStreetNetwork(city_list,out_folder)


# In[23]:


def createPoints(inshp, outshp, mini_dist):
    
    '''
    This function will parse throigh the street network of provided city and
    clean all highways and create points every mini_dist meters (or as specified) along
    the linestrings
    Required modules: Fiona and Shapely
    parameters:
        inshp: the input linear shapefile, must be in WGS84 projection, ESPG: 4326
        output: the result point feature class
        mini_dist: the minimum distance between two created point
    
    '''
    
    import fiona
    import os,os.path
    from shapely.geometry import shape,mapping
    from shapely.ops import transform
    from functools import partial
    import pyproj
    from fiona.crs import from_epsg
    import tqdm
    
    count = 0
    
    #name the road types NOT to be used.  
    s = {'trunk_link','motorway','motorway_link','steps', None,'trunk','bridleway'}
    
    # the temporaray file of the cleaned data
    root = os.path.dirname(inshp)
    basename = 'clean_' + os.path.basename(inshp)
    temp_cleanedStreetmap = os.path.join(root,basename)
    
    # if the tempfile exist then delete it
    if os.path.exists(temp_cleanedStreetmap):
        fiona.remove(temp_cleanedStreetmap, 'ESRI Shapefile')
    
    # clean the original street maps by removing highways, if it the street map not from Open street data, users'd better to clean the data themselve
    with fiona.open(inshp) as source, fiona.open(temp_cleanedStreetmap, 'w', driver=source.driver, crs=source.crs,schema=source.schema) as dest:
        
        for feat in source:
            try:
                i = feat['properties']['highway'] # for the OSM street data
                if i in s:
                    continue
            except:
                continue
               # if the street map is not osm, do nothing. You'd better to clean the street map, if you don't want to map the GVI for highways
               # key = dest.schema['properties'].keys()[0] # get the field of the input shapefile and duplicate the input feature
               # i = feat['properties'][key]
               # if i in s:
            #    continue
            
            dest.write(feat)

    schema = {
        'geometry': 'Point',
        'properties': {'id': 'int'},
    }

    # Create pointS along the streets
    with fiona.drivers():
        #with fiona.open(outshp, 'w', 'ESRI Shapefile', crs=source.crs, schema) as output:
        with fiona.open(outshp, 'w', crs = from_epsg(4326), driver = 'ESRI Shapefile', schema = schema) as output:
            for line in tqdm.tqdm(fiona.open(temp_cleanedStreetmap)):
                first = shape(line['geometry'])
                if first.geom_type != 'LineString': continue
                
                length = first.length
                
                try:
                    # convert degree to meter, in order to split by distance in meter
                    wgs84 = pyproj.CRS('EPSG:4326')
                    pseudo_wgs84 = pyproj.CRS('EPSG:3857')
                    project = pyproj.Transformer.from_crs(wgs84, pseudo_wgs84, always_xy=True).transform
                    line2= transform(project, first)
                    linestr = list(line2.coords)
                    dist = mini_dist #set
                    for distance in range(0,int(line2.length), dist):
                        point = line2.interpolate(distance)
                        
                        # convert the local projection back the the WGS84 and write to the output shp
                        project2 = pyproj.Transformer.from_crs(pseudo_wgs84, wgs84, always_xy=True).transform
                        point = transform(project2, point)
                        output.write({'geometry':mapping(point),'properties': {'id':1}})
                except Exception as e:
                    print(e)
                    print ("You should make sure the input shapefile is WGS84")
                    return
                    
    print("Process Complete")
    
    # delete the temprary cleaned shapefile
    fiona.remove(temp_cleanedStreetmap, 'ESRI Shapefile')


# Example to use the code, 
# Note: make sure the input linear featureclass (shapefile) is in WGS 84 or ESPG: 4326
# ------------main ----------
if __name__ == "__main__":
    import os,os.path
    import sys
    
    root = "/Users/koichiito/Documents/NUS/Academic Matter/2021 Spring/Thesis/Trial"
    inshp = os.path.join(root,'data/high_test.shp') #the input shapefile of road network
    outshp = os.path.join(root,'data/test_100m_point.shp') #the output shapefile of the points
    mini_dist = 100 #the minimum distance between two generated points in meters
    createPoints(inshp, outshp, mini_dist)


# In[41]:


def GSVpanoMetadataCollector(samplesFeatureClass,num,ouputTextFolder):
    '''
    This function is used to call the Google API url to collect the metadata of
    Google Street View Panoramas. The input of the function is the shapefile of the created sample points. If a GSV panorama exists within
    50 meters of the sample point location, the Google API returns the metadata of that panorama. 
    The output is the generated panoinfo matrics stored in the text file
    
    Parameters: 
        samplesFeatureClass: the shapefile of the created sample points
        num: the number of sites proced every time
        ouputTextFolder: the output folder for the panoinfo
        
    '''
    import pandas as pd
    import urllib.request
    import geopandas as gpd
    import xmltodict
    import time
    import os,os.path
    import requests
    import xml.etree.ElementTree as ET
    import math
    import tqdm
    
    if not os.path.exists(ouputTextFolder):
        os.makedirs(ouputTextFolder)
    
    # count the number of rows
    points=gpd.read_file(samplesFeatureClass)
    index = points.index
    number_of_rows = len(index)
    batch = (number_of_rows//num)+1

    # get lon and lat
    points['lon'] = points['geometry'].x
    points['lat'] = points['geometry'].y

    for b in tqdm.tqdm(range(batch),position=1):
        # for each batch process num GSV site
        start = b*num
        end = (b+1)*num
        if end > number_of_rows:
            end = number_of_rows
        
        ouputTextFile = 'Pnt_start%s_end%s.csv'%(start,end)
        ouputGSVinfoFile = os.path.join(ouputTextFolder,ouputTextFile)
        result_df=pd.DataFrame(columns=['panoLon','panoLat','panoId','panoDate','distDiff'])
        
        # skip over those existing txt files
        if os.path.exists(ouputGSVinfoFile):
            continue
        
        time.sleep(1)
        
        # process num feature each time
        for i in tqdm.tqdm(range(start, end),position=0):
            point=points.iloc[i,:]
            lon = point['lon']
            lat = point['lat']
            key = 'AIzaSyDmCPxG78h7L9Ts0HJH8ZI_ttAntfRHLR0'
            input_df=pd.DataFrame({'lon':[lon],
                                   'lat':[lat]})
            input_gdf=gpd.GeoDataFrame(input_df, geometry=gpd.points_from_xy(input_df.lon, input_df.lat))
            input_gdf=input_gdf.set_crs(epsg=4326)
            input_gdf=input_gdf.to_crs("EPSG:3857")
            # get the meta data of panoramas 
            urlAddress = 'https://maps.googleapis.com/maps/api/streetview/metadata?&location={LAT},{LON}&key={KEY}'.                format(LAT=lat,LON=lon,KEY=key)
            time.sleep(0.05)
            # the output result of the meta data is a xml object
            response = requests.get(urlAddress)
            data=response.json()
            
            # in case there is not panorama in the site, therefore, continue
            if data['status']!='OK':
                continue
            else:
                try:
                    panoDate = data['date']
                except KeyError:
                    panoDate = ''
                # get the meta data of the panorama
                panoId = data['pano_id']
                panoLat = data['location']['lat']
                panoLon = data['location']['lng']
                output_df=pd.DataFrame({'lon':[panoLon],
                                       'lat':[panoLat]})
                output_gdf=gpd.GeoDataFrame(output_df, geometry=gpd.points_from_xy(output_df.lon, output_df.lat))
                output_gdf=output_gdf.set_crs(epsg=4326)
                output_gdf=output_gdf.to_crs("EPSG:3857")
                # calculate the distance between input and output
                distDiff=math.sqrt((input_gdf['lon'][0]-output_gdf['lon'][0])**2+(input_gdf['lat'][0]-output_gdf['lat'][0])**2)
                list=[panoLon,panoLat,panoId, panoDate, distDiff]
                result_df.loc[len(result_df)] = list
                    
        result_df.to_csv(ouputGSVinfoFile)
                    


# ------------Main Function -------------------    
if __name__ == "__main__":
    import os, os.path

    root = "/Users/koichiito/Documents/NUS/Academic Matter/2021 Spring/Thesis/Trial"
    inputShp = os.path.join(root,'data/test_100m_point.shp')
    outputTxt = os.path.join(root,'data')
    
    GSVpanoMetadataCollector(inputShp,1000,outputTxt)


# In[3]:


def getSamplePoints(intxt,outtxt,sample_num):
    '''
    This function is used to get n sample points from input shapefile
    Parameters:
        inshp: a shapefile of point features
        outshp: a shapefile of sampled point features
        sample_num: a number of sample points
    '''
    import pandas as pd
    import geopandas as gpd
    
    if intxt.split('.')[1]=='shp':
        point_gdf=gpd.read_file(intxt)
        sample_points=point_gdf.sample(sample_num)
        sample_points.to_file(outtxt)
    elif intxt.split('.')[1]=='csv':
        point_df=pd.read_csv(intxt)
        sample_points=point_df.sample(sample_num)
        sample_points.to_csv(outtxt)
    else:
        print('Please use csv or shp file')

# ------------Main Function -------------------    
if __name__ == "__main__":
    import os, os.path

    sample_num=20
    root = "/Users/koichiito/Documents/NUS/Academic Matter/2021 Spring/Thesis/Trial"
    inshp=os.path.join(root,'data/Pnt_start0_end125.csv')
    outshp=os.path.join(root,'data/test_sample_point_{}.shp'.format(str(sample_num)))
    getSamplePoints(inshp,outshp,sample_num)


# In[84]:


def getGSV(inputCSV,outputImgFolder):
    '''
    This function is used to call the Google API url to collect the GSV images
    Parameters: 
        inputCSV: a CSV file that contains the pano ID
        outputImgFolder: a folder to which output images should be saved
    '''
    import pandas as pd
    import requests
    import tqdm
    import os

    if not os.path.exists(outputImgFolder):
        os.makedirs(outputImgFolder)

    input_pano_id=pd.read_csv(inputCSV)

    # set parameters
    FOV=90
    KEY='AIzaSyDmCPxG78h7L9Ts0HJH8ZI_ttAntfRHLR0'
    SIZE=640
    SOURCE='outdoor'
    
    # go through the list of pano id
    for index, row in tqdm.tqdm(input_pano_id.iterrows()):
        pano=row['panoId']
        for heading in [0,90,180,270]:
            url="https://maps.googleapis.com/maps/api/streetview?size={SIZE}x{SIZE}&pano={PANO}&fov={FOV}&heading={HEADING}&source={SOURCE}&key={KEY}".\
            format(SIZE=SIZE,PANO=pano,FOV=FOV,HEADING=heading,SOURCE=SOURCE,KEY=KEY)
            page = requests.get(url)
            f_ext = 'pano={}_head={}.jpg'.format(pano,heading)
            f_name = os.path.join(outputImgFolder, f_ext)
            with open(f_name, 'wb') as f:
                f.write(page.content)
                
# ------------Main Function -------------------    
if __name__ == "__main__":
    import os, os.path
    
    root = "/Users/koichiito/Documents/NUS/Academic Matter/2021 Spring/Thesis/Trial"
    inputCSV = os.path.join(root,'data/Pnt_start0_end125.csv')
    outputImgFolder = os.path.join(root,'img_test')
    getGSV(inputCSV,outputImgFolder)


# In[ ]:




