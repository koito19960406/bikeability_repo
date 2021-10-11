# ## Air quality
# - Input cities names
# - use api (https://api.waqi.info/map/bounds/?latlng=&token=) to get station names and their locations (lat&lon)
# - use station names to download the file


def getStationLocation(city_name, output_folder):
    """
    This function is used to get station location from waqi website and save the result as a csv file.
    input: 
        city_name: a string of the city names
        output_folder: a folder to store the output
    """
    import requests
    from pyspark.context import SparkContext
    from pyspark.sql.session import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql import Row
    import pandas as pd
    import os

    # create output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # get station names and locations
    TOKEN='e56a87a074890b0b2ca483097dc793af8eb403fe'
    url='https://api.waqi.info/search/?token={}&keyword={}'.format(TOKEN,city_name)
    response = requests.get(url)
    output=response.json()['data']

    # flatten the json
    # get or create spark context
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)
    # station = spark.createDataFrame(output)
    station=spark.createDataFrame(Row(**x) for x in output)
    station_mod1=station.select('station.name','station.geo')

    # convert spark dataframe to pandas dataframe to clean the data
    station_mod1 = station_mod1.toPandas()
    # remove (Japanese Name)
    station_mod1['name']=station_mod1['name'].str.split('(').str[0].str.strip()
    # split geo into lat and lon
    station_mod1['geo']=station_mod1['geo'].str.replace('\\[','').str.replace('\\]','')
    station_mod1[['lat', 'lon']] = station_mod1['geo'].str.split(', ', 1, expand=True)
    station_mod1=station_mod1[['name','lat','lon']]
    #save the dataframe
    print(station_mod1)
    station_mod1.to_csv(os.path.join(output_folder,'{}_aqi_station.csv'.format(city_name)))


def move(browser, element):
    """
    This function is used to move to a specific element.
    input:
        browser: a webdriver
        element: a selenium element
    """
    desired_y = (element.size['height'] / 2) + element.location['y']
    window_h = browser.execute_script('return window.innerHeight')
    window_y = browser.execute_script('return window.pageYOffset')
    current_y = (window_h / 2) + window_y
    scroll_y_by = desired_y - current_y
    browser.execute_script("window.scrollBy(0, arguments[0]);", scroll_y_by)

def downloadHistData(input_csv):
    """
    This function is used to download the historical data from the AQI website: https://aqicn.org/data-platform/register/ through web scraping
    Input:
        input_csv: a csv file that contains a column names "name" which contains names of stations
    """
    import pandas as pd
    import os
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
    import time

    # load the input
    name=pd.read_csv(input_csv)

    # set basic settings for scraping
    url='https://aqicn.org/data-platform/register/'
    options = webdriver.ChromeOptions()
    options.add_argument(' -- incognito')
    # options.headless = True
    browser = webdriver.Chrome(ChromeDriverManager(version="89.0.4389.23").install(),options=options)
    # browser = webdriver.Chrome(chrome_options=options)

    # start scraping
    browser.get(url)
    # wait for browser to open for 10 sec
    timeout = 10
    try:
        WebDriverWait(browser, timeout).until(
        EC.visibility_of_element_located(
        (By.XPATH, '/html/body/div[7]/center[3]/div/div[2]/div[1]/div[2]/div[1]/input')
        )
        )
    except TimeoutException:
        print('Timed Out Waiting for page to load')
        browser.quit()

    # loop through station names
    for index, row in name.iterrows():
        station_name=row['name']
        print(station_name)
        
        # input the station name
        inputStationElement=browser.            find_element_by_xpath('/html/body/div[7]/center[3]/div/div[2]/div[1]/div[2]/div[1]/input')
        move(browser,inputStationElement)
        inputStationElement.send_keys(station_name)

        # select the first suggestion
        time.sleep(1)
        try:
            WebDriverWait(browser, timeout).until(
            EC.element_to_be_clickable(
            (By.XPATH, '/html/body/div[7]/center[3]/div/div[2]/div[1]/div[2]/div[2]/a[1]/div/div')
            )
            )
        except TimeoutException:
            print('Timed Out Waiting for page to load first suggestion')
            browser.quit()
        firstSuggestion=browser.            find_element_by_xpath('/html/body/div[7]/center[3]/div/div[2]/div[1]/div[2]/div[2]/a[1]/div/div')
        firstSuggestion.click()
        
        # select the download button
        time.sleep(1)
        try:
            WebDriverWait(browser, timeout).until(
            EC.element_to_be_clickable(
            (By.XPATH, '/html/body/div[7]/center[3]/div/div[2]/div[3]/div[1]/center[2]/center/div')
            )
            )
        except TimeoutException:
            print('Timed Out Waiting for page to load download button')
            browser.quit()
        downloadBtn=browser.            find_element_by_xpath('/html/body/div[7]/center[3]/div/div[2]/div[3]/div[1]/center[2]/center/div')
        move(browser,downloadBtn)
        downloadBtn.click()

        # input name, email, and institution
        time.sleep(1)
        try:
            WebDriverWait(browser, timeout).until(
            EC.visibility_of_element_located(
            (By.XPATH, '/html/body/div[7]/center[3]/div/div[2]/div[3]/div[1]/center[2]/div/center/form/div[1]/div[1]/input')
            )
            )
        except TimeoutException:
            print('Timed Out Waiting for page to load')
            browser.quit()
        # input name
        inputName=browser.            find_element_by_xpath('/html/body/div[7]/center[3]/div/div[2]/div[3]/div[1]/center[2]/div/center/form/div[1]/div[1]/input')
        move(browser,inputName)
        inputName.send_keys('My name')
        # input email
        inputEmail=browser.            find_element_by_xpath('/html/body/div[7]/center[3]/div/div[2]/div[3]/div[1]/center[2]/div/center/form/div[1]/div[2]/input')
        inputEmail.send_keys('My@email')
        # input organization
        inputOrg=browser.            find_element_by_xpath('/html/body/div[7]/center[3]/div/div[2]/div[3]/div[1]/center[2]/div/center/form/div[1]/div[3]/input')
        inputOrg.send_keys('My org')
        # click a checkbox
        checkBtn=browser.            find_element_by_xpath('/html/body/div[7]/center[3]/div/div[2]/div[3]/div[1]/center[2]/div/center/form/div[2]/div/input')
        checkBtn.click()
        # click a submit button
        time.sleep(2)
        submitBtn=browser.            find_element_by_xpath('/html/body/div[7]/center[3]/div/div[2]/div[3]/div[1]/center[2]/div/center/form/div[4]')
        submitBtn.click()


        inputStationElement.clear()                           
        time.sleep(2)

    browser.quit()


def getStation(city_name,chrome_path,output_folder):
    """
    This function is used to get station location from waqi website and save the result as a csv file.
        input: 
            city_name: a string of the city names
            chrome_path:a file path to chrome driver
            output_folder: a folder to store the output
    """
    import pandas as pd
    import geopandas as gpd
    import os
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
    import time

    # create output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # set basic settings for scraping
    url='https://aqicn.org/map/{}/'.format(city_name)
    options = webdriver.ChromeOptions()
    options.add_argument(' -- incognito')
    options.headless = True
    browser = webdriver.Chrome(chrome_path,options=options)
    action = webdriver.ActionChains(browser)

    # start scraping
    browser.get(url)
    # wait for browser to open for 10 sec
    timeout = 10
    try:
        WebDriverWait(browser, timeout).until(
        EC.visibility_of_element_located(
        (By.XPATH, '//*[@id="map-stations"]/a[1]')
        )
        )
    except TimeoutException:
        print('Timed Out Waiting for page to load')
        browser.quit()
    
    # get a list of stations on the map
    station_list_raw=browser.find_elements_by_xpath('//*[@id="map-stations"]/a')
    move(browser,station_list_raw[0])
    action.move_to_element(station_list_raw[0])
    action.perform()
    station_list_txt=[station_raw.text.split('(')[0].strip() for station_raw in station_list_raw]
    # quit the browser
    browser.quit()

    # geocoding
    # convert station_list_txt to pandas dataframe
    station_df=pd.DataFrame(station_list_txt,columns=['name'])
    # create a new column for geometry
    # geopy.geocoders.options.default_user_agent = "yourmeial@emailprovider.com"
    station_df['geometry'] = station_df['name'].\
        apply(lambda x: gpd.tools.geocode(x, provider="google",api_key='AIzaSyDmCPxG78h7L9Ts0HJH8ZI_ttAntfRHLR0')['geometry'])
    # get lat and lon
    station_df['lat']=station_df['geometry'].map(lambda p: p.y)
    station_df['lon']=station_df['geometry'].map(lambda p: p.x)
    # save the dataframe
    print(station_df)
    station_df.to_csv(os.path.join(output_folder,'{}_aqi_station.csv'.format(city_name)))




