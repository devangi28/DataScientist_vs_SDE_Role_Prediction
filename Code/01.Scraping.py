#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sun May  9 00:19:26 2021

@author: junghopark

#!/usr/bin/env python3
# -- coding: utf-8 --
Created on Sat May  1 22:08:30 2021

@author: junghopark

1 - Collect at least 5,000 Job Ads for Data Scientists from Indeed.com.

2 - Collect at least 5,000 Job Ads for Software Engineers from Indeed.com.

3 - Get the html of the job description (as shown on the right side of the screen after you click on an Ad) for each Ad. 

4 - Extract the text from the html and create a csv with 1 Ad per line and 2 columns: <text>, <job title>

5- Train a classification model that can predict whether a given Ad is for a Data Scientist or Software Engineer.

Notes:

- Your trained model will be evaluated on a separate test set that you will not have access to before the deadline.

- The deliverables include:

The scraping script(s)
The classification script(s)
Instructions on how to run the 2 scripts
The csv from step 4
- Your classification script should be able to read a test csv that will include 1 job description per line (no labels).  It should then produce a new file that includes the predicted label for each line in the test file.
"""

import csv
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium import webdriver
import time
import os

def get_url(position, location):
    """Generate url from position and location"""
    template = 'https://www.indeed.com/jobs?q={}&l={}'
    position = position.replace(' ', '+')
    location = location.replace(' ', '+')
    url = template.format(position, location)
    return url


def save_data_to_file(records, position, location):
    """Save data to csv file"""
    #path to save scraped data
    path = '/Users/junghopark/Desktop/Stevens_Coursework/Spring_2021/BIA 660 Web mining/Final Project'
    
    #check if data folder exists and create one if not
    if not os.path.exists('{}/data'.format(path)):
        os.makedirs('{}/data'.format(path))
    #write csv file    
    with open('{}/data/results_{}_{}.csv'.format(path, position, location), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['JobDesc', 'JobTitle'])
        for i in range(len(records)):
            writer.writerow([records[i], position]) 


def get_record(card):
    """Extract job data from single card"""
    time.sleep(1)
    job_url = card.find_element_by_class_name('jobtitle').get_attribute('href')
    driver= webdriver.Chrome('./chromedriver')
    driver.get(job_url)
    time.sleep(1)
    #get full job desc
    try:
        job_desc = driver.find_elements_by_css_selector("div.jobsearch-jobDescriptionText")[0].text.lower()
        job_desc = job_desc.replace("\n", " ")
    except:
        job_desc = None
        pass

    driver.quit()
    
    return job_desc

def get_page_records(cards, job_list):
    """Extract all cards from the page"""
    for card in cards:
        record = get_record(card)
        if record is not None:
            job_list.append(record)

    return job_list


def main(position, location):
    """Run the main program routine"""
    scraped_jobs = []
    
    url = get_url(position, location)
    
    driver = webdriver.Chrome('./chromedriver')
    driver.get(url)
    time.sleep(2)
    
    # extract the job data
    while True:
        cards = driver.find_elements_by_class_name('jobsearch-SerpJobCard')
        get_page_records(cards, scraped_jobs)
        try:
            driver.find_element_by_xpath('//a[@aria-label="Next"]').click()
            time.sleep(2)
        except NoSuchElementException:
            break
        except ElementNotInteractableException:
            driver.find_element_by_id('popover-x').click()  # to handle job notification popup
            get_page_records(cards, scraped_jobs)
            continue
        
    print(str(len(scraped_jobs)) + ' {} jobs has been scraped from {}'.format(position, location))
    # close driver and save records
    driver.quit()
    save_data_to_file(scraped_jobs, position, location)

    return len(scraped_jobs)

if __name__=='__main__':
    
    # run a search
    start_time = time.time()
    #list of cities to scrape from
    cities = ['chattanooga tn']
    total_jobs = 0
    
    #scrape every jobs in cities list
    for city in range(len(cities)):
        scraped_jobs = main('data scientist', '{}'.format(cities[city])) #change job title
        total_jobs += scraped_jobs
    
    print('--- %s seconds ---' %(time.time()-start_time))
    print('{} total jobs scrapted from {} cities'.format(str(total_jobs), str(len(cities))))

