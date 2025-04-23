from time import sleep
from selenium import webdriver
# 这是新版本新库所需
from selenium.webdriver.common.by import By


# pip show Selenium = 4.4.2
# 当前pip show Selenium = 4.9.0


def get_cookie():
    url_login = 'http://192.168.3.1/login#/login'
    opt = webdriver.ChromeOptions()
    #opt.set_headless()
    opt.add_argument('--headless')
    driver = webdriver.Chrome(options=opt)
    driver.get(url_login)
    sleep(1)
    # driver.find_element_by_xpath('/html/body/div/div/div[2]/form/ul/li[2]/div/input').clear()
    # driver.find_element_by_xpath('/html/body/div/div/div[2]/form/ul/li[2]/div/input').send_keys("admin")
    driver.find_element(By.XPATH, '/html/body/div/div/div[2]/form/ul/li[2]/div/input').clear()
    driver.find_element(By.XPATH, '/html/body/div/div/div[2]/form/ul/li[2]/div/input').send_keys("admin")

    # driver.find_element_by_xpath('/html/body/div/div/div[2]/form/ul/li[3]/div/input').clear()
    # driver.find_element_by_xpath('/html/body/div/div/div[2]/form/ul/li[3]/div/input').send_keys("intel123")
    driver.find_element(By.XPATH, '/html/body/div/div/div[2]/form/ul/li[3]/div/input').clear()
    driver.find_element(By.XPATH, '/html/body/div/div/div[2]/form/ul/li[3]/div/input').send_keys("intel123")

    sleep(1)
    # driver.find_element_by_xpath('/html/body/div/div/div[2]/form/ul/li[4]/button').click()
    driver.find_element(By.XPATH, '/html/body/div/div/div[2]/form/ul/li[4]/button').click()
    
    sleep(1)
    cookie_list = driver.get_cookies()
    print(cookie_list)
    cookie = ""
    for i, element in enumerate(cookie_list):
        # print(element)
        cookie += element["name"] + "=" + element["value"]
        if i < 2:
            cookie +="; "
    print(cookie)
    
    return cookie
# get_cookie()
