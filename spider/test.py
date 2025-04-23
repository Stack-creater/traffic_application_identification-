from selenium import webdriver


def test():
    driver = webdriver.Chrome()
    driver.get("https://www.baidu.com")


test()

# curl -X POST -H "Content-Type: application/json" -d '{"func_name":"monitor_lanip","action":"show","param":{"TYPE":"conn,conn_num","ip":"192.168.3.26","interface":"wan1","proto":"all","maxnum":500,"limit":"0,20","ORDER_BY":"","ORDER":""}}' -u 'admin':'intel123' 'http://192.168.3.1/Action/call'


