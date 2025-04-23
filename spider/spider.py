import os
import datetime
from time import sleep
import requests
import csv
import pandas as pd
from time import sleep
import cookie
# name = 'test'
csv_file = './label.csv'
csv_header = ['protocol', 'dst_addr', 'src_port', 'dst_port', 'forward_addr', 'app_name','total_up', 'total_down', 'status', 'interface']
MAX_NUM = 5000
# 收集数据的ip
fixed_ip = '192.168.3.27'
cookie_str = cookie.get_cookie()

header = {
"Accept": 'application/json, text/plain, */*',
'Accept-Encoding': 'gzip, deflate',
'Accept-Language': 'zh-CN,zh;q=0.9',
'Connection': 'keep-alive',
'Content-Length': '185',
'Content-Type': 'application/json;charset=UTF-8',
# 'Cookie': 'sess_key=0795787d272976772f09608d9dc070e8; username=admin; login=1',
'Cookie': cookie_str,
'Host': '192.168.3.1',
'Origin': 'http://192.168.3.1',
'Referer': 'http://192.168.3.1/',
'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
}
 
url = 'http://192.168.3.1/Action/call'
# payload = '{"func_name":"monitor_lanip","action":"show","param":{"TYPE":"conn,conn_num","ip":"192.168.4.106","interface":"wan1","proto":"all","maxnum":500,"limit":"0,20","ORDER_BY":"","ORDER":""}}'
payload = '{"func_name":"monitor_lanip","action":"show","param":{"TYPE":"conn,conn_num","ip":"' + fixed_ip + '","interface":"wan1","proto":"all","maxnum":500,"limit":"0,20","ORDER_BY":"","ORDER":""}}'
# 开始时间
print(datetime.datetime.now())
name = datetime.datetime.now().strftime('%m_%d_%H_%M')
print(name)
with open(csv_file, 'a+', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_header)
    if not os.path.getsize(csv_file):
        writer.writeheader()
    cnt = 0 
    while True:
        response = requests.post(url, data=payload, headers=header)
        res = response.json()
        print(res)
        writer.writerows(res['Data']['conn'])
        sleep(1)
        # stop at proper timing
        cnt += 1
        print("*")
        print(cnt)
        print("*")
        if cnt >= MAX_NUM:
            break

# 结束时间
print(datetime.datetime.now())


# {'protocol': 'tcp', 'status': '已连接', 
# 'dst_addr': '36.25.245.251', 'src_port': 63441, 'dst_port': 443, 'forward_addr': '192.168.71.86', 
# 'app_name': '七牛云', 'interface': 'wan1', 'total_up': 1965, 'total_down': 8058}

# ----- delete duplicated records --------
save_file =  csv_file.replace('label', name)
data = pd.read_csv(csv_file, encoding='utf-8')
print(data.shape)

data.drop(columns=['status', 'interface', 'total_up', 'total_down'], inplace=True)
data.drop_duplicates(subset=['protocol','dst_addr','src_port','dst_port','forward_addr'], keep='last', inplace=True)
# data['forward_addr'] = fixed_ip
print(data.shape)
data.to_csv(save_file, encoding='utf-8')
os.remove('./label.csv')