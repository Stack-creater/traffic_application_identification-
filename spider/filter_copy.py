import os
import pandas as pd

# mxs_data
# 07_12_00_45及以前 - app_pro
# 07_12_12_14及以前 - app_pro_pro
ffile_name = "ojh_data/"
pcap_name = 'ojh_data/05_14_22_38.pcap'  # 文件名
label_file = './' + pcap_name.replace('.pcap', '.csv')
pcap_file = './' + pcap_name
data_dir = './app_ojh_01/'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

labels = pd.read_csv(label_file, encoding='utf-8')

# per app_name to mkdir
apps = labels['app_name'].unique()
apps = list(apps)

# apps.remove("网页浏览")
# apps.remove("HTTPS")
# apps.remove("DNS")
# apps.remove("DoH")
# apps.remove("SSL")
# apps.remove("未知P2P")
# apps.remove("TCP三次握手")

print(type(apps))

for app in apps:
    if not os.path.exists(os.path.join(data_dir, app)):
        os.mkdir(os.path.join(data_dir, app))

    # per app_name to determine the flow
    app_rows = labels[labels['app_name'] == app]  # Dataframe
    for index, row in app_rows.iterrows():  # row: Series
        proc = row['protocol']
        host = row['dst_addr']
        dst_port = str(row["dst_port"])
        port = str(row['src_port'])
        flow_name = proc + "_" + host + "_" + port + "_" + dst_port + ".pcap"
        # if this flow name exists
        while os.path.exists(os.path.join(data_dir, app, flow_name)):
            flow_name = 'f_' + flow_name
        cmd = 'tcpdump -r ' + pcap_file + ' -w ' + data_dir + app + "/" + flow_name + " " + proc + " and host " + host + " and port " + port
        print(cmd)
        if os.system(cmd):
            print("Something wrong happened during reading the pcap!")
            print("proc: " + proc)
            print("host " + host)
            print("port :" + port)
            continue
        else:
            flow_address = data_dir + app + "/" + flow_name
            # just to delete some invalid flows based on size
            if os.path.exists(flow_address) and os.path.getsize(flow_address) <= 24:
                os.remove(flow_address)
                print("Flow Not Found: " + flow_name)
                print("----------------------------------------------")

    # delete the records that don't have pcap
    # condition = (app_rows["protocol"] == "None")
    # print(app_rows)
    # app_rows.drop(app_rows[condition].index, inplace=True)

    # print(app + ": " )
    # print(app_rows.shape)
# labels.to_csv("./new_" + label_file, encoding="utf-8")

