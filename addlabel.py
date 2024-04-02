import pandas as pd

def jsdataprocessing(x):
    #对小层数据的解释序号进行预处理
    d = dict(zip("Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sup|Oct|Nov|Dec".split("|"),[str(i) for i in range(1,13)]))
    import re
    s = None
    p1 =re.match('(\d+)月(\d+)日',x)
    p2 = re.match('(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sup|Oct|Nov|Dec)-(\d+)',x)
    p3 = re.match('(\d+)-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sup|Oct|Nov|Dec)', x)
    if p1:
        # 如果A月B日，B日小于12，则要转成A-B，如6月1日转成6-1
        # 如果A月B日，B日大于12，则要转成B-A，如1月12日转成12-1
        if float(p1.groups()[1])>12:
            s = str(p1.groups()[1]+"-"+p1.groups()[0])
        else:
            s = str(p1.groups()[0] + "-" + p1.groups()[1])
        # s = 'a'
    if p2:
        s = str(p2.groups()[1]+"-"+d[p2.groups()[0]])
        # s = 'a'
    if p3:
        s = str(d[p3.groups()[1]]+ "-" +p3.groups()[0])
        # s = 'a'
    if s is None:
        s = x
    return s
def skoilceng(xcdata):
    #找到已经射孔小层,并对解释序号进行处理
    # data=pd.read_csv(readxcpath,encoding="utf_8")
    xcdata['解释序号'] = xcdata['解释序号'].apply(lambda x: jsdataprocessing(x))
    xcdata['解释序号']="解释序号:"+xcdata['解释序号']
    xcdata1 = xcdata[xcdata["射孔情况"] == "是"]
    xcdata1 = xcdata1.reset_index(drop=True)
    xcdata1['cellid'] = xcdata1['井号'].str.cat(xcdata1['砂层顶深'].astype(str), sep='$')
    xcdata2 = xcdata[xcdata["射孔情况"] == "否"]
    xcdata2 = xcdata2.reset_index(drop=True)
    xcdata2['cellid'] = xcdata2['井号'].str.cat(xcdata2['砂层顶深'].astype(str), sep='$')
    return xcdata1,xcdata2
    # print(data)
    # xcdata.to_csv(savexcpath,encoding="utf_8",index=False)
def dizuoilceng(xcdata,scdata):
    #找到低阻油层
    #思路：通过小层数据的井号+解释序号对应生产数据的井号+生产序号，对初始生产日产油量进行批分，判断是否小于三吨，结合含水判断是否为油层
    # oilceng=pd.DataFrame()
    xcdata["y_label"]=""
    for i in range(len(xcdata)):
        # print(xcdata)
        # print(xcdata.loc[1, '井号'])
        jingname=xcdata.loc[i,'井号']
        jielun=xcdata.loc[i,'电测解释结论']
        # print("井名",jingname)
        # print(xcdata.loc[i,'解释序号'])
        # print("电测解释结论",jielun)
        jsxuhao=xcdata.loc[i,'解释序号'].split(":")[1]
        # print("解释序号",xcdata.loc[i,'解释序号'])
        scdata1=scdata[(scdata["井号"]==jingname)&(scdata["生产层号"].str.contains(jsxuhao))]
        if len(scdata1)>0:
            scdata1=scdata1.sort_values(by="年月",ascending=True)
            scdata1=scdata1.reset_index(drop=True)
            # print(scdata1)
            num=len(scdata1.loc[0,'生产层号'].split(","))
            oilliang=scdata1.loc[0,'日产油量']
            waterliang = scdata1.loc[0, '含水']
            if (oilliang/num)>3 or waterliang<0.9:
                # oilceng=oilceng.append(xcdata.loc[i])
                xcdata.loc[i,"y_label"]="油层"
            if (waterliang>0.98) and (jielun=="水层"):
                print("电测解释结论",jielun)
                xcdata.loc[i, "y_label"] = "水层"
    xcdata=xcdata[(xcdata["y_label"]=="水层")|(xcdata["y_label"]=="油层")]
    return xcdata
if __name__ == '__main__':
    readxcpath = "./data/小层数据/马东小层数据.csv"
    savexcpath = "./out_put_射孔小层数据/已射孔小层.csv"
    savexcpathp = "./未射孔小层数据/未射孔小层.csv"
    readshpath = "./data/生产数据/马东油井.csv"
    savexcpath1 = "./out_put_射孔油层数据/已射孔油层.csv"
    xcdata = pd.read_csv(readxcpath, encoding="utf_8")
    scdata = pd.read_csv(readshpath, encoding="utf_8")
    data1,data1p=skoilceng(xcdata)
    data1.to_csv(savexcpath,encoding="utf_8",index=False)
    data1p.to_csv(savexcpathp, encoding="utf_8", index=False)
    data2=dizuoilceng(data1,scdata)
    data2.to_csv(savexcpath1,encoding="utf-8",index=False)
