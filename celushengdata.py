import pandas as pd
# cescjingfilename="./out_put_4/小层+测井数据_train_20220808.csv"
# cescjingfilename1="./out_put_4p/小层+测井数据_test_20220808.csv"
def celushengdata(data1,data2):
        # print(len(data1))
        # print(len(data2))
        # print(data2["JH"]+data2["顶深"])
        #data1:训练或者测试数据
        #data2:录井数据
        print(data2.columns)
        data2["cellid"] = data2["JH"] + "$" + data2["顶深"].astype(str)
        data2 = data2[["cellid", "含油级别", "YGYS", '岩石描述']]
        # #饱含油:1
        # #饱含油:2
        # #饱含油:3
        # #饱含油:4
        # #饱含油:5
        # #饱含油:6

        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["饱含油"],value =1)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["富含油"],value =2)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["荧光"],value =3)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["油斑"],value =4)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["油迹"],value =5)
        # data2["含油级别"]=data2["含油级别"].replace(to_replace =["油浸"],value =6)
        # # print(data2.head(10))
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["暗黄","暗黄色"],value =1)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["无色","无"],value =2)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["亮黄色"],value =3)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["棕黄","棕黄色"],value =4)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["黄白","黄白色"],value =5)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["淡黄","淡黄色"],value =6)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["黄","黄色"],value =7)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["浅黄","浅黄色"],value =8)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["黄褐","黄褐色","褐黄色"],value =9)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["亮黄"],value =10)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["蛋黄"],value =11)
        # data2["YGYS"]=data2["YGYS"].replace(to_replace =["乳白","白"],value =12)
        # data2["YGYS"] = data2["YGYS"].replace(to_replace=["乳白", "白"], value=12)
        data3 = pd.merge(data1, data2, on='cellid')
        return data3
if __name__ == '__main__':
    lujingfilename = "./data/录井数据/油气显示数据.csv"#录井数据
    file1="./out_put_4/小层+测井数据_train_20220808.csv"#训练数据
    file2="./out_put_4p/小层+测井数据_test_20220808.csv"#测试数据
    endfile1="./测录小层融合数据/测录小层融合训练数据.csv"
    endfile2="./测录小层融合数据/测录小层融合测试数据.csv"
    data1 = pd.read_csv(lujingfilename, encoding="gbk")
    data2 = pd.read_csv(file1, encoding="utf_8")
    data3 = pd.read_csv(file2, encoding="utf_8")
    data4=celushengdata(data2,data1)
    data4.to_csv("./测录小层融合数据/测录小层融合训练数据.csv", encoding="utf_8", index=False)
    data5=celushengdata(data3, data1)
    data5.to_csv("./测录小层融合数据/测录小层融合测试数据.csv", encoding="utf_8", index=False)



