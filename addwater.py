import pandas as pd
from random import choice
def addwater(data):
    youList = data[data['y_label'] == '油层']['cellid'].drop_duplicates().values.tolist()
    print("油数量: ", len(youList))
    shuiList = data[data['y_label'] == '水层']['cellid'].drop_duplicates().values.tolist()
    print("水数量: ", len(shuiList))
    # 遍历水层造水
    ans = pd.DataFrame()
    col = ['AC', 'GR', 'SP', 'CNL']
    cnt = 0
    for j in range(4):
        # 对哪一列进行改造
        colMul = col[j]
        # 遍历水层
        for k in range(len(shuiList)):
            # 拿到一个水层的id
            id = shuiList[k]
            shui = data[data['cellid'] == id]
            # 改造
            lst = shui[colMul].values.tolist()
            newCol = []
            List = [1.1, 0.9, 1.05]
            c=choice(List)
            for ls in lst:
                newCol.append(ls * c)
            shui[colMul] = newCol
            # 修改cellid，因为是新造的水层
            shui["cellid"] = id + "-" + colMul
            # 造完放起来
            # 拼接起来
            ans = pd.concat([ans, shui])
            cnt += 1
            # if cnt >= (len(youList) - len(shuiList))*2/3:
            if cnt+len(shuiList)>=len(youList)*2/3:
                break
        if cnt+len(shuiList)>=len(youList)*2/3:
            break
    data = pd.concat([data, ans])
    print(cnt)
    data = data.replace("油层", "1")
    data = data.replace("水层", "0")
    return data
if __name__ == '__main__':
    # readpath="./out_put_4/小层+测井数据+"
    readfilename="./测录小层融合数据/小层+测井数据+录井+"
    # endpath="./"
    savefilename="./增加水层数据/小层+测井数据+录井+"
    name = ['5RA25','6RA25','6RT']
    for i in name:
        f = open(readfilename+i+".csv", encoding='utf-8')
        data = pd.read_csv(f)
        print(len(data))
        data1=addwater(data)
        # print(data1["cellid"])
        data1.to_csv(savefilename+i + "-addShui.csv", encoding='utf_8_sig', index=False)
