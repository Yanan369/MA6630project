import csv
import math
import datetime
from average_precision import *
import numpy as np
import random
from sklearn.model_selection import train_test_split
#user_habit_dict:每个用户的乘车记录:起点,终点,距离
user_habit_dict={}
#start_end_dict:每条记录的起点,终点对
start_end_dict={}
#end_start_dict:每条记录的起点,终点对
end_start_dict={}
#user_habit_dict_test:test中每个用户的记录
user_habit_dict_test={}
#bike_dict:bike中的记录
bike_dict={}

#弧度转换
def rad(tude):
    return (math.pi/180.0)*tude

#geohash模块提取的
__all__ = ['encode','decode','bbox','neighbors']
_base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
#10进制和32进制转换，32进制去掉了ailo
_decode_map = {}
_encode_map = {}
for i in range(len(_base32)):
    _decode_map[_base32[i]] = i
    _encode_map[i]=_base32[i]
del i

# 交线位置给左下
def encode(lat,lon,precision=12):
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    geohash=[]
    code=[]
    j=0
    while len(geohash)<precision:
#         print(code,lat_range,lon_range,geohash)
        j+=1
        lat_mid=sum(lat_range)/2
        lon_mid=sum(lon_range)/2
        #经度
        if lon<=lon_mid:
            code.append(0)
            lon_range[1]=lon_mid
        else:
            code.append(1)
            lon_range[0]=lon_mid
        #纬度
        if lat<=lat_mid:
            code.append(0)
            lat_range[1]=lat_mid
        else:
            code.append(1)
            lat_range[0]=lat_mid
        ##encode
        if len(code)>=5:
            geohash.append(_encode_map[int(''.join(map(str,code[:5])),2)])
            code=code[5:]
    return ''.join(geohash)

def decode(geohash):
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    is_lon=True
    for letter in geohash:
        code=str(bin(_decode_map[letter]))[2:].rjust(5,'0')
        for bi in code:
            if is_lon and bi=='0':
                lon_range[1]=sum(lon_range)/2
            elif is_lon and bi=='1':
                lon_range[0]=sum(lon_range)/2
            elif (not is_lon) and bi=='0':
                lat_range[1]=sum(lat_range)/2
            elif (not is_lon) and bi=='1':
                lat_range[0]=sum(lat_range)/2
            is_lon=not is_lon
    return sum(lat_range)/2,sum(lon_range)/2

#返回 精确的经纬度和误差
def decode_exactly(geohash):
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    lat_err, lon_err = 90.0, 180.0
    is_even = True
    for c in geohash:
        cd = _decode_map[c]
        for mask in [16, 8, 4, 2, 1]:
            if is_even: # adds longitude info
                lon_err /= 2
                if cd & mask:
                    lon_interval = ((lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], (lon_interval[0]+lon_interval[1])/2)
            else:      # adds latitude info
                lat_err /= 2
                if cd & mask:
                    lat_interval = ((lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon, lat_err, lon_err

def neighbors(geohash):
    neighbors=[]
    lat_range,lon_range=180,360
    x,y=decode(geohash)
    num=len(geohash)*5
    dx=lat_range/(2**(num//2))
    dy=lon_range/(2**(num-num//2))
    for i in range(1,-2,-1):
        for j in range(-1,2):
            neighbors.append(encode(x+i*dx,y+j*dy,num//5))
#     neighbors.remove(geohash)
    return neighbors

def bbox(geohash):
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    is_lon=True
    for letter in geohash:
        code=str(bin(_decode_map[letter]))[2:].rjust(5,'0')
        for bi in code:
            if is_lon and bi=='0':
                lon_range[1]=sum(lon_range)/2
            elif is_lon and bi=='1':
                lon_range[0]=sum(lon_range)/2
            elif (not is_lon) and bi=='0':
                lat_range[1]=sum(lat_range)/2
            elif (not is_lon) and bi=='1':
                lat_range[0]=sum(lat_range)/2
            is_lon=not is_lon
    #左上、右下；(lat_max,lon_min),(lat_min,lon_max)
    return [(lat_range[1],lon_range[0]),(lat_range[0],lon_range[1])]

#返回 欧式距离  （其实还可以返回南北方向距离,东西方向距离,曼哈顿距离,方向(-0.5:0.5)，但是删了，没啥吊用）
def produceLocationInfo(latitude1, longitude1,latitude2, longitude2):
    radLat1 = rad(latitude1)
    radLat2 = rad(latitude2)
    a = radLat1-radLat2
    b = rad(longitude1)-rad(longitude2)
    R = 6378137
    d = R*2*math.asin(math.sqrt(math.pow(math.sin(a/2),2)+math.cos(radLat1)*math.cos(radLat2)*math.pow(math.sin(b/2),2)))
    detallat = abs(a)*R
    detalLon = math.sqrt(d**2-detallat**2)
    if b==0:
        direction = 1/2 if a*b>0 else -1/2
    else:
        direction = math.atan(detallat/detalLon*(1 if a*b>0 else -1))/math.pi
    return round(d)

#返回 欧式距离
def loc_2_dis(hotStartLocation,hotEndLocation):
    StartLocation = decode_exactly(hotStartLocation[:7])
    EndLocation = decode_exactly(hotEndLocation[:7])
    latitude1 = StartLocation[0]
    longitude1 = StartLocation[1]
    latitude2 = EndLocation[0]
    longitude2 = EndLocation[1]
    return produceLocationInfo(latitude1, longitude1, latitude2, longitude2)

#返回 是否放假,距0点的分钟数,距5月1的天数
def produceTimeInfo(TimeData):
    TimeData = TimeData.split(' ')
    baseData = datetime.datetime(2017, 5, 1, 0, 0, 1)
    mydata = TimeData[0].split('-')
    mytime = TimeData[1].split(':')
    mydata[0] = int(mydata[0])
    mydata[1] = int(mydata[1])
    mydata[2] = int(mydata[2])
    mytime[0] = int(mytime[0])
    mytime[1] = int(mytime[1])
    mytime[2] = int(mytime[2].split('.')[0])
    dt = datetime.datetime(mydata[0], mydata[1], mydata[2], mytime[0], mytime[1], mytime[2])
    minute = mytime[1]+mytime[0]*60
    # return int((dt-baseData).__str__().split(' ')[0]),miao,dt.weekday(),round(miao/900)
    isHoliday = 0
    if dt.weekday()in [5,6] or int((dt-baseData).__str__().split(' ')[0]) in [29,28]:
        isHoliday=1
    return isHoliday,minute,int((dt-baseData).__str__().split(' ')[0])

#模型之间的融合，粗暴的取了最值，这个可以再提升
def add2result(result1,result2):
    for each in result2:
        if each in result1:
            result1[each] = min(result1[each] ,result2[each] )
        else:
            result1[each] = result2[each]
    return result1

def splitData(data, seed, m, k):
    #将数据分成训练集和测试集，每次指定seed，更换K，重复M次，防止过拟合.
    test=[]
    train=[]
    np.random.seed(seed)
    for user,item in data.items():
        if np.random.randint(0,m)==k:
            test.append([user,item])
        else:
            train.append([user,item])
    return test, train

# 其实就是knn算法。这里主要有两点创新。一是给算出来的距离值除以频度的1.1次方，
# 这个加了很多分，二是对于新用户又使用了一个新的knn，其他算法在处理新用户的时候也可以参考下。
def training(trainfile = 'train.csv',testfile = 'test.csv',subfile = 'submission.csv' ,
             leak1 = 0.01 ,leak2 = 4 ,leak3 = 20,              #leak
             qidianquan = 10,shijianquan = 10,jiejiaquan = 2,bikequan = 0.5,  #都是拼音，字面意思，越大则这个特征比重越大
             zhishu = 1.1  #对结果影响很大
             ):
    tr = csv.DictReader(open(trainfile))
    #利用train.csv建立user_habit_dict和start_end_dict
    for rec in tr:
        #print(rec)
        user = rec['userid']
        start = rec['geohashed_start_loc']
        end = rec['geohashed_end_loc']
        rec['isHoliday'] , rec['minute'] , rec['data'] = produceTimeInfo(rec['starttime'])
        if user in user_habit_dict:
            user_habit_dict[user].append(rec)
        else:
            user_habit_dict[user] = [rec]

        if start in start_end_dict:
            start_end_dict[start].append(rec)
        else:
            start_end_dict[start] = [rec]

        if end in end_start_dict:
            end_start_dict[end].append(rec)
        else:
            end_start_dict[end] = [rec]
    #print(user_habit_dict)

    print('train done!')
    # te是测试文件
    te = csv.DictReader(open(testfile))
    for rec in te:
        #print(rec)
        user = rec['userid']
        bike = rec['bikeid']
        rec['isHoliday'], rec['minute'], rec['data'] = produceTimeInfo(rec['starttime'])
        if user in user_habit_dict_test:
            user_habit_dict_test[user].append(rec)
        else:
            user_habit_dict_test[user] = [rec]

        if bike in bike_dict:
            bike_dict[bike].append(rec)
        else:
            bike_dict[bike] = [rec]
    #print(user_habit_dict_test)

    print("test done!")


    #sub是提交文件
    sub = open(subfile, 'w', newline= '')
    writer = csv.writer(sub)
    writer.writerow(["orderid","loc1","loc2","loc3"])  #写入列名


    iter1 = 0
    # AllhotLocSort = sorted(end_start_dict.items(), key=lambda d: len(d[1]), reverse=True)
    te1 = csv.DictReader(open(testfile))
    for rec in te1:
        #print(rec)
        iter1 += 1
        if iter1  % 10000== 0:
            print(iter1/20000,'%',sep='')
        # testTime = timeSlipt(rec['minute'])
        rec['isHoliday'], rec['minute'], rec['data'] = produceTimeInfo(rec['starttime'])
        user1 = rec['userid']
        bikeid1 = rec['bikeid']
        order1 = rec['orderid']
        start1 = rec['geohashed_start_loc']
        hour1 = rec['minute']/60
        minute1 = rec['minute']
        isHoliday1 = rec['isHoliday']
        biketype1 = rec['biketype']
        data1 = rec['data']
        result = {}
        hotLoc = {}

        #knn
        if user1 in user_habit_dict:
            #print(user1)
            for eachAct in user_habit_dict[user1]:
                #print(eachAct)
                start2 = eachAct['geohashed_start_loc']
                end2 = eachAct['geohashed_end_loc']
                hour2 = eachAct['minute']/60
                isHoliday2 = eachAct['isHoliday']
                biketype2 = eachAct['biketype']
                data2 = rec['data']

                dis = loc_2_dis(start1, start2)
                dis = min(dis, 1000)    #1000 最后一公里
                qidian= qidianquan * (dis / 100) ** 2  #start权重

                detalaTime = abs(hour2 - hour1) if abs(hour2 - hour1) < 12 else 24 - abs(hour2 - hour1)
                shijian= shijianquan * (detalaTime / 12 * 10) ** 2  #时刻权重

                dayType = isHoliday2 - isHoliday1
                jiejia= jiejiaquan * (dayType * 10) ** 2         #日期权重（是否放假）

                biType = int(biketype2) - int(biketype1)
                bike= bikequan * (biType * 10) ** 2  #0.5 单车类型权重


                score = qidian+shijian+jiejia+bike   #jvli+fangxiang 得分

                # print(qidian,shijian,jiejia,bike,jvli,fangxiang)
                if end2 in hotLoc:    #是否热启动（是否有历史数据）
                    hotLoc[end2] += 1
                else:
                    hotLoc[end2] = 1

                if end2 in result:
                    if result[end2] > score:
                        result[end2] = score
                else:
                    result[end2] = score

            for each in hotLoc:
                result[each] = result[each] / (hotLoc[each]**zhishu)  #0

            for each in result:
                result[each] = math.sqrt(result[each])

        #利用test中的用户历史记录
        if user1 in user_habit_dict_test:
            resulttest = {}
            user_habit_dict_test[user1].sort(key = lambda x:x['data']*60*24+x['minute'])
            xuhao = 0
            for i in range(len(user_habit_dict_test[user1])-1):
                if user_habit_dict_test[user1][i]['orderid'] == order1:
                    xuhao = i
                    resulttest[user_habit_dict_test[user1][i+1]['geohashed_start_loc']] = 21
            for i in range(len(user_habit_dict_test[user1])):
                if i not in [xuhao,xuhao+1]:
                    resulttest[user_habit_dict_test[user1][i]['geohashed_start_loc']] = 21+abs(i-xuhao)
                result = add2result(result, resulttest)


        #起点终点对的knn
        if start1 in start_end_dict:
            endDict = {}
            resultqizhong={}
            for eachAct in start_end_dict[start1]:
                score = 0
                score += (24-abs(hour1-eachAct['minute']/60))/24
                score += (1-abs(isHoliday1-eachAct['isHoliday']))*0.4
                if eachAct['geohashed_end_loc'] in endDict:
                    endDict[eachAct['geohashed_end_loc']] += score
                else:
                    endDict[eachAct['geohashed_end_loc']] = score
            hotLoc = sorted(endDict.items(),key = lambda x:x[1],reverse=True)
            #print(hotLoc)
            if len(hotLoc)>=1:
                resultqizhong[hotLoc[0][0]] = 1000
            if len(hotLoc) >= 2:
                resultqizhong[hotLoc[1][0]] = 1001
            if len(hotLoc) >= 3:
                resultqizhong[hotLoc[2][0]] = 1002
            result = add2result(result, resultqizhong)

        #剔除不合理结果
        for each in result:
            distance = loc_2_dis(each,start1)
            if distance > 2500:
                result[each] = 1999


        if start1 in result:
            result[start1] = min(2000, result[start1])
        else:
            result[start1]=2000
        result['fail2'] = 2001
        result['fail3'] = 2002
        #print(result)

        bestResult = sorted(result.items(), key=lambda d: d[1])  #赋值比较排序
        #print(bestResult)
        string = rec['orderid']
        #print(string)
        num = 0
        for item in bestResult:
            string += ',' + item[0]
            # string += ':' + str(item[1]) + '\t'
            num += 1
            if num == 3:
                break
        sub.write(string + '\n')

    sub.close()
    print('ok')

if __name__ =="__main__":
    training('train1.csv', 'test1.csv', 'submission1.csv' )

with open('submission1.csv', 'r') as predicted:
    reader = csv.reader(predicted)
    column = [row[:] for row in reader]
    predicted_list = list(column)
slice = random.sample(predicted_list, 10)
print(slice)

# #提取出test中的终点位置list
# with open('test1.csv', 'r') as actual:
#     reader = csv.reader(actual)
#     column = [row[6] for row in reader]
#     actual_list = list(column)
# #提取出submission中的终点预测位置list
# with open('submission1.csv', 'r') as predicted:
#     reader = csv.reader(predicted)
#     column = [row[1:4] for row in reader]
#     predicted_list = list(column)

#print(actual_list[1:])
#print(predicted_list[1:])


