import numpy as np
import cv2
import math
import time
import sys
from datetime import datetime
import smtplib

'''
    Параметры:
        sourceType      -- image (изображение из файла), video (видеопоток), webcamera (веб-камера)
        source          -- путь к источнику изображения
        xf, yf, hf, wf  -- фрагмент изображения для анализа
        thresh          -- порог тонового отличия линии от фона (от 0 до 255) -- до thresh превращается в черый, после -- в белый
        bMin, bMax      -- минимальный и максимальный радиус окружности вокруг центра датчика, в которой должно находиться начало линии-указател
        eMin, eMax      -- минимальный и максимальный радиус окружности вокруг центра датчика, в которой должен находиться конец линии-указател
        degMn, degMx    -- минимальное и максимальное значения на шкале градусов, соответствующих шкале значений измеряемой величины (degMx может быть меньше degMn; оба значения в градусах от 0 до 360)
        valMin, valMax  -- минимальное и максимальное значения измеряемой величины (valMin должно быть всегда меньше valMax)
        direction       -- направление шкалы значений: 1 -- по часовой стрелке, -1 -- против часовой стрелки
        alarmLimit      -- предельное значение, после которого отправляется email
        pause           -- время в секундах между двумя итерациями
        mail_server     -- адрес почтового сервера
        mail_port       -- порт почтового сервера
        mail_login      -- логин для входа на почтовый сервер (SMTP)
        mail_password   -- пароль
        mail_sender     -- email отправителя
        mail_receiver   -- email получателя

    Результат:
        value           -- значение
''' 

'''
    Изображение (фрагмент) уменьшается до 600x600 с сохранением соотношения сторон, либо увеличивается до 400x400 пикселей
    Изображение превращается в оттенки серого и размывается, на нем ищутся круги:
        минимальный радиус -- 10% высоты изображения
        максимальный радиус -- 95% высоты изображения
        минимальная дистанция между центрами -- 80% высоты изображения
    Берется самый вероятный круг
    На черно-белом, но не размытом изображении ищутся линии
    Самая вероятная линия будет указателем на датчике
    Если 5 подряд значений с разбросом не более 15% были за пределами допустимого предела (alarmLimit), то отправляется email
'''

params = {
    "sourceType" : "video",
    "source" : "rtsp://91.146.47.10/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream",
    "xf" : 700,
    "yf" : 230,
    "hf" : 300,
    "wf" : 300,
    "bMin" : 0,
    "bMax" : 0.4,
    "eMin" : 0.4,
    "eMax" : 0.99,
    "thresh" : 120,
    "degMin" : 225,
    "degMax" : 315,
    "valMin" : 0,
    "valMax" : 4,
    "direction" : 1,
    "pause" : 2,
    "alarmLimit": 1,
    "mail_server" : "smtp.yandex.ru",
    "mail_port" : 465,
    "mail_login" : "iaroslava.galimova",
    "mail_password" : "qhkrwahzvhpyiis",
    "mail_sender": "iaroslava.galimova@yandex.ru",
    "mail_receiver" : "iaroslava.galimova@yandex.ru"
}

def log(message):
    now = datetime.now()
    print(now.strftime("%d.%m.%Y %H:%M:%S"), message)

def readParams():
    if len(sys.argv) <= 1:
        log("Using default parameters")
    else:
        f = open(sys.argv[1])
        log("Reading from file " + sys.argv[1])
        content = f.readlines()
        if len(content) == 0:
            log("Using default parameters")
            return;
        for line in content:
            kv = line.split("=")
            if len(content) < 2:
                continue
            k = kv[0].strip()
            l = len(kv[0])
            v = line[l+1 : len(line)-1]
            if v == "":
                continue
            params[k] = v.strip()

def logParams():
    for k in params:
        log(k + " = " + str(params[k]))

def convert(fangle):
    value = fangle
    degDiff = 360

    fa = fangle
    degMin = int(params["degMin"])
    degMax = int(params["degMax"])
    valMin = float(params["valMin"])
    valMax = float(params["valMax"])
    dmn = degMin
    dmx = degMax
    direction = int(params["direction"])
    if direction == 1:
        if degMin < degMax:
            if fangle < degMin:
                fa = fa + 360
            dmn = dmn + 360
        degDiff = dmn - dmx

    if direction == -1:
        if degMax < degMin:
            if fangle < degMax:
                fa = fa + 360
            dmx = dmx + 360
        degDiff = dmx - dmn

    g = float(degDiff) / (valMax - valMin)
    
    if direction == 1:
        value = float(dmn - fa) / g
    if direction == -1:
        value = float(fa - dmx) / g
    return value

def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def resizePreserveAspectRatio(img, minWidth, maxWidth, minHeight, maxHeight):
    newHeight, newWidth = img.shape[:2]
    scaleFactor = 1
    while newWidth > maxWidth or newHeight > maxHeight:
        if newWidth > maxWidth:
            scaleFactor = maxWidth / newWidth
            newWidth = maxWidth
            newHeight = int(newHeight * scaleFactor)
        if newHeight > maxHeight:
            scaleFactor =maxHeight / newHeight
            newHeight = maxHeight
            newWidth = int(newWidth * scaleFactor)

    while newWidth < minWidth or newHeight < minHeight:
        if newWidth < minWidth:
            scaleFactor = minWidth / newWidth
            newWidth = minWidth
            newHeight = int(newHeight * scaleFactor)
        if newHeight < minHeight:
            scaleFactor =minHeight / newHeight
            newHeight = minHeight
            newWidth = int(newWidth * scaleFactor)

    img2 = cv2.resize(img, (newWidth, newHeight), interpolation = cv2.INTER_AREA)
    return img2

def cropAndResize(img):
    height, width = img.shape[:2]

    img3 = img.copy()

    #сначала обрежем по размеру нового кадра (newStartX,Y -- newWidth,Height)
    newStartX=int(params["xf"]) #0
    newStartY=int(params["yf"]) #0
    newWidth=int(params["wf"]) #width
    newHeight=int(params["hf"])  #height

    img3 = img3[newStartY:newStartY+newHeight, newStartX:newStartX+newWidth]
    height, width = img3.shape[:2]

    #потом уменьшим или увеличим
    img3 = resizePreserveAspectRatio(img3, 400, 600, 400, 600)
    return img3

def sendAlarm(alarm):
    if (params["mail_server"] == "" or params["mail_port"] == "" \
        or params["mail_login"] == "" or params["mail_password"] == ""\
        or params["mail_sender"] == "" or params["mail_receiver"] == ""):
        return
    server = smtplib.SMTP_SSL(params["mail_server"], int(params["mail_port"]))
    message = 'From: %s\nTo: %s\nSubject: %s\n\n%s' % (params["mail_sender"], params["mail_receiver"], alarm, alarm)

    server.login(params["mail_login"], params["mail_password"])
    server.sendmail(params["mail_sender"], params["mail_receiver"], message)
    server.quit()

'''
    Начало основной программы
'''

log("Running Gauger -- gauge analyzer")
#прочитаем параметры из файла, который передан первым аргументом в командной строке
readParams()
logParams()

#определим источник видеопотока
cap = None

if params["sourceType"] == "video":
    source = params["source"]
    log("Connecting to video source: " + source)
    cap = cv2.VideoCapture(source)
else:
    if params["sourceType"] == "webcamera":
        cap = cv2.VideoCapture(0)

#массив значений для проверки на alartLimit
values = []

#признак, что уведомление уже отправлено, чтобы не отправлять много раз одно и то же
alarmSent = False

#вечный цикл
while True:
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

    img = None
    #получим изображение из видеопотока
    log("Getting an image (frame)")
    if params["sourceType"] == "video" or params["sourceType"] == "webcamera":
        ret, img = cap.read()
        if not ret:
            log("Video input error")
            quit()
    else:
        if params["sourceType"] == "image":
            img = cv2.imread(params["source"])

    img2 = img.copy()

    #покажем основное изрбражение
    cv2.rectangle(img2, (int(params["xf"]), int(params["yf"])), (int(params["xf"])+int(params["wf"]), int(params["yf"]) + int(params["hf"])), (0,255,0), 2)
    img2 = resizePreserveAspectRatio(img2, 900, 960, 400, 540)
    cv2.imshow('Input', img2)

    #обрежем и уменьшим изображение
    img = cropAndResize(img)
    cv2.imshow('Output', img)
    
    height, width = img.shape[:2]
    
    # определить окружности: лучше это делать над черно-белым и размытым изображением
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlurred = cv2.medianBlur(gray, 25) #
    #cv2.imshow('Gray', grayBlurred)
    circles = cv2.HoughCircles(grayBlurred, cv2.HOUGH_GRADIENT, 4, int(height*0.8), param1=100, param2=100, minRadius=int(height*0.1), maxRadius=int(width*0.95))

    if circles is None:
        log("Circles not found")
        continue

    circles = np.uint16(np.around(circles))
    #for c in circles[0, :]:
        #print(c)
        #cv2.circle(img, (c[0], c[1]), c[2], (0, 255, 0), 2)
        #cv2.circle(img, (c[0], c[1]), 2,    (0, 0, 255), 3)

    #возмьмем самую вероятную окружность
    c = circles[0][0]
    cv2.circle(img, (c[0], c[1]), c[2], (0, 255, 0), 2)
    cv2.circle(img, (c[0], c[1]), 2,    (0, 0, 255), 3)
    #нарисовали окружность и её центр

    x = c[0]
    r = c[2]
    y = c[1]
    
    #отрисуем шкалу и метки
    separation = 10.0
    interval = int(360 / separation)

    p1 = np.zeros((interval, 2))
    p2 = np.zeros((interval, 2))
    p_text = np.zeros((interval, 2))
    pi = math.pi

    text_offset_x = 10
    text_offset_y = 5

    for i in range(0, interval):
        p1[i][0] = x + 0.9 * r * np.cos(separation * i * pi / 180)
        p1[i][1] = y - 0.9 * r * np.sin(separation * i * pi / 180)
        p2[i][0] = x + r * np.cos(separation * i * pi / 180)
        p2[i][1] = y - r * np.sin(separation * i * pi / 180)
        p_text[i][0] = x - text_offset_x + 1.2 * r * np.cos((separation) * i * pi / 180)
        p_text[i][1] = y + text_offset_y - 1.2 * r * np.sin((separation) * i * pi / 180)

    for i in range(0,interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), (100, 255, 60), 2)
        cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)

    #определим линии
    maxValue = 255
    th, dst2 = cv2.threshold(gray, int(params["thresh"]), maxValue, cv2.THRESH_BINARY_INV);
    #cv2.imshow('Gray', dst2)
    minLineLength = 10
    maxLineGap = 5
    lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=100, minLineLength=minLineLength, maxLineGap=0)

    #оставим только те линии, начало которых находится только в некоторой окрестности центра, 
    #а конец -- в некоторой окрестности края окружности
    final_line_list = []
    diff1LowerBound = float(params["bMin"])
    diff1UpperBound = float(params["bMax"])
    diff2LowerBound = float(params["eMin"])
    diff2UpperBound = float(params["eMax"])

    cv2.circle(img, (x, y), int(r*diff1LowerBound), (180, 180, 180), 2)
    cv2.circle(img, (x, y), int(r*diff1UpperBound), (180, 180, 180), 2)
    cv2.circle(img, (x, y), int(r*diff2LowerBound), (220, 220, 220), 2)
    cv2.circle(img, (x, y), int(r*diff2UpperBound), (220, 220, 220), 2)

    for i in range(0, len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        #cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist(x, y, x1, y1)  
            diff2 = dist(x, y, x2, y2)
            if (diff1 > diff2):
                diff1, diff2 = diff2, diff1
                x2, y2, x1, y1 = x1, y1, x2, y2
            if (((diff1 < diff1UpperBound * r) and (diff1 > diff1LowerBound * r) and (diff2 < diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                line_length = dist(x1, y1, x2, y2)
                final_line_list.append([x, y, x2, y2])


    if len(final_line_list) == 0:
        log("Lines not found")
        continue

    #если линия найдена
    x1, y1, x2, y2 = final_line_list[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    x_angle = x2 - x
    y_angle = y - y2
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    res = np.rad2deg(res)
    if x_angle >= 0 and y_angle >= 0:  #квадрант 1, res положительный
        final_angle = res
    if x_angle < 0 and y_angle >= 0:  #квадрант 2, res отрицательный
        final_angle = 180 + res
    if x_angle < 0 and y_angle < 0:  #квадрант 3, res положительный
        final_angle = 180 + res
    if x_angle >= 0 and y_angle < 0:  #квадрант 4, res отрицательный
        final_angle = 360 + res

    final_angle = int(final_angle)    
    value = int(convert(final_angle) * 100) / 100;
    log("Angle: " + str(final_angle))

    sv = str(value)
    log("Gauge value: " + sv)

    #проверим 10 последних значений на то, не превышают ли они лимит
    #обновим список значений
    if len(values) == 5:
        values.pop(0)
    values.append(value)
    
    lv = len(values) 
    print(values)

    alarm = "";
    if (lv == 5):
        #посчитаем среднее и среднеквадратичное отклонение
        avg = 0
        for v in values:
            avg = avg + v
        avg = avg / lv

        d = 0
        for v in values:
            d = d + (v - avg)**2
        d = math.sqrt(d)

        print(avg)
        print(d)
        #если значение стабильно больше лимита, отправим сообщение
        if (avg > float(params["alarmLimit"])) and (d / avg <= 0.15):
            alarm = "Alarm! Value " + str(int(avg * 100) / 100) + " > " + str(params["alarmLimit"])
            if (not alarmSent):
                sendAlarm(alarm)
                alarmSent = True
        else:
            #если значение стабильно меньше лимита, сбросим флаг отправки
            if (avg < float(params["alarmLimit"])) and (d / avg <= 0.15):
                alarmSent = False

    cv2.putText(img, '%s' %(str(final_angle)), (0, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(img, '%s' %(sv), (0, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,200), 2, cv2.LINE_AA)
    cv2.putText(img, '%s' %(alarm), (0, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (10,0,255), 2, cv2.LINE_AA)
    
    cv2.imshow('Output', img)

    time.sleep(int(params["pause"]))

cv2.destroyAllWindows()