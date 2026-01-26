import csv
from pymavlink import mavutil

# مسار ملف .log أو .bin (بعد تحويله إذا لزم الأمر)
log_path = "D:\Saher Hassaballah\Downloads\skyhawks logs.bin"

# فتح ملف MAVLink للقراءة
mav = mavutil.mavlink_connection(log_path)


# فتح ملف CSV للكتابة
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'Latitude', 'Longitude', 'Altitude', 'Yaw'])

    # متغيرات لتخزين أحدث بيانات GPS و Yaw
    lat = None
    lon = None
    alt = None
    yaw = None

    while True:
        msg = mav.recv_match()
        if msg is None:
            break

        msg_type = msg.get_type()

        if msg_type == "GPS_RAW_INT":
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.alt / 1000
            gps_time = msg.time_usec / 1e6  # الوقت بالثواني

        elif msg_type == "ATTITUDE":
            yaw = msg.yaw
            att_time = msg._timestamp  # توقيت استقبال الرسالة

            # نكتب السطر فقط لو عندنا بيانات GPS و Yaw كاملة
            if lat is not None and lon is not None and alt is not None and yaw is not None:
                writer.writerow([att_time, lat, lon, alt, yaw])
