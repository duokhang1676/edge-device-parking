import threading
import time
import datetime
import cv2
import serial
import torch
import vlc
from gtts import gTTS
from ultralytics import YOLO
from resources.tracking.sort import Sort 
from app.utils import *
import requests
import json
import os
import random 
import globals
from resources.xg26.xg26_sensor import start_xg26_sensor
# KEEPING THINGS SIMPLE

# varible
start_detect_qr = False
start_detect_license = False
customer_type = ""
new_car = ""
car_in = False
car_out = False
id_code_in = ""
id_code_out = ""
license_car_in = ""
license_car_out = ""
mp3_url = "resources/mp3/"
ClOUD_SERVER_URL = 'https://parking-cloud-server.onrender.com/api/'
#ClOUD_SERVER_URL = 'http://127.0.0.1:5000/api/'
parking_id = 'parking_001'
# Danh sách Dictionary chứa các xe đã đỗ trong bãi (Hiện tại lưu trữ tạm thời trong RAM, cần lưu vào DB sqlite)
parked_vehicles = []
available_list = []
occupied_list = []
occupied_license_list = []
arduino_update = False
direction = []
slot_table = []
wrong_slot = [] 
qr_thread = True
license_thread = True
update_coordinate_arduino = False
# Danh sách Dictionary các biển số xe đã được đăng ký
# sài tạm
registered_vehicles = [{
    "parking_id": parking_id,
    "user_id": "01234",
    "license_plate":  "30G-49344",
},
{
    "parking_id": parking_id,
    "user_id": "01234",
    "license_plate":  "30G-53507",
},
{    
    "parking_id": parking_id,
    "user_id": "01234",
    "license_plate":  "30K-55055",
}
]

CLOUD_NAME = "dcs6zqppp"
UPLOAD_PRESET = "parking-data"

TRACKING_CAMERA_ID = 1
QR_CAMERA_ID = "/dev/video1"
LICENSE_CAMERA_ID = "/dev/video2"
#

def get_coordinates(parking_id, camera_id):
    url = f'{ClOUD_SERVER_URL+"coordinates/"}{parking_id}/{camera_id}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()[0]
    else:
        return None

def update_coordinates(parking_id, camera_id, data):
    url = f'{ClOUD_SERVER_URL+"coordinates/"}update/{parking_id}/{camera_id}'
    response = requests.put(url, json=data)
    return response.status_code == 200

def insert_coordinates(data):
    url = f'{ClOUD_SERVER_URL+"coordinates/"}add'
    response = requests.post(url, json=data)
    return response.status_code == 201

def insert_parked_vehicle(data):
    url = f'{ClOUD_SERVER_URL+"parked_vehicles/"}add_vehicle'
    response = requests.post(url, json=data)
    return response.status_code == 200

def remove_parked_vehicle(data):
    url = f'{ClOUD_SERVER_URL+"parked_vehicles/"}remove_vehicle'
    response = requests.delete(url, json=data)
    if response != 200:
        print(response)
    return response.status_code == 200


def update_parked_vehicle(data):
    url = f'{ClOUD_SERVER_URL+"parked_vehicles/"}update_vehicle'
    response = requests.put(url, json=data)
    return response.status_code == 200

def update_parked_vehicle_list(data):
    url = f'{ClOUD_SERVER_URL+"parked_vehicles/"}update_vehicle_list'
    response = requests.put(url, json=data)
    return response.status_code == 200

def update_parking_lot(data):
    url = f'{ClOUD_SERVER_URL+"parking_slots/"}update_parking_slots'
    response = requests.post(url, json=data)
    return response.status_code == 200

def update_environment(data):
    url = f'{ClOUD_SERVER_URL+"environments/"}update_environment'
    response = requests.post(url, json=data)
    return response.status_code == 200

def insert_history(data):
    url = f'{ClOUD_SERVER_URL+"histories/"}'
    response = requests.post(url, json=data)
    return response.status_code == 201

  
# Detect, tracking xe, hiển thị kết quả: danh sách chổ trống, chổ chiếm, danh sách parked_vehicle
def tracking_car():
    global new_car, car_in, parking_id, parked_vehicles, available_list, occupied_list, occupied_license_list, direction, slot_table, wrong_slot, arduino_update, wrong_slot, update_coordinate_arduino
    coordinates_data = read_yaml("resources/coordinates/data/coordinates_"+str(0)+'.yml')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy trên: {device}")
    model = YOLO("resources/models/detect-car-yolov8n-v2.engine")
    # model = YOLO("resources/models/detect-car-yolov8n-v2.pt")
    # Set log level to ERROR
    model.overrides['verbose'] = False
    tracker = Sort(max_age=100, iou_threshold=0.1, min_hits = 5)
    #cap = cv2.VideoCapture("test_data/video/4.mp4")
    gst_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=640, height=640, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    time.sleep(5)
    ret, frame = cap.read()

    # Load camera image to cloudserver
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()
        # Upload image to Cloudinary
        response = requests.post(
            f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/image/upload",
            files={"file": img_bytes},
            data={"upload_preset": UPLOAD_PRESET}
        )
        if response.status_code == 200:
            print("Đã tải hình ảnh lên Cloudinary")
            image_url = response.json()["secure_url"]
            print(image_url)
            cam = get_coordinates(parking_id, str(TRACKING_CAMERA_ID))
            if cam is not None:
                cam['image_url'] = image_url
                if update_coordinates(parking_id, str(TRACKING_CAMERA_ID), cam):
                    print("Đã tải hình ảnh lên Cloud Server")
                else:
                    print("Lỗi khi tải hình ảnh lên Cloud Server")
                # Download coordinates from cloud server
                if cam.get('coordinates_list') is not None:
                    coordinates_data = cam.get('coordinates_list')
                    write_yaml_file("resources/coordinates/data/coordinates_"+str(0)+'.yml', coordinates_data)
                    print("Đã tải tọa độ từ Cloud Server\n")
                else:
                    print("Không có tọa độ nào trên Cloud Server")
            else:
                if insert_coordinates({
                    'parking_id': parking_id,
                    'camera_id': str(TRACKING_CAMERA_ID),
                    'image_url': image_url,
                    'coordinates_list': []
                }):
                    print("Đã tạo mới coordinates trên Cloud Server")
                else:
                    print("Lỗi khi tạo mới coordinates trên Cloud Server")
        else:
            print("Lỗi khi tải hình ảnh lên Cloudinary:", response.status_code)
######################################
    track_licenses = [] # danh sách biển số xe đang ở trong bãi xe đang được theo dõi
    track_ids_full = [] # danh sách tất cả ids track được 
    license_ids_full = [] # danh sách license id đã được theo dõi tương ứng với track_ids_full
    pre_track_ids = [] # danh sách id track được ở vòng lặp trước
    pre_hidden_ids = [] # danh sách vị trí có xe ở vòng lặp trước
    last_id = 0 # id được trackq gần nhất
    
    occoccupied_delay = 0
    out_delay = 0 # biến delay khi xe bị mất track, khi đạt ngưỡng xác nhận là đã xe ra
    hidden_ids_map_track_licenses = [] # Danh sách chứa các id được tracking tương ứng với các vị trí bị che khuất
    # tính fps
    start_time = time.time()
    frame_count = 0
    change_count = 0
    count = 0
    while True:
        if car_in or car_out:# nhường GPU cho detect_license
            time.sleep(1)
            start_time = time.time()
            frame_count = 0
            continue
        count += 1
        ret, frame = cap.read()
        if not ret:
            print("Detect car frame is none!")
            continue

# tính fps
        # Tăng frame count
        frame_count += 1
        # Tính thời gian đã trôi qua
        elapsed_time = time.time() - start_time
        # Tính FPS
        if elapsed_time > 0:
            fps = int(frame_count / elapsed_time)
#
        # Danh sách các tọa độ và id của xe đã track được
        detected_boxes, track_ids = tracking_objects(tracker, model, frame, confidence_threshold = 0.6, device=device)
# xe đi vào
        if len(track_ids) != 0:
            # Phát hiện xe mới vào
            if track_ids[-1] > last_id:
                # danh sách các id mới được track
                new_car_list = [x for x in track_ids if x > last_id]
                for i in new_car_list:
                    last_id = i
                    # xe vào thật có license
                    if new_car != "":   
                        if last_id not in track_ids_full:
                            track_ids_full.append(last_id)
                            license_ids_full.append(new_car)
                        new_car = ""
                    # xe vào do đặt vào hoặc detect sai gán license là id track được
                    else:
                        if last_id not in track_ids_full:
                            track_ids_full.append(last_id)
                            license_ids_full.append(last_id)
            
# xe đi ra
        track_licenses = [license_ids_full[track_ids_full.index(track_id)] for track_id in track_ids]
      
# so khớp vị trí đỗ
        # danh sách vị trí có xe, không có xe và tracking id tại các vị trí đỗ có xe
        hidden_ids, visible_ids, hidden_ids_map_track_licenses = check_occlusion(coordinates_data, detected_boxes, track_licenses)
        
        # tạo delay cho xe tại vị trí đỗ để xác nhận chắc chắn xe đã đỗ hoặc rời đi tránh trường hợp chỉ chạy qua
        if hidden_ids != pre_hidden_ids:
            occoccupied_delay += 1
        else:
            pre_hidden_ids = hidden_ids
            occoccupied_delay = 0

# Xác định có thay đổi vị trí đỗ xe trong ds khi danh sách tọa độ đã không thay đổi trong 100 frame và kích thước biển số và id map với nhau
        if occoccupied_delay >= 100: 
            print("Có sự thay đổi vị trí đỗ xe")
            pre_hidden_ids = hidden_ids
            occoccupied_delay = 0

            available_list = visible_ids
            occupied_list = hidden_ids
            occupied_license_list = hidden_ids_map_track_licenses

            #gửi đến arduino 
            update_coordinate_arduino = True
            arduino_update = True
            # gợi ý hướng vào (làm tạm) fix
            direction = find_min_slots(available_list)
            print("derection", direction)
            # tính toán số lượng  ô đỗ
            slot_table = count_groups(occupied_list)
            print("slot table",slot_table)

            # Tạo parked-vehicles
            
            print("occupied",hidden_ids)
            print("license occupied",hidden_ids_map_track_licenses)
            print("")

            # Gán slot name và num slot cho parked vehicle
            temp = False
            for i, parked_vehicle in enumerate(parked_vehicles):
                numslot = 0
                for j, license in enumerate(occupied_license_list):
                    if parked_vehicle["license_plate"] == license:
                        numslot += 1
                        parked_vehicles[i]["slot_name"] = occupied_list[j] # fix
                        parked_vehicles[i]["num_slot"] = numslot
                        temp = True
            # cập nhật parked_vehicle
            if temp:
                data = {
                    'parking_id': parking_id,
                    'list': parked_vehicles
                }
                if update_parked_vehicle_list(data):
                    print("câp nhật parked vehicles thành công")
                else:
                    print("cập nhật parked vehicles thất bại")
            # cập nhật parking lot server
            data = {
                'parking_id': parking_id,
                'available_list': available_list,
                'occupied_list': occupied_list,
                'occupied_license_list': occupied_license_list
            }
            if update_parking_lot(data):
                print("cập nhật parkingslot thành công")
            else:
                print("cập nhật parkingslot thất bại")

            # cảnh báo đỗ sai vị trí
            wrong_slot = []
            for parked_vehicle in parked_vehicles:
                if parked_vehicle["num_slot"] > 1:
                    wrong_slot.append(parked_vehicle['license_plate'])
            #print("wrong slot", wrong_slot)
            if wrong_slot != []:
                text = "Xe có biển kiểm xoát, "+ str(wrong_slot) + ", đậu xe không đúng vị trí vui lòng di chuyển, xin cảm ơn!"
                threading.Thread(target=speech_text, args=(text,)).start()

            


# trực quan
    ##########
        if count % 200 == 0:
            #print("IDs:", track_ids_full)
            #print("Tất biển số xe đã vào:", license_ids_full)
            print("-------")
            print("license:", track_licenses)
            #print("ID hiện tại:",track_ids)
            #print("Tọa độ:",detected_boxes)
            print("Các vị trí đã có xe đỗ: ", hidden_ids)
            print("Các xe đã đỗ tương ứng: ", hidden_ids_map_track_licenses)
            #print("Số lần thay đổi vị trí đỗ xe: ", change_count)
            print("-------")

        draw_points_and_ids(frame, coordinates_data, hidden_ids, track_ids, detected_boxes, track_licenses, fps, hidden_ids_map_track_licenses)
       
        # Hiển thị
        # cv2.imshow("Tracking camera", frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break
        ###########


    # cap.release()
    # cv2.destroyAllWindows()

def count_groups(a):
    group_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    for slot in a:
        letter = slot[0]
        if letter in group_counts:
            group_counts[letter] += 1

    return [group_counts[k] for k in ['A', 'B', 'C', 'D']]

def find_min_slots(a):
    groups = {'A': float('inf'), 'B': float('inf'), 'C': float('inf'), 'D': float('inf')}

    for slot in a:
        letter = slot[0]
        number = int(slot[1:])

        if letter in groups:
            groups[letter] = min(groups[letter], number)

    # Convert to desired format as a list with empty strings for missing values
    result = [f"{k}{groups[k]}" if groups[k] != float('inf') else "" for k in ['A', 'B', 'C', 'D']]

    return result

# Giao tiếp serial với micro controller
def connect_sensor():
   # Thiết lập cổng Serial (kiểm tra cổng COM trong Device Manager)
    port = "/dev/ttyUSB0"  
    baudrate = 9600
    global car_in, car_out, id_code_in, id_code_out, license_car_in, license_car_out, new_car, customer_type, parked_vehicles, parking_id, registered_vehicles, update_coordinate_arduino, direction, slot_table, qr_thread, license_thread, start_detect_qr, start_detect_license
    # Thiết lập cổng Serial (kiểm tra cổng COM trong Device Manager)
    try:
        # Kết nối Serial
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Kết nối thành công với {port}")
        threading.Thread(target=play_sound, args=('resources/mp3/arduino_connection.mp3',)).start()
        # API arduino
        # 0: barie_in:0 (đóng barie vào)
        # 1: barie_in:1 (mở barie vào)
        # 2: barie_out:0 (đóng barie ra)
        # 3: barie_out:1 (mở barie ra)
        # 4: update_slot (cập nhật lại số chỗ trống và hướng đi)
        while True:
            time.sleep(1)
            if update_coordinate_arduino:
                print("Cập nhật direction")
                update_coordinate_arduino = False
                # ser.write(("4" + '\n').encode('utf-8'))
                text = f" {str(direction[3])}-{str(direction[2])}    {str(direction[1])}-{str(direction[0])}   "
                ser.write((text + '\n').encode('utf-8'))
                sum_slot = sum(slot_table)
                text = str(slot_table[0])+","+str(slot_table[1])+","+str(slot_table[2])+","+str(slot_table[3])+","+str(sum_slot)
                ser.write((text + '\n').encode('utf-8'))
            # Xe đi vào
            if car_in and id_code_in != "" and license_car_in != "":
                # Kiểm tra customer_type
                if customer_type == "customer":
                    # Kiểm tra biển số xe đã được đăng ký chưa
                    user_valid = False
                    license_valid = False
                    for vehicle in registered_vehicles:
                        if vehicle['user_id'] == id_code_in:
                            user_valid = True
                            if vehicle['license_plate'] == license_car_in:
                                license_valid = True
                                # mở barie
                                new_car = license_car_in
                                print("new car: ", new_car)
                                print("Xe vào bãi đỗ")
                                ser.write(("1" + '\n').encode('utf-8'))
                                threading.Thread(target=play_sound, args=('resources/mp3/xin-moi-vao.mp3',)).start()
                                # Tạo mới parked_vehicle
                                time_in = datetime.datetime.utcnow()+ datetime.timedelta(hours=7) 
                                parked_vehicle = {
                                    'user_id': id_code_in,
                                    'customer_type': customer_type,
                                    'time_in': time_in.isoformat(),
                                    'license_plate': license_car_in,
                                    'slot_name': '',
                                    'num_slot': 0
                                }
                                # Thêm parked_vehicle vào danh sách
                                parked_vehicles.append(parked_vehicle)
                                # Gửi dữ liệu lên server
                                # data = {
                                #     'parking_id': parking_id,
                                #     'vehicle': parked_vehicle
                                # }
                                # if insert_parked_vehicle(data):
                                #     print("Gửi parked_vehicle thành công!")
                                # else:
                                #     print("Gửi parked_vehicle không thành công!")
                                break
                    car_in = False
                    if not user_valid:
                        print("Khách hàng không hợp lệ")
                        threading.Thread(target=play_sound, args=('resources/mp3/khach-hang-khong-hop-le.mp3',)).start()
                    elif not license_valid:
                        print("Biển số không hợp lệ")
                        threading.Thread(target=play_sound, args=('resources/mp3/bien-so-khong-hop-le.mp3',)).start()

                # else: # Khách vãng lai
                #     # mở barie
                #     new_car = license_car_in
                #     print("Xe vào bãi đỗ")
                #     print(new_car)
                #     threading.Thread(target=play_sound, args=('resources/mp3/xin-moi-vao.mp3',)).start()
                #     ser.write(("1" + '\n').encode('utf-8'))
                #     # Tạo mới parked_vehicle
                #     time_in = datetime.datetime.utcnow()+ datetime.timedelta(hours=7) 
                #     parked_vehicle = {
                #                     'user_id': id_code_in,
                #                     'customer_type': customer_type,
                #                     'time_in': time_in.isoformat(),
                #                     'license_plate': license_car_in,
                #                     'slot_name': '',
                #                     'num_slot': 0
                #                 }
                #     # Thêm parked_vehicle vào danh sách
                #     parked_vehicles.append(parked_vehicle)
                #     # Gửi dữ liệu lên server
                #     # data = {
                #     #         'parking_id': parking_id,
                #     #         'vehicle': parked_vehicle
                #     #     }
                #     # if insert_parked_vehicle(data):
                #     #     print("Gửi parked_vehicle thành công!")
                #     # else:
                #     #     print("Gửi parked_vehicle không thành công!")
            # Xe đi ra
            if car_out and id_code_out != "" and license_car_out != "":
                # Kiểm tra id_code_out và license có trong danh sách parked_vehicles không
                user_valid = False
                license_valid = False
                for vehicle in parked_vehicles:
                    if vehicle['user_id'] == id_code_out:
                        user_valid = True
                        if vehicle['license_plate'] == license_car_out:
                            license_valid = True
                            # Nếu có, thì xe đã được đỗ và có thể ra
                            print("Xe ra khỏi bãi đỗ")
                            threading.Thread(target=play_sound, args=('resources/mp3/tam-biet-quy-khach.mp3',)).start()
                            # Mở barie ra
                            ser.write(("3" + '\n').encode('utf-8'))
                            # Tạo history
                            user_id = ""
                            total_price = 0
                            time_in = vehicle['time_in']
                            time_out = datetime.datetime.utcnow()+ datetime.timedelta(hours=7)
                            
                            # start_time_parsed = datetime.fromisoformat(time_in)

                            # # Hiệu thời gian
                            # elapsed = time_out - start_time_parsed

                            # # Đổi thành số giờ (dưới dạng float)
                            # parking_time = elapsed.total_seconds() / 3600
                            parking_time = 0.1
                            if vehicle['customer_type'] == "customer":
                                user_id = vehicle['user_id']
                                total_price = 0
                            else:
                                user_id = "guest"
                                # Tính toán giá tiền dựa trên thời gian đỗ xe
                                # 5 giời đầu tiên là 50k, từ giờ thứ 6 là 10k
                                total_price = 50000 + (int(parking_time.split(':')[0]) - 5) * 10000  
                            history = {
                                'parking_id': parking_id,
                                'user_id': user_id,
                                'license_plate': vehicle['license_plate'],
                                'time_in': time_in,
                                'time_out': time_out.isoformat(),
                                'parking_time': parking_time,
                                'total_price': total_price,
                            }
                            # Gửi dữ liệu lên server
                            if insert_history(history):
                                print("Gửi history thành công!")
                            else:
                                print("Gửi history không thành công!")

                            # Xóa parked_vehicle khỏi danh sách
                            parked_vehicles.remove(vehicle)
                            # Xóa trên server
                            if remove_parked_vehicle(vehicle):
                                print("Xóa parked-vehicle thành công")
                            else:
                                print("Xóa parked-vehicle không thành công")

                            break
                car_out = False
                if not user_valid:
                    print("Khách hàng không hợp lệ")
                    threading.Thread(target=play_sound, args=('resources/mp3/khach-hang-khong-hop-le.mp3',)).start()
                elif not license_valid:
                    print("Biển số không hợp lệ")
                    threading.Thread(target=play_sound, args=('resources/mp3/bien-so-khong-hop-le.mp3',)).start()

            # Danh sách dữ liệu từ Arduino
            if ser.in_waiting > 0:
                for _ in range(ser.in_waiting):
                    # Đọc dữ liệu từ Arduino
                    data = ser.readline().decode('utf-8').strip()
                    # Kiểm tra định dạng và tách key, value
                    if ":" in data:
                        key, value = data.split(":", 1)
                        print(f"Key: {key}, Value: {value}")
                        # Xe vào
                        if key == "car_in":
                            if value == "1":
                                car_in = True
                                #if license_thread:
                                #license_thread = False
                                start_detect_license = True
                                if qr_thread:
                                    qr_thread = False
                                    start_detect_qr = True
                            else:
                                car_in = False
                                id_code_in = ""
                                license_car_in = ""
                                # đóng barie vào
                                ser.write(("0" + '\n').encode('utf-8'))
                        # Xe ra
                        elif key == "car_out":
                            if value == "1":
                                car_out = True  
                                if license_thread:
                                    license_thread = False
                                    start_detect_license = True
                                if qr_thread:
                                    qr_thread = False
                                    start_detect_qr = True
                            else:
                                car_out = False
                                id_code_out = ""
                                license_car_out = ""
                                # đóng barie ra
                                ser.write(("2" + '\n').encode('utf-8'))
                                
                        # # Khách vãng lai
                        # elif key == "rfid":
                        #     customer_type = "guest"
                        #     id_code_in = value

                        # elif key == "env":
                        #     env = json.loads(value)
                        #     data = {
                        #         'parking_id': parking_id,
                        #         'temperature': env[0],
                        #         'humidity': env[1],
                        #         'light': env[2]   
                        #     }
                        #     update_environment(data)

            # Xe đi vào
            
            # Xe đi ra
            


    except serial.SerialException:
        print(f"Không thể kết nối tới {port}")
    except KeyboardInterrupt:
        print("\nĐã thoát chương trình.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Đã đóng cổng Serial.")

# Detect QR
def detect_QR():
    global car_in, car_out, id_code_in, id_code_out, customer_type, qr_thread, start_detect_qr
    cap = cv2.VideoCapture(QR_CAMERA_ID)
    # Khởi tạo QRCodeDetector
    qr_decoder = cv2.QRCodeDetector()
    while True:
        if not start_detect_qr:
            time.sleep(1)
        else:
            # Nếu xe vào hoặc ra và chưa có mã QR
            if (car_in and id_code_in == "") or (car_out and id_code_out == ""):
                ret, frame = cap.read()
                if frame is None:
                    print("Camera QR lỗi!")
                    continue
                # Giải mã mã QR
                # cv2.imshow("Detect QR Camera",frame)
                # if cv2.waitKey(1) == ord('q'):
                #     break
                print("detect QR")
                qr_code, points, _ = qr_decoder.detectAndDecode(frame)
                qr_code = "01234"
                time.sleep(random.randint(3, 5))
                if True:
                # if points is not None:
                    if qr_code:
                        print("detected qr")
                        if car_in:
                            id_code_in = qr_code
                            customer_type = "customer"
                        else:
                            id_code_out = qr_code
                        print(qr_code)
                        threading.Thread(target=play_sound, args=(mp3_url+'scan.mp3',)).start()
                        # cv2.destroyWindow("Detect QR Camera")
                        # cap.release()
                        qr_thread = True
                        start_detect_qr = False
                        # break
            else:
                # cv2.destroyWindow("Detect QR Camera")
                # cap.release()
                qr_thread = True
                start_detect_qr = False
                # break

    
    
# Detect license
import resources.license_plate_recognition.function.helper as helper
import resources.license_plate_recognition.function.utils_rotate as utils_rotate

def detect_license():
    # Load model YOLO tùy chỉnh để phát hiện biển số xe
    yolo_LP_detect = torch.hub.load('resources/license_plate_recognition/yolov5', 'custom', path='resources/license_plate_recognition/model/LP_detector_nano_61.pt', force_reload=True, source='local')#.to('cpu')
    # Load model YOLO tùy chỉnh để nhận diện chữ trên biển số xe
    yolo_license_plate = torch.hub.load('resources/license_plate_recognition/yolov5', 'custom', path='resources/license_plate_recognition/model/LP_ocr_nano_62.pt', force_reload=True, source='local')#.to('cpu')
    # Đặt ngưỡng độ tự tin (confidence threshold) để nhận diện biển số xe
    yolo_license_plate.conf = 0.60
    cap = cv2.VideoCapture(LICENSE_CAMERA_ID)
    global car_in, car_out, license_car_in, license_car_out, license_thread, start_detect_license
    lp_temp = ""
    delay = 0
    while(True):
        if not start_detect_license:
            time.sleep(1)
        else:
            # Nếu xe vào hoặc ra và chưa có biển số
            if (car_in and license_car_in == "") or (car_out and license_car_out == ""):
                ret, frame = cap.read()
                if frame is None:
                    print("license frame is none!")
                    continue
                print("detect license")
                # cv2.imshow("Detect License Camera", frame)
                # if cv2.waitKey(1) == ord('q'):
                #     break
                # Sử dụng mô hình YOLO để phát hiện biển số xe trong khung hình
                plates = yolo_LP_detect(frame, size=640)
                # Lấy danh sách các biển số xe được phát hiện (tọa độ bounding box)
                list_plates = plates.pandas().xyxy[0].values.tolist()
                # Tạo một tập hợp để lưu các biển số xe đã đọc
                list_read_plates = []
                # Lặp qua tất cả các biển số xe được phát hiện
                for plate in list_plates:
                    x = int(plate[0]) # Lấy tọa độ xmin của bounding box
                    y = int(plate[1]) # Lấy tọa độ ymin của bounding box
                    w = int(plate[2] - plate[0]) # Tính toán chiều rộng của bounding box
                    h = int(plate[3] - plate[1]) # Tính toán chiều cao của bounding box
                    # Cắt hình ảnh của biển số xe từ khung hình
                    crop_img = frame[y:y+h, x:x+w]
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, 0, 0))
                    # Nếu biển số được nhận diện không phải "unknown"
                    if lp != "unknown":
                        # Thêm biển số đã nhận diện vào danh sách
                        list_read_plates.append(lp)
                # Xác nhận biển số xe có chính xác không bằng cách kiểm tra giá trị trong 3 lần detect có giống nhau không
                if len(list_read_plates) == 1: # chỉ lấy 1 biển số xe, trường hợp có nhiều hơn 1 biến số xe hoặc không có cái nào thì bỏ qua
                    if list_read_plates[0] == lp_temp:
                        delay += 1
                    else:
                        delay = 0
                        lp_temp = list_read_plates[0]
                    if delay >= 5:
                        delay = 0
                        # Nếu xe vào
                        if car_in:
                            license_car_in = lp_temp
                        # Nếu xe ra
                        else:
                            license_car_out = lp_temp
                        print(lp_temp)
                        threading.Thread(target=play_sound, args=('resources/mp3/scan.mp3',)).start()
                        # cv2.destroyWindow("Detect License Camera")
                        # cap.release()
                        license_thread = True
                        start_detect_license = False
            else:
                # cv2.destroyWindow("Detect License Camera")
                # cap.release()
                license_thread = True
                start_detect_license = False
        
def speech_text(text):
    # Tạo tts
    tts = gTTS(text=text, lang='vi', slow=False)
    path = mp3_url + 'temp.mp3'
    tts.save(path)
    player = vlc.MediaPlayer(path)
    player.play()

def play_sound(path):
    player = vlc.MediaPlayer(path)
    player.play()

def send_env():
    global parking_id
    while True:
        if globals.get_humidity() is not None:
            data = {
                'parking_id': parking_id,
                'temperature': globals.get_temperature(),
                'humidity': globals.get_humidity(),
                'light': globals.get_light()   
            }
            print("update env")
            update_environment(data)
        time.sleep(5)

def main():
    # Tạo luồng
    threading.Thread(target=play_sound, args=('resources/mp3/start-program.mp3',)).start()

    thread1 = threading.Thread(target=tracking_car)
    thread2 = threading.Thread(target=connect_sensor)
    thread3 = threading.Thread(target=detect_QR)
    thread4 = threading.Thread(target=detect_license)
    thread5 = threading.Thread(target=start_xg26_sensor)
    thread6 = threading.Thread(target=send_env)

    # Bắt đầu luồng
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()

    # Chờ tất cả luồng kết thúc
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    
if __name__ == "__main__":
    main()