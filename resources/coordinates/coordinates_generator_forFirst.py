import cv2 as open_cv
import numpy as np
import re
from resources.coordinates.colors import COLOR_WHITE
from resources.coordinates.colors import COLOR_BLUE

class CoordinatesGeneratorForFirst:

    def __init__(self, image, output, color,nameCam, coordinates_data):
        self.coordinates_data = coordinates_data
        self.output = output
        self.caption = "Camera "+nameCam
        self.color = color

        self.image = image
        self.image_copy = self.image.copy()
    
        self.ids = 0
        self.spaceName = "A"
        self.coordinate = (0,0)
        self.coordinates = []
        self.titles = []

        self.titlesForFirst = []
        self.idsForFirst = []
        self.coordinatesForFirst = []
        self.titlesForUpdate = []
        self.coordinatesForUpdate = []
        self.distances = []

        open_cv.namedWindow(self.caption, open_cv.WINDOW_GUI_EXPANDED)
        open_cv.setMouseCallback(self.caption, self.__mouse_callback)
        
    def regain(self):
        for item in self.coordinates_data:
            open_cv.circle(self.image, item["coordinate"], 5, (0, 255, 0), -1)     
            open_cv.putText(self.image,str(item["id"]), (item["coordinate"][0]-10,item["coordinate"][1]-10), open_cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 2)
            self.titles.append(str(item["id"]))
            self.coordinates.append(item["coordinate"])

    def autoUpdate(self):
        # câp nhật lại các vị trí  
        if len(self.titlesForFirst) > 0:
            return
        while len(self.coordinates) > 0: # khi còn các vị trí chưa được gán
            titlesForUpdate_temp = []
            coordinatesForUpdate_temp = []
            coordinatesForFirst_temp = []
            coordinatesIndex_temp =  []
            for i, item in enumerate(self.coordinatesForFirst): # duyệt quan n hàng
                titlesForUpdate_temp.append("")
                coordinatesForUpdate_temp.append([])
                coordinatesForFirst_temp.append([])
                coordinatesIndex_temp.append(0)
                minDistance = float('inf')
                index = 0
                for i2, item2 in enumerate(self.coordinates): # duyệt qua các giá trị chưa được gán
                    self.distances[i] = np.sqrt((item[0]-item2[0])**2 + (item[1]-item2[1])**2)
                    if self.distances[i] < minDistance:
                        minDistance = self.distances[i]
                        index = i2
                # dùng các biến tạm để lưu trữ tạm thời các giá trị cần update
                titlesForUpdate_temp[i] = str(self.titlesForFirst[i]+str(self.idsForFirst[i]))
                coordinatesForUpdate_temp[i] = self.coordinates[index]
                coordinatesForFirst_temp[i] = self.coordinates[index]
                coordinatesIndex_temp[i] = index

            seen = {}
            duplicates = {}
            # tìm danh sách vị trí có giá trị tọa độ trùng nhau
            for index, (num, dist) in enumerate(zip(map(tuple, coordinatesForFirst_temp), self.distances)):
                if num in seen:
                    if num in duplicates:
                        if dist < duplicates[num][1]:
                            duplicates[num] = (index, dist)
                    else:
                        if dist < seen[num][1]:
                            duplicates[num] = (index, dist)
                            seen[num] = (index, dist)
                        else:
                            duplicates[num] = seen[num]
                else:
                    seen[num] = (index, dist)
            
            unique_positions = [seen[num][0] for num in seen if num not in duplicates]
            combined_positions = sorted(unique_positions + [duplicates[num][0] for num in duplicates])
            
            # gán lại các giá trị vào các biến chính
            positions_to_remove = [coordinatesIndex_temp[i] for i in combined_positions]
            self.coordinates  = [self.coordinates[i] for i in range(len(self.coordinates)) if i not in positions_to_remove]

            for i in combined_positions:
                self.titlesForUpdate.append(titlesForUpdate_temp[i])
                self.coordinatesForUpdate.append(coordinatesForUpdate_temp[i])
                self.coordinatesForFirst[i] = coordinatesForFirst_temp[i]
                self.idsForFirst[i] += 1

        
        # cập nhật lại file
        self.output.seek(0)  # Quay lại đầu file
        self.output.truncate(0)  # Xóa toàn bộ nội dung cũ
        for i in range(len(self.titlesForUpdate)):
            self.output.write("- id: " + str(self.titlesForUpdate[i]) + "\n  coordinate: [" +
                           str(self.coordinatesForUpdate[i][0]) +", "+str(self.coordinatesForUpdate[i][1])+ "]\n")
       

    def generate(self):
        keys = ""
        self.regain()
        while True:
            open_cv.imshow(self.caption,self.image)
            key = open_cv.waitKey(0)

            if key == ord('\r'):# enter
                self.autoUpdate()
                break
            elif key >= 48 and key <=57:# chọn số cho ô
                keys += str(chr(key))
                self.ids = int(keys)
            else: #Chọn tên cho ô
                self.spaceName = chr(key)
                self.ids = 0
                keys = ""
            
        open_cv.destroyWindow(self.caption)
        
          
    def __mouse_callback(self, event, x, y, flags, params):

        if event == open_cv.EVENT_LBUTTONDOWN:
            self.coordinate=(x, y)
            self.__handle_done()

        open_cv.imshow(self.caption, self.image)

    def __handle_done(self):
        # Vẽ dấu chấm
        open_cv.circle(self.image, self.coordinate, 5, (0, 255, 0), -1)
        self.coordinatesForFirst.append(self.coordinate)
        open_cv.putText(self.image, str(self.spaceName+str(self.ids)), (self.coordinate[0]-10,self.coordinate[1]-10), open_cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 2)
        if self.spaceName == "":
            self.spaceName = "A"
        self.titlesForFirst.append(self.spaceName)
        self.idsForFirst.append(self.ids)
        self.distances.append(0)
        self.ids += 1   
        
        
        
