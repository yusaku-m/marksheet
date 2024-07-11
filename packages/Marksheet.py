from .Image import Image

import os
import cv2
import numpy as np
import pandas as pd


class Marksheet():
    """

    等間隔のマークシート。青色のみ使用可能。

    path:スキャンした画像のパス

    row:用紙全体の行数

    column:用紙全体の列数
    
    marker_position:黒で塗りつぶしたマーカーのリスト。(r,c)のタプルの形で行列を指定，方向等の調整に使用

    direction:horizontalで横向き，verticalで縦向き

    binary_threshold:２値化時の閾値マーク部分が真っ黒になるように調整

    mark_threshold:マークとの合致率の閾値

    monitor:Trueで動作ログ表示

    """
    def __init__(self, path, row, column,
                 marker_positions, 
                 direction = "horizontal", 
                 binary_threshold = 240,
                 mark_threshold = 0.5,
                 monitor = False):
        self.sheet = None
        self.path = path
        self.row = row
        self.column = column
        self.marker_positions = marker_positions
        self.direction = direction
        self.binary_threshold = binary_threshold
        self.mark_threshold = mark_threshold
        self.monitor = monitor
        self.grids =[]
        self.result = []

        if self.monitor is True:
            print("init")

    def read(self):
        if self.monitor is True:
            print("open", self.path)
        self.sheet = Image(self.path)
        
        if self.monitor is True:
            print("read image shape:", self.sheet.img.shape)
            self.sheet.show("raw")
    
    def threshold_check(self):

        ret, bufimage = cv2.threshold(self.sheet.img[:,:,0], self.binary_threshold, 255, cv2.THRESH_BINARY)
        image = Image(img=bufimage)
        image.show("check")

    def make_reference(self):
        # 画像のサイズを設定
        height = self.sheet.img.shape[0]
        width = self.sheet.img.shape[1]

        # マーク１ブロックのサイズ
        req_height = height / self.row
        req_width = width / self.column
        
        # 白い画像を作成
        image = np.ones((int(height - req_height * 2), 
                         int(width - req_width * 2)), 
                         dtype=np.uint8) * 255

        # 指定された位置に黒い正方形を描画
        for row, col in self.marker_positions:
            top_left = (int((col-2) * req_width), 
                        int((row-2) * req_height))
            
            bottom_right = (int((col-1) * req_width), 
                            int((row-1) * req_height))
            
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)

        image = Image(img=image)

        if self.monitor is True:
            print("make reference image shape:", image.img.shape, "reqtangle h,w:", req_height, req_width)
            image.show("reference")

        return image
    
    def rotation(self):
        """
        用紙の向きを調整する。
        """
        height = self.sheet.img.shape[0]
        width = self.sheet.img.shape[1]

        if (self.direction == "horizontal") and (width < height):
            self.sheet.img = cv2.rotate(self.sheet.img, cv2.ROTATE_90_CLOCKWISE)
            if self.monitor is True:
                print(f"rotated to horizontal")

        if (self.direction == "vertical") and (height < width):
            self.sheet.img = cv2.rotate(self.sheet.img, cv2.ROTATE_90_CLOCKWISE)
            if self.monitor is True:
                print(f"rotated to vertical")


        # 方向判定（上下）
        best_match = 0
        
        reference = self.make_reference().img

        for j in range(2):
            result_img = cv2.matchTemplate(self.sheet.img[:,:,0], reference, cv2.TM_CCOEFF_NORMED)
            __min_val, max_val, __min_loc, __max_loc = cv2.minMaxLoc(result_img)
            #print(i, max_val)
            if max_val > best_match:
                best_match = max_val
                best_rotation = j

            #回転してもう一度確認
            reference = cv2.rotate(reference, cv2.ROTATE_180)
            
            if self.monitor is True:
                print(f"rotation: {j}, max_val: {max_val}")

        # 結果をもとに回転
        if self.monitor is True:
            print(f"best_rotation: {best_rotation}")

        if best_rotation == 1:
            self.sheet.img = cv2.rotate(self.sheet.img, cv2.ROTATE_180)

        if self.monitor is True:
            print("rotation complete")
            self.sheet.show("rotation")
        

    def aliment(self, scale_range = (0.98,1.05), angle_range = (-1,1), num = 3, times = 3, black = 100):
        """
        マーク部分を抽出する。
        times: 角度，倍率の精度（10^-times乗まで絞る）
        """
        if self.monitor is True:
            print("aliment start")

        image = self.sheet.img

        height = self.sheet.img.shape[0]
        width = self.sheet.img.shape[1]
        center = (int(width/2), int(height/2))

        reference = self.make_reference().img
        best_match = 0

        hscale_range = scale_range
        vscale_range = scale_range
        
        for t in range(times):
            hscales = np.linspace(hscale_range[0], hscale_range[1], num)
            vscales = np.linspace(vscale_range[0], vscale_range[1], num)
            angles = np.linspace(angle_range[0], angle_range[1], num)

            for angle in angles:
                for hscale in hscales:
                    for vscale in vscales:
                        #getRotationMatrix2D関数を使用
                        trans = cv2.getRotationMatrix2D(center, angle , 1)

                        #アフィン変換
                        bufimage = cv2.warpAffine(image, trans, (width,height), borderValue=(255, 255, 255))
                        bufimage = cv2.resize(bufimage, (int(width * hscale), int(height * vscale)))

                        #テンプレートマッチ
                        ret, bufimage = cv2.threshold(bufimage[:,:,0], black, 255, cv2.THRESH_BINARY)
                        result_img = cv2.matchTemplate(bufimage, reference, cv2.TM_CCOEFF_NORMED)
                        __min_val, max_val, __min_loc, max_loc = cv2.minMaxLoc(result_img)

                        if max_val > best_match:
                            best_match = max_val
                            best_loc = max_loc
                            best_angle = angle
                            best_hscale = hscale
                            best_vscale = vscale

            hscale_width = (hscale_range[1] - hscale_range[0]) / num / 2
            vscale_width = (vscale_range[1] - vscale_range[0]) / num / 2
            angle_width = (angle_range[1] - angle_range[0]) / num / 2
            
            hscale_range = (best_hscale - hscale_width, best_hscale + hscale_width)
            vscale_range = (best_vscale - vscale_width, best_vscale + vscale_width)
            angle_range = (best_angle - angle_width, best_angle + angle_width)
            
        #ベストな状態を抽出
        trans = cv2.getRotationMatrix2D(center, best_angle , 1)
        #アフィン変換
        self.sheet.img = cv2.warpAffine(image, trans, (width,height), borderValue=(255, 255, 255))
        self.sheet.img = cv2.resize(self.sheet.img, (int(width * best_hscale), int(height * best_vscale)))
        #トリミング
        self.sheet.img = self.sheet.img[best_loc[1]:, best_loc[0]:, :]
        self.sheet.img = self.sheet.img[:reference.shape[0], :reference.shape[1], :]
        self.row -= 2; self.column -= 2

        if self.monitor is True:
            print("aliment complete, best match:",best_match, "angle:", best_angle, "scale:", best_hscale, best_vscale, "shape", self.sheet.img.shape)
            self.sheet.show("aliment")
            ret, img_thresh = cv2.threshold(self.sheet.img[:,:,0], black, 255, cv2.THRESH_BINARY)
            image = Image(img=img_thresh)
            image.show("aliment except b")

    def draw_grid(self):

        # 画像のサイズを設定
        height = self.sheet.img.shape[0]
        width = self.sheet.img.shape[1]

        # マーク１ブロックのサイズ
        req_height = height / self.row
        req_width = width / self.column

        # 白い画像を作成
        image = self.sheet.img

        # グリッドの間隔
        grid_size = 40

        # 垂直線を描画
        for x in np.linspace(0, width, self.column + 1):
            cv2.line(image, (int(x), 0), (int(x), height), (0, 0, 255), 1)

        # 水平線を描画
        for y in np.linspace(0, height, self.row + 1):
            cv2.line(image, (0, int(y)), (width, int(y)), (0, 0, 255), 1)

        image = Image(img=image)

        if self.monitor is True:
            print("make reference image shape:", image.img.shape, "reqtangle h,w:", req_height, req_width)
            image.show("grid")

        return image
    
    def mark_check(self):
        """
        マークを読み取って結果を配列で返す
        """
        result = []
        for line in self.grids:
            row = []
            for grid in line:
                grid.make_mark()
                row.append(grid.mark_check())
            result.append(row)

        if self.monitor is True:
            print("mark check")
            self.sheet.show("mark check")

        self.result = pd.DataFrame(np.array(result))

    def make_grids(self):
        # 画像のサイズを設定
        height = self.sheet.img.shape[0]
        width = self.sheet.img.shape[1]

        # マーク１ブロックのサイズ
        req_height = height / self.row
        req_width = width / self.column

        self.grids =[]
        for r in range(self.row):
            line =[]
            for c in range(self.column):
                img = self.sheet.img[int(r * req_height) : int((r+1) * req_height), 
                                     int(c * req_width) : int((c+1) * req_width), :]
                
                line.append(Markgrid(img, self.binary_threshold, self.mark_threshold, monitor=self.monitor))
            self.grids.append(line)
    
    def save(self):
        filename = os.path.splitext(os.path.basename(self.path))[0]
        self.sheet.save(f"./files/result/image/{filename}.jpg")

        os.makedirs(f"./files/result/csv", exist_ok= True)
        self.result.to_csv(f"./files/result/csv/{filename}.csv")

class Markgrid():
    """
    マークの１マスに関するクラス。
    
    img:対象マスの画像
    """

    def __init__(self, img, binary_threshold, mark_threshold, vratio = 0.8, hratio = 0.5, monitor = False):

        self.img = img
        self.binary_threshold = binary_threshold
        self.mark_threshold = mark_threshold
        self.vratio = vratio
        self.hratio = hratio
        self.monitor = monitor

    def make_mark(self):
        """
        マークの楕円形状を作成

        vratio:枠内でのマークの比率，実際より小さめに設定する
        """
        # 画像のサイズを設定
        height = int(self.img.shape[0])
        width = int(self.img.shape[1])

        # 楕円半径
        vradius = int(height * self.vratio / 2)
        hradius = int(width * self.hratio / 2)

        # 黒い画像を作成
        image = np.zeros((vradius*2, hradius*2), dtype=np.uint8)

        # 楕円の中心座標
        center_coordinates = (hradius, vradius)

        # 楕円のサイズ (長軸と短軸の半径)
        axes_length = (hradius, vradius)  # 半径なので20×30ピクセルの楕円

        # 楕円の色 (黒)
        color = (255, 255, 255)

        # 楕円を描画
        cv2.ellipse(image, center_coordinates, axes_length, 0, 0, 360, color, -1)
        
        return image

    def mark_check(self):
        """
        マークとの合致率が閾値以上かを判定する。
        """
        ret, bufimage = cv2.threshold(self.img[:,:,0], self.binary_threshold, 255, cv2.THRESH_BINARY)
        bufimage = 255 - bufimage
        reference = self.make_mark()

        result_img = cv2.matchTemplate(bufimage, reference, cv2.TM_CCOEFF_NORMED)
        __min_val, max_val, __min_loc, __max_loc = cv2.minMaxLoc(result_img)

        
        if max_val > self.mark_threshold:
            # 画像のサイズを設定
            height = int(self.img.shape[0])
            width = int(self.img.shape[1])

            # 円半径
            radius = int(height * 0.1)

            # 楕円の中心座標
            center_coordinates = (width//2, height//2)

            # 楕円のサイズ (長軸と短軸の半径)
            axes_length = (radius, radius)

            # 楕円の色 (黒)
            color = (0, 0, 255)

            # 楕円を描画
            cv2.ellipse(self.img, center_coordinates, axes_length, 0, 0, 360, color, -1)

            return True
        else:
            return False
