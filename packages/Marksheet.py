import cv2
import numpy as np

class Marksheet():
    def __init__(self, raw_img, x_start, x_end, y_start, y_end, row_num, column_num):
        """
        マークシート（１ブロック）のクラス，引数は元画像，ブロックのある範囲割合（0-1）行列数
        ブロックは，マーク領域を示す黒太線の枠，マーク部分（選択肢等は青で印刷）で構成される。
        """
        self.img = raw_img
        self.block = self.img[int(raw_img.shape[0] * y_start) : int(raw_img.shape[0] * y_end),
                              int(raw_img.shape[1] * x_start) : int(raw_img.shape[1] * x_end)]
        self.row = row_num
        self.col = column_num
        self.num_list = 0 #読み込み時に更新される。
    
    def read(self, binary_threshold, mark_threshold, check_flag):
        """
        マーク読み込み関数。引数は，2値化閾値，マーク検出の閾値，チェックの有無
        2023/2以降は閾値の自動調整機能を追加
        """

        # ブロック全体の表示
        cv2.imshow('trim', self.block); cv2.waitKey(1)
        # 青色のみ抽出（背景部分の削除）
        b_img, __g_img, __r_img = cv2.split(self.block)
        # 2値化
        __buf_img, binary_img = cv2.threshold(b_img, binary_threshold, 255, cv2.THRESH_BINARY)
        # エッジ抽出
        edge_img = cv2.Canny(binary_img, 50, 110)
        cv2.imshow('edge', edge_img); cv2.waitKey(1)
        # 回転角補正
        """
        rotation = self.get_rotate_angle(edge_img)
        center = (int(edge_img.shape[1]/2), int(edge_img.shape[0]/2))
        trans = cv2.getRotationMatrix2D(center, np.degrees(-rotation) , 1)
        edge_img  = cv2.warpAffine(edge_img , trans, (edge_img.shape[1],edge_img.shape[0]))
        self.block  = cv2.warpAffine(self.block , trans, (edge_img.shape[1],edge_img.shape[0]))
        """

        # 直線検出
        min_line_length = 5; max_line_gap = 10
        lines = cv2.HoughLinesP(edge_img, 1, np.pi/2, 10, min_line_length, max_line_gap)
        # 枠の確定
        max_x = 0; min_x = int(self.img.shape[1])
        max_y = 0; min_y = int(self.img.shape[0])
        for line in lines:
            x_1, y_1, x_2, y_2 = line[0]
            cv2.line(self.block, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)
            if x_1 == x_2:
                if x_1 > max_x:
                    max_x = x_1
                if x_1 < min_x:
                    min_x = x_1
            if y_1 == y_2:
                if y_1 > max_y:
                    max_y = y_1
                if y_1 < min_y:
                    min_y = y_1
        cv2.line(self.block, (max_x, max_y), (max_x, min_y), (255, 0, 255), 1)
        cv2.line(self.block, (min_x, max_y), (min_x, min_y), (255, 0, 255), 1)
        cv2.line(self.block, (max_x, max_y), (min_x, max_y), (255, 0, 255), 1)
        cv2.line(self.block, (max_x, min_y), (min_x, min_y), (255, 0, 255), 1)
        cv2.imshow('trim', self.block); cv2.waitKey(1)
        # 検出候補領域にトリミング
        table_img = binary_img[min_y : max_y, min_x : max_x]
        cv2.imshow('table1', table_img); cv2.waitKey(1)
        # 枠線の削除
        min_y = self.black_trim(table_img, 0, min_y)
        max_y = self.black_trim(table_img, 1, max_y)
        min_x = self.black_trim(table_img, 2, min_x)
        max_x = self.black_trim(table_img, 3, max_x)
        #print(f"{min_y}, {max_y}")
        table_img = binary_img[min_y : max_y, min_x : max_x]
        # 輪郭を可視化
        cv2.line(self.block, (max_x, max_y), (max_x, min_y), (0, 0, 255), 1)
        cv2.line(self.block, (min_x, max_y), (min_x, min_y), (0, 0, 255), 1)
        cv2.line(self.block, (max_x, max_y), (min_x, max_y), (0, 0, 255), 1)
        cv2.line(self.block, (max_x, min_y), (min_x, min_y), (0, 0, 255), 1)

        # マーク有無の判定，書き込み（見つからない場合は閾値を変更して再実行）
        self.num_list = []
        density_list = []
        detect_error = []
        for i in range(self.row):

            try_count = 0

            while True:
                detect_count = 0
                for j in range(self.col):
                    mark_img = table_img[int(table_img.shape[0] / self.row * i) :
                                         int(table_img.shape[0] / self.row * (i + 1)),
                                         int(table_img.shape[1] / self.col * j) :
                                         int(table_img.shape[1] / self.col * (j + 1))]

                    tf, density = self.mark_check(mark_img, mark_threshold)

                    if tf:
                        self.num_list.append(j)
                        density_list.append(density)                
                            
                        detect_count += 1

                        
                if detect_count == 1:
                    #検出マークが単一であれば修了
                    detect_error.append(0)
                    break

                elif detect_count > 1:
                    #多ければ，読み込んだマークを削除し，閾値を上げて再実行（１を上限）
                    mark_threshold*=1.1
                    print(f"up threshold to {mark_threshold}")

                    try_count += 1

                    act_mark = self.num_list[np.argmax((density_list[-detect_count:]))]
                    del self.num_list[-detect_count:]
                    
                    #閾値の上限に達したら，最も確率の高いマークを採用
                    if mark_threshold > 1 or try_count > 300:
                        detect_error.append(1)
                        self.num_list.append(act_mark)
                        break

                else:
                    #見つからない場合，閾値を下げて再実行（0.01を下限）
                    mark_threshold /= 1.1
                    print(f"down threshold to {mark_threshold}")

                    if mark_threshold < 0.01:
                        detect_error.append(1)
                        self.num_list.append(-1)
                        break

            #読み取り結果の可視化
            cv2.circle(self.block, (
                int(min_x + table_img.shape[1] / self.col * (self.num_list[-1] + 0.5)),
                int(min_y + table_img.shape[0] / self.row * (i + 0.5))),
                    4, (0, 0, 255), thickness=-1)
                    
        # 読み取りチェック
        cv2.imshow('table2', table_img); cv2.waitKey(1)
        cv2.imshow('trim', self.block); cv2.waitKey(1)
        
        # 読み取り結果の修正


        # 詳細修正（check_flagが1で実行）
        if check_flag == 1:

            # エラー箇所修正
            for i in range(self.row):
                if detect_error[i] == 1:
                    rev_row_num = i
                    while 1:
                        print(self.num_list)
                        rev_col_num = input(str(rev_row_num + 1) + '行目の正しい値 : ')
                        test_img = self.block.copy()
                        cv2.circle(test_img, (
                            int(min_x + table_img.shape[1] / self.col * (int(rev_col_num) + 0.6)),
                            int(min_y + table_img.shape[0] / self.row * (int(rev_row_num) + 0.6))),
                                6, (255, 0, 0), thickness=-1)
                        cv2.imshow('trim', test_img); cv2.waitKey(1)
                        check_answer = input('修正はこれでOKですか？　yes: 1, no: 0 : ')
                        if int(check_answer) == 1:
                            self.num_list[int(rev_row_num)] = int(rev_col_num)
                            cv2.circle(self.block, (
                                int(min_x + table_img.shape[1] / self.col * (int(rev_col_num) + 0.6)),
                                int(min_y + table_img.shape[0] / self.row    * (int(rev_row_num) + 0.6))),
                                    6, (255, 0, 0), thickness=-1)
                            break


            check_answer = input('読み込み結果は正しいですか？　yes: 1, no: 0 : ')
            if int(check_answer) == 0:
                while 1:
                    rev_row_num = input('修正行番号 : ')
                    rev_col_num = input('正しい値 : ')
                    test_img = self.block.copy()
                    cv2.circle(test_img, (
                        int(min_x + table_img.shape[1] / self.col * (int(rev_col_num) + 0.6)),
                        int(min_y + table_img.shape[0] / self.row * (int(rev_row_num) - 1 + 0.6))),
                            6, (255, 0, 0), thickness=-1)
                    cv2.imshow('trim', test_img); cv2.waitKey(1)
                    check_answer = input('修正はこれでOKですか？　yes: 1, no: 0 : ')
                    if int(check_answer) == 1:
                        self.num_list[int(rev_row_num) -1] = int(rev_col_num)
                        cv2.circle(self.block, (
                            int(min_x + table_img.shape[1] / self.col * (int(rev_col_num) + 0.6)),
                            int(min_y + table_img.shape[0] / self.row * (int(rev_row_num) - 1 + 0.6))),
                                6, (255, 0, 0), thickness=-1)
                        break

    def mark_check(self, mark_img, threshold):
        """
        mark_imgがマークされていればtrue，否ならばfalseを返す。
        """

        #マスクの読み込み
        mask = cv2.imread("./packages/images/mark.png")

        top_padding = int((mark_img.shape[0] - mask.shape[0])/2)

        bottom_padding = top_padding
        if int(mark_img.shape[0] - mask.shape[0]) % 2 != 0:
            bottom_padding += 1

        left_padding = int((mark_img.shape[1] - mask.shape[1])/2)

            
        right_padding = left_padding
        if int(mark_img.shape[1] - mask.shape[1]) % 2 != 0:
            right_padding += 1
        #print(f"v:{top_bottom_pading},h:{left_right_pading}")

        mask = cv2.copyMakeBorder(mask, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(255,255,255))
        mask, __g_img, __r_img = cv2.split(mask)
        __buf_img, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        buf = mark_img.copy(); buf = buf + 255
        buf[binary_mask==0] = mark_img[binary_mask==0]

        mark_density = (buf.size-cv2.countNonZero(buf)) / (binary_mask.size-cv2.countNonZero(binary_mask))
        #print(mark_density)
        #cv2.imshow('mark', buf); cv2.waitKey(1)
        #cv2.imshow('mask', binary_mask); cv2.waitKey(1)
        if mark_density > threshold:
            return True, mark_density
        else:
            return False, mark_density

    def get_rotate_angle(self, edge_img):
        """
        入力されたエッジ画像を元に，予想される回転角（ラジアン）を出力する。
        """
        #直線検出
        min_line_length = 5; max_line_gap = 10
        lines = cv2.HoughLines(edge_img, 1, np.pi/1800, 10, min_line_length, max_line_gap)
        #出力から角度だけ抽出
        thetas = lines.squeeze()[:,1]
        #print(thetas)
        rotations = np.mod(thetas - np.pi/4, np.pi/2) - np.pi/4
        #print(rotations)
        rotation = np.average(rotations)

        return rotation

    def black_trim(self, raw_img, xy_direction, outer_value):
        """
        入力画像の外側の黒線太さを吐き出す。引数は，元画像，方向(0:上，1:下，2:左，3:右), 元の枠位置
        2022/2/15 与えられた画像の際外側が黒線でない場合に対応
        """
        reference_white_density = 0.5
        if xy_direction <= 1:
            num = raw_img.shape[0] - 1
        else:
            num = raw_img.shape[1] - 1
        #探索範囲を設定
        num = int(num * 0.1)
        break_flag = 0

        black_checked = 0

        for i in range(num):
            # 指定方向のライン１ピクセル分を取り出す。
            if xy_direction == 0:
                check_img = raw_img[i : i + 1, 0 : raw_img.shape[1] - 1]
            elif  xy_direction == 1:
                check_img = raw_img[raw_img.shape[0] - 1 - (i + 1) : raw_img.shape[0] - 1 - i, 0 : raw_img.shape[1] - 1]
            elif  xy_direction == 2:
                check_img = raw_img[0 : raw_img.shape[0] - 1, i : i + 1]
            elif  xy_direction == 3:
                check_img = raw_img[0 : raw_img.shape[0] - 1, raw_img.shape[1] - 1 - (i + 1) : raw_img.shape[1] - 1 - i]
            # 白の密度を計算
            white_density = cv2.countNonZero(check_img) / check_img.size
            
            # 黒線を一度でも確認したかどうかのフラグ管理
            if white_density < reference_white_density:
                black_checked = 1

            # 黒線突入済み，かつ白の密度が閾値を2回超えれば現状の値を返して終了
            if white_density > reference_white_density and black_checked == 1:
                break_flag = break_flag + 1
                if break_flag > 2:
                    if xy_direction == 0 or xy_direction == 2:
                        return outer_value + (i - 2)
                    elif  xy_direction == 1 or xy_direction == 3:
                        return outer_value - (i - 1)
            else:
                break_flag = 0
        #条件を満たさなかった場合初期値で黒線除去完了
        return outer_value

    def readV1(self, binary_threshold, mark_threshold, check_flag):
        """
        マーク読み込み関数。引数は，2値化閾値，マーク検出の閾値，チェックの有無
        2023/2以降は角度補正機能を追加したメソッドへ置き換え
        """

        cv2.imshow('trim', self.block); cv2.waitKey(1)
        # 青色のみ抽出
        b_img, __g_img, __r_img = cv2.split(self.block)
        #cv2.imshow('blue', b_img)
        # 2値化
        __buf_img, binary_img = cv2.threshold(b_img, binary_threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow('binary', binary_img)
        # エッジ抽出
        edge_img = cv2.Canny(binary_img, 50, 110)
        cv2.imshow('edge', edge_img); cv2.waitKey(1)

        # 直線検出
        min_line_length = 5
        max_line_gap = 10
        lines = cv2.HoughLinesP(edge_img, 1, np.pi/2, 10, min_line_length, max_line_gap)
        #枠の確定
        max_x = 0
        min_x = int(self.img.shape[1])
        max_y = 0
        min_y = int(self.img.shape[0])
        for line in lines:
            x_1, y_1, x_2, y_2 = line[0]
            cv2.line(self.block, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)
            if x_1 == x_2:
                if x_1 > max_x:
                    max_x = x_1
                if x_1 < min_x:
                    min_x = x_1
            if y_1 == y_2:
                if y_1 > max_y:
                    max_y = y_1
                if y_1 < min_y:
                    min_y = y_1

        cv2.line(self.block, (max_x, max_y), (max_x, min_y), (255, 0, 255), 1)
        cv2.line(self.block, (min_x, max_y), (min_x, min_y), (255, 0, 255), 1)
        cv2.line(self.block, (max_x, max_y), (min_x, max_y), (255, 0, 255), 1)
        cv2.line(self.block, (max_x, min_y), (min_x, min_y), (255, 0, 255), 1)
        cv2.imshow('trim', self.block); cv2.waitKey(1)
        # 検出候補領域にトリミング
        table_img = binary_img[min_y : max_y, min_x : max_x]
        cv2.imshow('table', table_img); cv2.waitKey(1)
        # 枠線の削除
        min_y = self.black_trim(table_img, 0, min_y)
        max_y = self.black_trim(table_img, 1, max_y)
        min_x = self.black_trim(table_img, 2, min_x)
        max_x = self.black_trim(table_img, 3, max_x)
        #print(f"{min_y}, {max_y}")
        table_img = binary_img[min_y : max_y, min_x : max_x]
        # 輪郭を可視化
        cv2.line(self.block, (max_x, max_y), (max_x, min_y), (0, 0, 255), 1)
        cv2.line(self.block, (min_x, max_y), (min_x, min_y), (0, 0, 255), 1)
        cv2.line(self.block, (max_x, max_y), (min_x, max_y), (0, 0, 255), 1)
        cv2.line(self.block, (max_x, min_y), (min_x, min_y), (0, 0, 255), 1)
        # マーク有無の判定，書き込み
        self.num_list = []
        detect_error = []
        for i in range(self.row):
            detect_count = 0
            for j in range(self.col):
                mark_img = table_img[int(table_img.shape[0] / self.row * i) :
                                    int(table_img.shape[0] / self.row * (i + 1)),
                                    int(table_img.shape[1] / self.col * j) :
                                    int(table_img.shape[1] / self.col * (j + 1))]
                black_density = 1 - cv2.countNonZero(mark_img) / mark_img.size
                if black_density > mark_threshold:
                    if detect_count == 0:
                        self.num_list.append(j)
                        #print(j)
                    detect_count += 1
                    cv2.circle(self.block, (
                        int(min_x + table_img.shape[1] / self.col * (j + 0.5)),
                        int(min_y + table_img.shape[0] / self.row * (i + 0.5))),
                            4, (0, 0, 255), thickness=-1)
            if detect_count == 1:
                detect_error.append(0)
            elif detect_count > 1:
                detect_error.append(1)
            else:
                detect_error.append(1)
                self.num_list.append(0)
        # 読み取りチェック
        cv2.imshow('table', table_img); cv2.waitKey(1)
        cv2.imshow('trim', self.block); cv2.waitKey(1)
        # 読み取り結果の修正
        # エラー箇所修正
        for i in range(self.row):
            if detect_error[i] == 1:
                rev_row_num = i
                while 1:
                    print(self.num_list)
                    rev_col_num = input(str(rev_row_num + 1) + '行目の正しい値 : ')
                    test_img = self.block.copy()
                    cv2.circle(test_img, (
                        int(min_x + table_img.shape[1] / self.col * (int(rev_col_num) + 0.6)),
                        int(min_y + table_img.shape[0] / self.row * (int(rev_row_num) + 0.6))),
                            6, (255, 0, 0), thickness=-1)
                    cv2.imshow('trim', test_img); cv2.waitKey(1)
                    check_answer = input('修正はこれでOKですか？　yes: 1, no: 0 : ')
                    if int(check_answer) == 1:
                        self.num_list[int(rev_row_num)] = int(rev_col_num)
                        cv2.circle(self.block, (
                            int(min_x + table_img.shape[1] / self.col * (int(rev_col_num) + 0.6)),
                            int(min_y + table_img.shape[0] / self.row    * (int(rev_row_num) + 0.6))),
                                6, (255, 0, 0), thickness=-1)
                        break
        # 詳細修正（check_flagが1で実行）
        if check_flag == 1:
            check_answer = input('読み込み結果は正しいですか？　yes: 1, no: 0 : ')
            if int(check_answer) == 0:
                while 1:
                    rev_row_num = input('修正行番号 : ')
                    rev_col_num = input('正しい値 : ')
                    test_img = self.block.copy()
                    cv2.circle(test_img, (
                        int(min_x + table_img.shape[1] / self.col * (int(rev_col_num) + 0.6)),
                        int(min_y + table_img.shape[0] / self.row * (int(rev_row_num) - 1 + 0.6))),
                            6, (255, 0, 0), thickness=-1)
                    cv2.imshow('trim', test_img); cv2.waitKey(1)
                    check_answer = input('修正はこれでOKですか？　yes: 1, no: 0 : ')
                    if int(check_answer) == 1:
                        self.num_list[int(rev_row_num) -1] = int(rev_col_num)
                        cv2.circle(self.block, (
                            int(min_x + table_img.shape[1] / self.col * (int(rev_col_num) + 0.6)),
                            int(min_y + table_img.shape[0] / self.row * (int(rev_row_num) - 1 + 0.6))),
                                6, (255, 0, 0), thickness=-1)
                        break
                    
    def black_trimV1(self, raw_img, xy_direction, outer_value):
        """
        入力画像の外側の黒線太さを吐き出す。引数は，元画像，方向(0:上，1:下，2:左，3:右), 元の枠位置
        2022/2/15 与えられた画像の際外側が黒線でない場合に対応
        """
        reference_white_density = 0.2
        if xy_direction <= 1:
            num = raw_img.shape[0] - 1
        else:
            num = raw_img.shape[1] - 1
        break_flag = 0

        for i in range(num):
            # 指定方向のライン１ピクセル分を取り出す。
            if xy_direction == 0:
                check_img = raw_img[i : i + 1, 0 : raw_img.shape[1] - 1]
            elif  xy_direction == 1:
                check_img = raw_img[raw_img.shape[0] - 1 - (i + 1) : raw_img.shape[0] - 1 - i,
                                    0 : raw_img.shape[1] - 1]
            elif  xy_direction == 2:
                check_img = raw_img[0 : raw_img.shape[0] - 1, i : i + 1]
            elif  xy_direction == 3:
                check_img = raw_img[0 : raw_img.shape[0] - 1,
                                    raw_img.shape[1] - 1 - (i + 1) : raw_img.shape[1] - 1 - i]
            #白の密度を計算
            white_density = cv2.countNonZero(check_img) / check_img.size
            
            #白の密度が閾値を超えていれば現状の値を返して終了
            if white_density > reference_white_density:
                break_flag = break_flag + 1
                if break_flag > 2:
                    if xy_direction == 0 or xy_direction == 2:
                        return outer_value + (i - 2)
                    elif  xy_direction == 1 or xy_direction == 3:
                        return outer_value - (i - 1)
            else:
                break_flag = 0

class Marksheet2():
    def __init__(self):
        pass

    def read(self):
        pass

    def mark_check(self):
        pass