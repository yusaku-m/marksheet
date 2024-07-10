import os
import cv2
import math
import pandas as pd
import numpy as np

from .Image import Image
from .Marksheet import Marksheet

class Paper(Image):
    def __init__(self, basepath):
        self.basepath = basepath
        self.img            = 0         #用紙
        self.student_num    = 0         #出席番号

    def makesheet(self, path1, path2=0, icon_path = ["./packages/images/icon_kosen.png", "./packages/images/icon_kagawa.png"]):
        """
        答案のスキャンファイルを画像化。path2を指定した場合は，裏表の答案として，一枚の結合画像を得る。
        icon_pathは表裏・向き判別に使用する。未指定の場合学校のアイコンが使用される。自作する場合は1mm4pixelに対応するように作成する。
        """

        buf = []
        buf.append(Image(path1).img)
        if path2 != 0:
            buf.append(Image(path2).img)

        icons = []
        for path in icon_path:
            icons.append(Image(path).img)

        #縦画像は横にする。
        for image in buf:

            if  image.shape[1] < image.shape[0]:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
            #サイズ補正
            image = cv2.resize(image, dsize=(297*4, 210*4))
            
            # 方向判定（裏表，回転）
            best_match = 0
            for i, checker in enumerate(icons):
                for j in range(2):
                    result_img = cv2.matchTemplate(image, checker, cv2.TM_CCOEFF_NORMED)
                    __min_val, max_val, __min_loc, __max_loc = cv2.minMaxLoc(result_img)
                    #print(i, max_val)
                    if max_val > best_match:
                            best_match = max_val
                            best_rotation = j
                            best_icon = i
                    #回転してもう一度確認
                    checker = cv2.rotate(checker, cv2.ROTATE_180)
                    print(f"icon: {i}, rotation: {j}, max_val: {max_val}")

            # 結果をもとに回転
            print(f"best_rotation: {best_rotation}, best_icon: {best_icon}")
            if best_rotation == 1:
                image = cv2.rotate(image, cv2.ROTATE_180)
            if best_icon == 0:
                front = image
            else:
                back = image          
        # 結合
        if path2 == 0:
            self.img = front                
        else:
            self.img = cv2.vconcat([front, back])  

    def readsheet(self, tergetfolder, student_num):
        path = f"{self.basepath}/{tergetfolder}/{str(student_num).zfill(2)}.jpg"
        self.img = Image(path).img
        self.student_num = student_num

    def get_student(self, startpoints = (0.1, 0.04), endpoints = (0.3, 0.095), binary_threshold = 180, ratio_threshold = 0.2):
        """
        binary_threshold: マーク検出における２値化閾値
        ratio_threshold: マーク検出における黒面積比率の閾値
        """
        mark_block = Marksheet(self.img, startpoints[0], endpoints[0], startpoints[1], endpoints[1], 2, 10)
        mark_block.read(binary_threshold, ratio_threshold, 0)
        print(mark_block.num_list)
        self.student_num = mark_block.num_list[0]*10 + mark_block.num_list[1]*1

    def update_score(self, score, note ='', secret_regions = [[(0,0), (0.5, 0.095)], [(0,0.5), (0.5,0.55)]]):
        path = f"{self.basepath}/result/{str(self.student_num).zfill(2)}.jpg"
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            print('Folder is not exist...')
            os.makedirs(folder)
        if not os.path.exists(path):
            path = f"{self.basepath}/concat/{str(self.student_num).zfill(2)}.jpg"

        buf = Image(path).img
        y = buf.shape[0]; x = buf.shape[1]
        #名前を隠しつつ採点結果を記入
        raw = self.img.copy()
        raw_crop = raw[int(y*secret_regions[0][0][1]):int(y*secret_regions[0][1][1]), int(x*secret_regions[0][0][0]):int(x*secret_regions[0][1][0])]
        cv2.rectangle(raw_crop, (0,0), (int(x*secret_regions[0][1][0])-int(x*secret_regions[0][0][0]),int(y*secret_regions[0][1][1])-int(y*secret_regions[0][0][1])), (255, 255, 255), thickness=-1)
        cv2.putText(raw_crop, str(math.ceil(score)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 8)
        cv2.putText(raw_crop, note, (40, int(y*0.080)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        buf[int(y*secret_regions[0][0][1]):int(y*secret_regions[0][1][1]), int(x*secret_regions[0][0][0]):int(x*secret_regions[0][1][0])] = raw_crop

        raw = self.img.copy()
        raw_crop = raw[int(y*secret_regions[1][0][1]):int(y*secret_regions[1][1][1]), int(x*secret_regions[1][0][0]):int(x*secret_regions[1][1][0])]
        cv2.rectangle(raw_crop, (0,0), (int(x*secret_regions[1][1][0])-int(x*secret_regions[1][0][0]),int(y*secret_regions[1][1][1])-int(y*secret_regions[1][0][1])), (255, 255, 255), thickness=-1)
        buf[int(y*secret_regions[1][0][1]):int(y*secret_regions[1][1][1]), int(x*secret_regions[1][0][0]):int(x*secret_regions[1][1][0])] = raw_crop
        Image(img = raw).show("raw")
        Image(img = buf).save(f"{self.basepath}/result/{str(self.student_num).zfill(2)}",True)

    def update_score_MD(self, score, note =''):
        path = f"{self.basepath}/result/{str(self.student_num).zfill(2)}.jpg"
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            print('Folder is not exist...')
            os.makedirs(folder)
        if not os.path.exists(path):
            path = f"{self.basepath}/concat/{str(self.student_num).zfill(2)}.jpg"
        buf = Image(path).img
        y = buf.shape[0]; x = buf.shape[1]
        #名前を隠しつつ採点結果を記入
        raw = self.img.copy()
        raw = raw[0:int(y*0.17), 0:int(x*0.5)]
        cv2.rectangle(raw, (0,0), (int(x*0.5),int(y*0.17)), (255, 255, 255), thickness=-1)
        cv2.putText(raw, str(math.ceil(score)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 8)
        cv2.putText(raw, note, (40, int(y*0.165)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        buf[0:int(y*0.17), 0:int(x*0.5)]= raw

        Image(img = raw).show("raw")
        Image(img = buf).save(f"{self.basepath}/result/{str(self.student_num).zfill(2)}",True)

    def save(self, tergetfolder):
        path = f"{self.basepath}/{tergetfolder}/{str(self.student_num).zfill(2)}"
        #print(self.student_num)
        super().save(path)
        
class Question():
    """問題の共通メソッドはここに"""
    def __init__(self, sheets, name, regions, score):
        """
        問題の場所と配点を定義
        sheets:回答用紙全体。paperクラスを渡す。
        neme:問題の名前，("Q1"等)
        region: 開始点，終了点(x,y)のタプルを１要素に持つリスト（例[[(x1,y1), ((x2,x2))], [(x3,y3), ((x4,x4))]]）値は高さ，幅を1とした比率で渡す。
        score:各問の得点をリストで指定

        メンバ変数
        self.student_num = 学生の出席番号リスト
        """
        self.name = name
        self.regions = regions
        self.score = score
        self.answers = []
        self.correct = []
        self.grade = 0
        self.basepath = 0
        self.sumscore = 0
        self.basepath =sheets[0].basepath
        self.sheets = sheets
        self.student_num =[]       
        for sheet in self.sheets:
            self.student_num.append(sheet.student_num)

    def read(self):
        """全学生の回答を読み込み"""
        #読み込み
        print(f"{self.name} reading...")
        self.answers = []     
        for i, sheet in enumerate(self.sheets):
            #画像の読み込み
            raw_img = sheet.img
            y = raw_img.shape[0]; x = raw_img.shape[1]
            for j, region in enumerate(self.regions):
                part = raw_img[int(y * region[0][1]) : int(y * region[1][1]), int(x * region[0][0]) : int(x * region[1][0])].copy()
                if j==0:
                    answer = part
                else:
                    part = cv2.resize(part, (answer.shape[1], part.shape[0]))
                    answer = cv2.vconcat([answer.copy(), part])  
            self.answers.append(answer)
            
            Image(img = raw_img).show("raw_img")
            Image(img = self.answers[i]).show("self.answers")

        cv2.destroyAllWindows()

    def grading(self, index=0):
        """単一学生の採点"""
        #採点して結果を配列へ格納
        path = f"{self.basepath}/{self.name}.csv"
        if os.path.isfile(path):
            self.grade = pd.read_csv(path, index_col=0)
        else:
            data = np.zeros((len(self.score), len(self.student_num)))
            columns = map(str, self.student_num)
            rows = range(1, len(self.score) + 1)
            print(data)
            print(columns)
            print(rows)
            self.grade = pd.DataFrame(data, rows, columns)

    def scoring(self, basepath=0):
        """採点結果から問題の得点を集計"""
        if self.basepath == 0:
            self.basepath = basepath
        path = f"{self.basepath}/{self.name}.csv" 
        if os.path.isfile(path) and type(self.grade) is int:
            self.grade = pd.read_csv(path, index_col=0)
        self.sumscore = np.dot(np.array(self.score), self.grade.values)

    def save(self):
        pass

    def load(self):
        pass

    def get_correct_answer(self, index=99):
        pass

    def show(self, index):
        """単一学生の答案を表示"""
        Image(img = self.answers[index-1]).show(str(index).zfill(2))

    def update(self):
        pass

class SingleQuestion(Question):
    """一問一答形式"""
    def __init__(self, sheets, name, regions, column, score):
        super().__init__(sheets, name, regions, score)
        self.col = column
        self.row = int(len(self.score) / self.col)

    def read(self):
        super().read()
        #小問毎に結合
        buf = []
        raw = []
        row = self.row
        for i in range(len(self.score)):
            for j, sheet in enumerate(self.answers):
                x = sheet.shape[1]; y = sheet.shape[0] 
                answer = sheet[int(y * (i%row)/row) : int(y * ((i%row)/row + 1/row)), int(x * int(i/row)/self.col) : int(x * (int(i/row)+1)/self.col)]
                Image(img = answer).show("answer")
                if j == 0:
                    concat = answer
                else:
                    concat = cv2.vconcat([concat, answer])
            #3列に並べる
            x = answer.shape[1]; y = answer.shape[0]
            answer = answer.copy()
            cv2.rectangle(answer, (0, 0), (x, y), (0, 0, 0), thickness=-1)
            for k in range((3-len(self.answers)%3) % 3):
                concat = cv2.vconcat([concat, answer])
            x = concat.shape[1]; y = concat.shape[0]
            y -= (y % 3)
            a = concat[0:int(1/3*y), 0:x]
            b = concat[int(1/3*y):int(2/3*y), 0:x]
            c = concat[int(2/3*y):y, 0:x]
            concat = cv2.hconcat([a, b, c])           
            buf.append(concat)
            copy = concat.copy()
            raw.append(copy)
        self.answerslist = buf
        self.rawanswerlist = raw

    def grading(self, index):
        super().grading()
        #採点
        path = f"{self.basepath}/{self.name}/{str(index).zfill(2)}.jpg"
        if os.path.isfile(path):
            current = Image(path).img
        else:
            current = self.answerslist[index-1]
        raw = self.rawanswerlist[index-1].copy()
        
        i = 0

        while i < len(self.student_num):

            buf = current.copy()
            x = self.answerslist[index-1].shape[1]; y = self.answerslist[index-1].shape[0] 
            row = math.ceil(len(self.student_num) / 3)
            s = (int(x * int(i/row)/3), int(y*(i%row)/row))
            e = (int(x * (int(i/row)+1)/3), int(y * ((i%row)/row + 1/row)))
            g = (s[0] + int((e[0]-s[0])*0.9), s[1]+ int((e[1]-s[1])*0.5))
            r = int((e[1]-s[1])*0.1)
            cv2.rectangle(buf, s, e, (0, 0, 255), thickness=3)

            Image(img = self.rawanswerlist[index-1]).show("raw")

            amp = 1300 / max(buf.shape[0],buf.shape[1])

            dst = cv2.resize(buf, dsize=None, fx=amp, fy=amp)

            Image(img = dst).show("grading",enlargement=True)
            Image(img = self.answers[i]).show("answer",enlargement=True)

            grade = input("Right: 1, half: 2, Wrong: 0, Back: 3, Skip: Escape")

            try:
                grade = int(grade)
                if grade == 3:
                    i -= 2
                    
                else:
                    #採点箇所を復元
                    current[s[1]:e[1],s[0]:e[0]] = raw[s[1]:e[1],s[0]:e[0]]
                    if grade == 1:
                        cv2.circle(current, g, r, (0, 0, 255), thickness=2)
                    elif grade == 0:
                        cv2.line(current, (g[0]-r, g[1]-r), (g[0]+r, g[1]+r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(current, (g[0]-r, g[1]+r), (g[0]+r, g[1]-r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    elif grade == 2:
                        points = np.array([(g[0], g[1]- r), (g[0] - r, g[1] + r), (g[0] + r, g[1] + r)])
                        cv2.polylines(current, [points], True, (0, 0, 255), 2)
                        grade = 0.5
                    else:
                        grade=0
                        i -= 2

                    
                    
                    self.grade.at[index, str(self.student_num[i])] = grade
                    #保存
                    self.result = current
                    #元画像へ反映
                    row = self.row
                    x = self.answers[i].shape[1]; y = self.answers[i].shape[0] 
                    ys = int(y * ((index-1)%row)/row)
                    ye = int(y * (((index-1)%row)/row + 1/row))
                    xs = int(x * int((index-1)/row)/self.col)
                    xe = int(x * (int((index-1)/row)+1)/self.col)
                    #print(f"[{ys} : {ye}, {xs} : {xe}], [{s[1]}:{s[1]+ye-ys},{s[0]}:{s[0]+xe-xs}]")
                    self.answers[i][ys : ye, xs : xe] = current[s[1]:s[1]+ye-ys,s[0]:s[0]+xe-xs]
                    Image(img = self.answers[i]).show("answer")
                    self.save(index)

            except:
                pass

            i += 1



    def save(self, index):
        #問題毎の採点画像を保存
        path = f"{self.basepath}/{self.name}/{str(index).zfill(2)}"
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            print('Folder is not exist...')
            os.makedirs(folder)

        Image(img = self.result).save(path, overwrite = True)

        #問題毎の採点集計を保存
        path = f"{self.basepath}/{self.name}"
        self.grade.to_csv(f"{path}.csv")

    def update(self):
        print(f"{self.name} updating...")
        self.scoring()
        #学生ループ
        for i in range(len(self.sheets)):
            #採点結果を原紙に反映
            path = f"{self.basepath}/result/{str(self.student_num[i]).zfill(2)}.jpg"
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                print('Folder is not exist...')
                os.makedirs(folder)
            if not os.path.exists(path):
                path = f"{self.basepath}/concat/{str(self.student_num[i]).zfill(2)}.jpg"
            buf = Image(path).img

            #領域毎の高さを取得
            region_heights = []; total_height = 0
            for region in self.regions:
                region_heights.append(region[1][1] - region[0][1])
                total_height += region[1][1] - region[0][1]

            #領域毎の小問数を取得
            region_columns = []
            for region_height in region_heights:
                region_columns.append(round(len(self.score) * region_height / total_height / self.col))

            #各小問ループ
            for j in range(len(self.score)):
                # 対象学生の更新前領域を定義
                path = f"{self.basepath}/{self.name}/{str(j+1).zfill(2)}.jpg"
                sheet = Image(path).img
                y = buf.shape[0]; x = buf.shape[1]
                b = self.col; h = math.ceil(len(self.score)/self.col)

                # 描画領域の検索
                sumrow = 0; offset_row = 0
                for k, region_column in enumerate(region_columns):
                    sumrow += region_column
                    if j < sumrow * self.col:
                        region = self.regions[k]; break
                    offset_row += region_column
                
                ys = int(y * (region[0][1] + (region[1][1] - region[0][1]) * ((j-offset_row)%region_columns[k]/region_columns[k]))) 
                xs = int(x * (region[0][0] + (region[1][0] - region[0][0]) * (math.floor((j-offset_row)/region_columns[k])/b)))
                
                #対象学生の採点済み画像を取得
                b = 3; h = math.ceil(len(self.sheets) / b)
                startpoints = (int(sheet.shape[1] * (math.floor(i/h))/b), int(sheet.shape[0] * (i % h)/h)) 
                endpoints   = (int(sheet.shape[1] * (math.floor(i/h)+1)/b), int(sheet.shape[0] * (i % h + 1)/h))

                # 更新
                buf[ys : ys + endpoints[1] - startpoints[1], xs : xs + endpoints[0] - startpoints[0]] = sheet[startpoints[1]:endpoints[1], startpoints[0]:endpoints[0]]
                if j == 0:
                    cv2.rectangle(buf, (xs, ys), (xs+50, ys+30), (255, 255, 255), thickness=-1)
                    cv2.putText(buf, str(self.sumscore[i]), (xs+10, ys+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    #print(self.student_num[i])
            Image(img=buf).save(f"{self.basepath}/result/{str(self.student_num[i]).zfill(2)}",True)

class PartialQuestion(SingleQuestion):
    """小問毎に採点，くわえて，各小問で複数の採点項目がある形式"""
    def __init__(self, sheets, name, regions, partial, score):
        super().__init__(sheets, name, regions, 1, score)
        self.partial = partial

    def grading(self, index):
        #採点して結果を配列へ格納
        path = f"{self.basepath}/{self.name}.csv"
        if os.path.isfile(path):
            self.grade = pd.read_csv(path, index_col=0)
        else:
            data = np.zeros((len(self.score), len(self.student_num)))
            columns = map(str, self.student_num)
            rows = range(1, len(self.score) + 1)
            self.grade = pd.DataFrame(data, rows, columns)
        #print(self.grade)

        #採点
        path = f"{self.basepath}/{self.name}/{str(index).zfill(2)}.jpg"
        if os.path.isfile(path):
            current = Image(path).img
        else:
            current = self.answerslist[index-1].copy()
        raw = self.rawanswerlist[index-1].copy()

        i = 0
        while i < len(self.student_num):

            #学生単位ループ
            h = self.partial[index-1][0]; b = self.partial[index-1][1]
            totalgrade = 0
            buf = current.copy()
            x = self.answerslist[index-1].shape[1]; y = self.answerslist[index-1].shape[0] 
            row = math.ceil(len(self.student_num) / 3)
            s = (int(x * int(i/row)/3), int(y*(i%row)/row))
            e = (int(x * (int(i/row)+1)/3), int(y * ((i%row)/row + 1/row)))
            #採点箇所の強調
            cv2.rectangle(buf, s, e, (0, 0, 255), thickness=3)
            Image(img = self.rawanswerlist[index-1]).show("raw")
            dst = cv2.resize(buf, dsize=None, fx= 1300/max(buf.shape[0], buf.shape[1]), fy=1300/max(buf.shape[0], buf.shape[1]))

            Image(img = dst).show("grading",enlargement=True)
            Image(img = self.answers[i]).show("answer",enlargement=True)

            for j in range(h*b):
                #小問内枠（数式，数値など）ループ
                grade = input("Right: 1, half: 2, Wrong: 0, Back: 3, Skip: Escape")

                try:
                    grade = int(grade)

                    if grade == 3:
                        i -= 2
                        break

                    else:   
                        ps = (s[0] + int((e[0]-s[0]) * ((j%b)/b)), s[1]+ int((e[1]-s[1]) * (int(j/b)/h)))
                        pe = (s[0] + int((e[0]-s[0]) * ((j%b+1)/b)), s[1]+ int((e[1]-s[1]) * (int(j/b+1)/h)))
                        r = int((pe[1]-ps[1])*0.1)
                        g = (pe[0] - int(r*1.3), ps[1] + int(r*1.3))
                        #採点箇所を復元
                        Image(img = raw[ps[1] : pe[1], ps[0] : pe[0]]).show("grading",enlargement=True)
                        cv2.imshow("rawextract", raw[ps[1] : pe[1], ps[0] : pe[0]]); cv2.waitKey(1)

                        current[ps[1] : pe[1], ps[0] : pe[0]] = raw[ps[1] : pe[1], ps[0] : pe[0]].copy()
                        if grade == 1:
                            cv2.circle(current, g, r, (0, 0, 255), thickness=2)
                        if grade == 0:
                            cv2.line(current, (g[0]-r, g[1]-r), (g[0]+r, g[1]+r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                            cv2.line(current, (g[0]-r, g[1]+r), (g[0]+r, g[1]-r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                        if grade == 2:
                            points = np.array([(g[0], g[1]- r), (g[0] - r, g[1] + r), (g[0] + r, g[1] + r)])
                            cv2.polylines(current, [points], True, (0, 0, 255), 2)
                            grade = 0.5
                        totalgrade += grade /(b*h)
                        amp = 1300 / max(buf.shape[0], buf.shape[1])
                        dst = cv2.resize(buf, dsize=None, fx=amp , fy=amp)
                        Image(img = dst).show("grading",enlargement=True)

                except:
                    totalgrade = self.grade.at[index, str(self.student_num[i])]

            self.grade.at[index, str(self.student_num[i])] = totalgrade
            #保存
            self.result = current
            #元画像へ反映
            row = self.row
            self.answers[i] = current
            Image(img = self.answers[i]).show("answer")
            self.save(index)

            i += 1

class BigQuestion(SingleQuestion):
    """前後の小問の流れを含め採点する問題"""
    def __init__(self, sheets, name, regions, heights, partial, score):
        super().__init__(sheets, name, regions, 1, score)
        self.partial = partial
        self.heights = heights

    def read(self):
        Question.read(self)

    def grading(self, index = 1):
        """学生ごとに採点，indexは開始学生の出席番号-1"""
        #採点結果配列の読み込み
        path = f"{self.basepath}/{self.name}.csv"
        if os.path.isfile(path):
            self.grade = pd.read_csv(path, index_col=0)
        else:
            #無ければ作成
            data = np.zeros((len(self.score), len(self.student_num)))
            columns = map(str, self.student_num)
            rows = range(1, len(self.score) + 1)
            self.grade = pd.DataFrame(data, rows, columns)

        #indexで指定した出席番号以降の学生インデックスを作成
        start_index = self.student_num.index(index)

        for i in range(start_index, len(self.student_num)):
            #学生単位ループ
            print(self.student_num[i])
            x = self.answers[i].shape[1]; y = self.answers[i].shape[0]
            
            path = f"{self.basepath}/{self.name}/{str(self.student_num[i]).zfill(2)}.jpg"
            if os.path.isfile(path):
                current = Image(path).img
            else:
                current = self.answers[i].copy()
            raw = self.answers[i].copy()
            row = len(self.heights) - 1
            
            j = 0
            while j < row:
                #小問単位ループ
                h = self.partial[j][0]; b = self.partial[j][1]
                totalgrade = 0
                buf = current.copy()

                s = (0, int(y * self.heights[j]))
                e = (x, int(y * self.heights[j+1]))

                #採点箇所の強調
                cv2.rectangle(buf, s, e, (0, 0, 255), thickness=3)
                dst = cv2.resize(buf, dsize=None, fx= 1300/max(buf.shape[0], buf.shape[1]), fy=1300/max(buf.shape[0], buf.shape[1]))

                Image(img=dst).show("grading",enlargement=True)
                Image(img=raw).show("raw",enlargement=True)
                Image(img=self.answers[i]).show("answer",enlargement=True)               

                for k in range(h*b):
                    #小問内枠（数式，数値など）ループ
                    grade = input("Correct: 1, half: 2, Wrong: 0, Back: 3")

                    try:
                        grade = int(grade)

                        if grade == 3:
                            j -= 1
                            break

                        ps = (s[0] + int((e[0]-s[0]) * ((k%b)/b)), s[1]+ int((e[1]-s[1]) * (int(k/b)/h)))
                        pe = (s[0] + int((e[0]-s[0]) * ((k%b+1)/b)), s[1]+ int((e[1]-s[1]) * (int(k/b+1)/h)))
                        r = int((pe[1]-ps[1])*0.15)
                        g = (pe[0] - int(r*1.5), ps[1] + int(r*1.5))
                        #採点箇所を復元
                        current[ps[1] : pe[1], ps[0] : pe[0]] = raw[ps[1] : pe[1], ps[0] : pe[0]].copy()
                        if grade == 1:
                            cv2.circle(current, g, r, (0, 0, 255), thickness=2)
                        if grade == 0:
                            cv2.line(current, (g[0]-r, g[1]-r), (g[0]+r, g[1]+r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                            cv2.line(current, (g[0]-r, g[1]+r), (g[0]+r, g[1]-r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                        if grade == 2:
                            points = np.array([(g[0], g[1]- r), (g[0] - r, g[1] + r), (g[0] + r, g[1] + r)])
                            cv2.polylines(current, [points], True, (0, 0, 255), 2)
                            grade = 0.5
                        totalgrade += grade /(b*h)
                        dst = cv2.resize(buf, dsize=None, fx= 1300/buf.shape[0], fy=1300/buf.shape[0])

                        Image(img=dst).show("grading",enlargement=True)
                        Image(img=raw[ps[1] : pe[1], ps[0] : pe[0]]).show("rawextract")
                        Image(img=raw).show("raw",enlargement=True)
                        Image(img=current).show("current",enlargement=True)
                        
                    except:
                        totalgrade = self.grade.at[j+1, str(self.student_num[i])]

                if grade == 3:
                    continue

                self.grade.at[j+1, str(self.student_num[i])] = totalgrade
                #保存
                self.result = current
                #元画像へ反映
                row = self.row
                self.answers[i] = current

                Image(img=self.answers[i]).show("answer")     

                self.save(self.student_num[i])
                self.answers[i] = raw.copy()

                j += 1

        cv2.destroyAllWindows()

    def update(self):
        print(f"{self.name} updating...")
        self.scoring()
        for i in range(len(self.sheets)):
            #結果集約用ファイルの読み込み
            path = f"{self.basepath}/result/{str(self.student_num[i]).zfill(2)}.jpg"
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                print('Folder is not exist...')
                os.makedirs(folder)
            if not os.path.exists(path):
                path = f"{self.basepath}/concat/{str(self.student_num[i]).zfill(2)}.jpg"
            buf = Image(path).img
            #採点結果ファイルの読み込み
            path = f"{self.basepath}/{self.name}/{str(self.student_num[i]).zfill(2)}.jpg"
            sheet = Image(path).img

            y = buf.shape[0]; x = buf.shape[1]
            cv2.rectangle(sheet, (0, 0), (50, 30), (255, 255, 255), thickness=-1)
            cv2.putText(sheet, str(self.sumscore[i]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
            sx0 = 0; sy0 = 0
            for region in self.regions:
                ys = int(y * region[0][1]); ye = int(y * region[1][1])
                xs = int(x * region[0][0]); xe = xs + sheet.shape[1]

                #print (f"ys{ys},ye{ye},xs{xs},xe{xe}")
                
                buf[ys : ye, xs : xe] = sheet[sy0 : sy0 + ye-ys, sx0 : xe-xs]
                sy0 = ye-ys

            Image(img=buf).save(f"{self.basepath}/result/{str(self.student_num[i]).zfill(2)}",True)

class FreeQuestion(BigQuestion):
    """前後の小問の流れを含め採点する問題，小問配置も自由"""
    def __init__(self, sheets, name, regions, coordinates, partial, score):
        """
        coordinates:各小問の支点(x,y)，終点（x,y)の２次元リスト。各座標はタプル指定，高さ，幅を１とした比率
        partial:各小門が何行何列で分割されているか。(行，列)のタプルで構成されるリスト。
        """
        super().__init__(sheets, name, regions, 1, partial, score)
        self.coordinates = coordinates

    def grading(self, index = 0):
        #採点結果配列の読み込み
        path = f"{self.basepath}/{self.name}.csv"
        if os.path.isfile(path):
            self.grade = pd.read_csv(path, index_col=0)
        else:
            #無ければ作成
            data = np.zeros((len(self.score), len(self.student_num)))
            columns = map(str, self.student_num)
            rows = range(1, len(self.score) + 1)
            self.grade = pd.DataFrame(data, rows, columns)

        #indexで指定した出席番号以降の学生インデックスを作成
        start_index = self.student_num.index(index)
        #print(start_index)

        for i in range(start_index, len(self.student_num)):
            #学生単位ループ
            print(self.student_num[i])
            x = self.answers[i].shape[1]; y = self.answers[i].shape[0]
            path = f"{self.basepath}/{self.name}/{str(self.student_num[i]).zfill(2)}.jpg"
            if os.path.isfile(path):
                current = Image(path).img
            else:
                current = self.answers[i].copy()
            raw = self.answers[i].copy()

            for j in range(len(self.coordinates)):
                #小問単位ループ
                h = self.partial[j][0]; b = self.partial[j][1]
                totalgrade = 0
                buf = current.copy()
                s = (int(x * self.coordinates[j][0][0]), int(y * self.coordinates[j][0][1]))
                e = (int(x * self.coordinates[j][1][0]), int(y * self.coordinates[j][1][1]))

                #採点箇所の強調
                cv2.rectangle(buf, s, e, (0, 0, 255), thickness=3)
                dst = cv2.resize(buf, dsize=None, fx= 1300/max(buf.shape[0], buf.shape[1]), fy=1300/max(buf.shape[0], buf.shape[1]))

                Image(img=dst).show("grading",enlargement=True)
                Image(img=raw).show("raw",enlargement=True)
                Image(img=self.answers[i]).show("answer",enlargement=True)

                for k in range(h*b):
                    #小問内枠（数式，数値など）ループ
                    grade = input("Right: 1, half: 2, Wrong: 0")
                    try:
                        grade = int(grade)
                        ps = (s[0] + int((e[0]-s[0]) * ((k%b)/b)), s[1]+ int((e[1]-s[1]) * (int(k/b)/h)))
                        pe = (s[0] + int((e[0]-s[0]) * ((k%b+1)/b)), s[1]+ int((e[1]-s[1]) * (int(k/b+1)/h)))
                        r = int((pe[1]-ps[1])*0.15)
                        g = (pe[0] - int(r*1.5), ps[1] + int(r*1.5))
                        #採点箇所を復元
                        current[ps[1] : pe[1], ps[0] : pe[0]] = raw[ps[1] : pe[1], ps[0] : pe[0]].copy()
                        if grade == 1:
                            cv2.circle(current, g, r, (0, 0, 255), thickness=2)
                        if grade == 0:
                            cv2.line(current, (g[0]-r, g[1]-r), (g[0]+r, g[1]+r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                            cv2.line(current, (g[0]-r, g[1]+r), (g[0]+r, g[1]-r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                        if grade == 2:
                            points = np.array([(g[0], g[1]- r), (g[0] - r, g[1] + r), (g[0] + r, g[1] + r)])
                            cv2.polylines(current, [points], True, (0, 0, 255), 2)
                            grade = 0.5
                        totalgrade += grade /(b*h)
                        dst = cv2.resize(buf, dsize=None, fx= 1300/buf.shape[0], fy=1300/buf.shape[0])

                        Image(img=dst).show("grading",enlargement=True)
                        Image(img=raw[ps[1] : pe[1], ps[0] : pe[0]]).show("rawextract")
                        Image(img=raw).show("raw",enlargement=True)
                        Image(img=current).show("current",enlargement=True)

                    except:
                        totalgrade = self.grade.at[j+1, str(self.student_num[i])]
                self.grade.at[j+1, str(self.student_num[i])] = totalgrade
                #保存
                self.result = current
                #元画像へ反映
                row = self.row
                self.answers[i] = current
                
                Image(img=self.answers[i]).show("answer")
                self.save(self.student_num[i])
                self.answers[i] = raw.copy()

class MarkQuestion(Question):
    def __init__(self, sheets, name, regions, column, score, correct_answers, partial_credit = []):
        """
        correct_answers: 正答。0開始の数値またはアルファベットで指定可能
        partial_credit:  該当正答が部分点か否か。0であれば通常正答，1であれば部分点として採点する。correct_answersと同じ形状で指定。
        """
        super().__init__(sheets, name, regions, score)
        self.col = column
        self.row = len(self.score)

        #正答文字列を数値へ変換
        def convert_str_to_int(n):
            if type(n) is list:
                for i, item in enumerate(n):
                    n[i] = ord(item)-65
            if type(n) is str:
                n = ord(n)-65
            return n

        self.correct = list(map(convert_str_to_int, correct_answers))
        self.partial = partial_credit
        self.num_list = []

    def grading(self, index=0, binary_threshold = 180, ratio_threshold = 0.2):
        """
        binary_threshold: マーク検出における２値化閾値
        ratio_threshold: マーク検出における黒面積比率の閾値
        """

        def list_to_array(input_list):#入力された正答または部分点指定リストを最大要素数まで-1で埋めて配列化する。
            #正答の最大要素数の計算
            max_elements = 1
            for element in input_list:
                if type(element) is list:
                    if max_elements < len(element):
                        max_elements = len(element)

            for j, element in enumerate(input_list):
                if max_elements > 1:
                    if type(element) is not list:
                        input_list[j] = [element, -1]
                    while max_elements > len(input_list[j]):
                        input_list[j].append(-1)

            # numpy array 化
            output_list = np.array(input_list, dtype=int, ndmin=2)
            if max_elements == 1:
                output_list = output_list.transpose()

            return output_list      

        super().grading()

        self.num_list = []
        for i, sheet in enumerate(self.answers):
            if index == 0 or self.student_num[i] == index:
                row = self.row
                mark_block = Marksheet(sheet, 0, 1, 0, 1, row, self.col)
                mark_block.read(binary_threshold, ratio_threshold, 0)
                result = np.array(mark_block.num_list)

                correct = list_to_array(self.correct)
                if self.partial == []:
                    partial = np.ones(correct.shape)
                else:
                    partial = list_to_array(self.partial) + 1

                buf = np.zeros(correct[:,0].shape)
                for j in range(correct.shape[1]):
                    print(correct[:,j])
                    print(partial[:,j])
                    #正解と一致する場合を指定PARTIALに，そうでない場合元の値を維持
                    buf = np.where((result == correct[:,j]), partial[:,j], buf) 
                    #print(buf)
                self.grade.iloc[:,i] = buf 

                #正解/不正解の書き込み
                for j in range(row):
                    grade = self.grade.iat[j,i]
                    x = sheet.shape[1]; y = sheet.shape[0]
                    s = (0, int( y * (j % row) / row ))
                    e = (x, int( y * ((j % row) / row + 1 / row )))
                    g = (s[0] + int((e[0] - s[0]) * 0.95), s[1] + int((e[1] - s[1]) * 0.50))
                    r = int((e[1] - s[1]) * 0.20)

                    if grade == 1:
                        cv2.circle(sheet, g, r, (0, 0, 255), thickness=2)
                    if grade == 0:
                        cv2.line(sheet, (g[0]-r, g[1]-r), (g[0]+r, g[1]+r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(sheet, (g[0]-r, g[1]+r), (g[0]+r, g[1]-r), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    if grade == 2:
                        points = np.array([(g[0], g[1]- r), (g[0] - r, g[1] + r), (g[0] + r, g[1] + r)])
                        cv2.polylines(sheet, [points], True, (0, 0, 255), 2); self.grade.iat[j,i] = 0.8

                self.scoring()

                cv2.rectangle(sheet, (0,0), (30,30), (255, 255, 255), thickness=-1)        
                cv2.putText(sheet, str(self.sumscore[i]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                self.save(i)
                Image(img=sheet).show("grading",enlargement=True)

                self.num_list.append(mark_block.num_list)

    def get_correct_answer(self, index=99):
        """指定出席番号(初期値99)を解答例として読み込む"""
        sheet = self.answers[-1]
        row = self.row
        mark_block = Marksheet(sheet, 0, 1, 0, 1, row, self.col)
        mark_block.read(180, 0.2, 0)
        #読み取り結果を正答として保存
        self.correct = mark_block.num_list

    def save(self, index):
        #学生毎の採点画像を保存
        path = f"{self.basepath}/{self.name}/{str(self.student_num[index]).zfill(2)}"
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            print('Folder is not exist...')
            os.makedirs(folder)

        Image(img=self.answers[index]).save(path,True)
                
        #問題毎の採点集計を保存
        path = f"{self.basepath}/{self.name}"
        self.grade.to_csv(f"{path}.csv")

    def update(self):
        print(f"{self.name} updating...")
        for i in range(len(self.sheets)):
            #結果集約用ファイルの読み込み
            path = f"{self.basepath}/result/{str(self.student_num[i]).zfill(2)}.jpg"
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                print('Folder is not exist...')
                os.makedirs(folder)
            if not os.path.exists(path):
                path = f"{self.basepath}/concat/{str(self.student_num[i]).zfill(2)}.jpg"
            buf = Image(path).img
            
            #採点結果ファイルの読み込み
            path = f"{self.basepath}/{self.name}/{str(self.student_num[i]).zfill(2)}.jpg"
            sheet = Image(path).img

            y = buf.shape[0]; x = buf.shape[1]
            
            sx0 = 0; sy0 = 0
            for region in self.regions:
                ys = int(y * region[0][1]); ye = int(y * region[1][1])
                xs = int(x * region[0][0]); xe = int(x * region[1][0])
                buf[ys : ye, xs : xe] = sheet[sy0 : sy0+ye-ys, sx0 : xe-xs]
                sy0 = ye-ys

            Image(img=buf).save(f"{self.basepath}/result/{str(self.student_num[i]).zfill(2)}",True)
