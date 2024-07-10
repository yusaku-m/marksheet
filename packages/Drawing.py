from .Image import Image 
from .Marksheet import Marksheet
import cv2

#サブ課題用クラス
class SubSubject(Image):
    def __init__(self, path):
        super().__init__(path)
        #横画像は縦にする。
        if self.img.shape[1] > self.img.shape[0]:
            self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        self.img = cv2.resize(self.img, dsize=(210*4, 297*4))
        #上下判定
        s_ratio = 0.05
        e_ratio = 0.12
        b_img, __g_img, __r_img = cv2.split(self.img)
        top_img = b_img[int(self.img.shape[0]*s_ratio) : int(self.img.shape[0]*e_ratio),0:self.img.shape[1],]
        bot_img = b_img[int(self.img.shape[0]*(1-e_ratio)) : int(self.img.shape[0]*(1-s_ratio)),0:self.img.shape[1],]
        if cv2.mean(top_img) > cv2.mean(bot_img):
            self.img = cv2.rotate(self.img, cv2.ROTATE_180)
        cv2.imshow("top", top_img)
        cv2.waitKey(1)
        cv2.imshow("bot", bot_img)
        cv2.waitKey(1)

    def get_student(self):
        mark_block = Marksheet(self.img, 0.1, 0.36, 0.05, 0.12, 2, 10)
        mark_block.read(180, 0.15, 0)
        student_num = mark_block.num_list[0]*10 + mark_block.num_list[1]*1
        return student_num

    def get_subject(self):
        mark_block = Marksheet(self.img, 0.6, 1, 0.05, 0.12, 1, 10)
        mark_block.read(180, 0.15, 0)
        subject_num = mark_block.num_list[0]
        return subject_num
