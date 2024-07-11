import os
import cv2
import math
import pandas as pd
import numpy as np

from .Image import Image
from .Marksheet import Marksheet

class Exam():
    """
    path:単一学生の読み取り結果csv一覧，先頭パスに出席番号を含む
    """
    def __init__(self, path=[]):
        self.path = path
        self.student_num = 0         #出席番号
        self.parts = []
        self.marks = []
        self.score = 0
    
    def get_marks(self):
        self.marks = []

        for path in self.path:
            self.marks.append(np.array(pd.read_csv(path, header=0, index_col=0)))

    def get_student(self, start_rc = (3, 7)):
        """
        2桁の出席番号を読み取る。マーク領域は2行10列で0-9の数値固定
        """

        number = np.array(range(10))
        number = np.vstack([number*10,number])
        
        start_r = start_rc[0]-2
        start_c = start_rc[1]-2

        buf = np.array(self.marks[0][start_r : start_r+2, start_c : start_c+10])
        
        self.student_num = np.sum(buf*number)
    
    def scoring(self):
        self.score=0
        try:
            for i, part in enumerate(self.parts):
                print("scoring start of Q", i+1)
                part.scoring()
                self.score += part.score

        except:
            print("exam scoring error")

class Part():
    """大問"""
    def __init__(self, marks, corrects, allocations):
        self.marks = marks
        self.corrects = corrects
        self.allocations = allocations
        self.questions = []
        self.score = 0
    
    def scoring(self):
        self.score=0
        try:
            for question in self.questions:
                question.scoring()
                self.score += question.score

        except:
            print("part scoring error")

class SingleAlphabetPart(Part):
    """
    アルファベット単一回答の大問
    """

    def __init__(self, marks, corrects, allocations):
        super().__init__(marks, corrects, allocations)
        for q in range(marks.shape[0]):
            self.questions.append(SingleAlphabetQuestion(marks[q], corrects[q], allocations[q]))

class DualNumberPart(Part):
    """
    二つの数値を回答する大問
    """

    def __init__(self, marks, corrects, allocations):
        super().__init__(marks, corrects, allocations)
        for q in range(len(allocations)):
            self.questions.append(DualNumberQuestion(marks[q*4:q*4+4], corrects[q], allocations[q]))
        
class Question():
    """小問"""
    def __init__(self, marks, correct, allocation):
        self.marks = marks
        self.correct = correct
        self.allocation = allocation
        self.answers = []
        self.score = 0

    def scoring(self):
        self.score=0
        try:
            for answer in self.answers:
                answer.scoring()
                self.score += answer.score

        except:
            print("question scoring error")        

class SingleAlphabetQuestion(Question):
    """
    アルファベット単一回答の小門
    """
    def __init__(self, marks, correct, allocation):
        super().__init__(marks, correct, allocation)
        self.answers.append(SingleAlphabetAnswer(marks, correct, allocation))
    
class DualNumberQuestion(Question):
    """
    2つの数値を回答する小問，順序自由
    """
    def __init__(self, marks, correct, allocation):
        super().__init__(marks, correct, allocation)
        self.answers.append(NumberAnswer(marks[0:2], correct[0], allocation/2))
        self.answers.append(NumberAnswer(marks[2:4], correct[1], allocation/2))
        self.answers.append(NumberAnswer(marks[0:2], correct[1], allocation/2))
        self.answers.append(NumberAnswer(marks[2:4], correct[0], allocation/2))

    def scoring(self):
        self.score=0
        try:
            scoreA = 0
            for answer in self.answers[:2]:
                answer.scoring()
                scoreA += answer.score

            scoreB = 0
            for answer in self.answers[2:]:
                answer.scoring()
                scoreB += answer.score

            if scoreB > scoreA:
                self.score = scoreB
                self.answers = self.answers[2:]
            else:
                self.score = scoreA
                self.answers = self.answers[:2]

        except:
            print("question scoring error")

class Answer():
    """回答"""
    def __init__(self, marks, correct, allocation, partial_score_ratio = 0):
        self.marks = marks
        self.correct = correct
        self.allocation = allocation
        self.score = 0
        self.partial_score_ratio = partial_score_ratio

    def scoring():
        pass

class SingleAlphabetAnswer(Answer):
    """
    単一アルファベット回答
    correct:単一文字列
    """
    def __init__(self, marks, correct, allocation):
        super().__init__(marks, correct, allocation)

    def scoring(self):
        
        correct = ord(self.correct) - ord('A')
        number = np.array(range(len(self.marks)))
        answer = np.sum(number * np.array(self.marks))
        #print(answer, correct)
        
        if answer == correct:
            self.score = self.allocation
        else:
            self.score = 0

class NumberAnswer(Answer):
    """
    数値回答
    正解要素は以下３要素，各要素で配点を当分
    ・化数
    ・指数と接頭辞を合わせた桁
    ・単位
    
    部分点は，以下の誤りに対してつける
    ・化数の符号
    ・桁±1

    correct:数値（m系統一,単位の文字列リスト）のタプル

    """
    def __init__(self, marks, correct, allocation, partial_score_ratio = 0.5):
        super().__init__(marks, correct, allocation, partial_score_ratio)

    def scoring(self):
        self.score = 0

        print("correct:", self.correct)

        prefixes = [-9, -6, -3, 3, 6, 9]
        unit_character = ["N", "m", "m^2", "m^3", "m^4", "/s", "Pa", "K"]
        unit_prime = [2, 3, 5, 7, 11, 13, 17, 19]


        #正答の取得
        # 浮動小数点型の値を1桁の有効桁数で文字列に変換
        formatted_value = format(self.correct[0], ".0e")
        #print(formatted_value)

        # フォーマットされた文字列から数値部分を抽出
        correct_number = int(formatted_value.split('e')[0])
        correct_exponent = int(formatted_value.split('e')[1])

        correct_unit = np.prod([unit_prime[unit_character.index(unit)] for unit in self.correct[1]])

        """
        化数
        """
        #符号の取得
        if self.marks[0, 0] == 1:
            number_sign = -1
        else:
            number_sign = 1
        
        #値の取得
        number = sum(self.marks[0, 1:11] * np.array(range(10))) * number_sign

        #採点
        if number == correct_number:
            self.score += self.allocation / 3

        elif abs(number) == abs(correct_number):
            self.score += self.allocation / 3 * self.partial_score_ratio

        """
        指数
        """

        #符号の取得
        if self.marks[0,13] == -1:
            exponent_sign = -1
        else:
            exponent_sign = 1

        #値の取得
        exponent = sum(self.marks[0, 14:23] * np.array(range(1,10))) * exponent_sign + sum(self.marks[1, 0:6] * np.array(prefixes))

        #採点
        if exponent == correct_exponent:
            self.score += self.allocation / 3

        elif (exponent  <= correct_exponent + 1) and (exponent  >= correct_exponent - 1):
            self.score += self.allocation / 3 * self.partial_score_ratio
        
        """
        単位
        """
        #各単位に素数を割り当てて，積を出し一致していれば
        unit_answer = self.marks[1, 8:24:2]
        #print(unit_answer)
        unit = np.prod(np.where(unit_answer == True, unit_prime, 1))

        #採点
        if unit == correct_unit:
            self.score += self.allocation / 3
            print("cu")

        elif (exponent  <= correct_exponent + 1) and (exponent  >= correct_exponent - 1):
            self.score += self.allocation / 3 * self.partial_score_ratio

        #print("number:", number,  "exponent", exponent)
        print("answer:", number * 10 ** exponent , unit)


class Equation(Answer):
    """数式回答"""
    def __init__(self, array):
        pass
    def scoring():
        pass

