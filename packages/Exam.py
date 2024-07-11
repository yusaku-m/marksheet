import os
import cv2
import math
import pandas as pd
import numpy as np

from .Image import Image
from .Marksheet import Marksheet

class Exam(Image):
    """
    path:単一学生の読み取り結果csv一覧，先頭パスに出席番号を含む
    """
    def __init__(self, path=[]):
        self.path = path
        self.student_num = 0         #出席番号
        self.parts = []
        self.marks = []
    
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

class Part():
    """大問"""
    def __init__(self, marks, collects, allocations):
        self.marks = marks
        self.collects = collects
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
            pass


class SingleAlphabetPart(Part):
    """
    アルファベット単一回答の大問
    """

    def __init__(self, marks, collects, allocations):
        super().__init__(marks, collects, allocations)
        for q in range(marks.shape[0]):
            self.questions.append(SingleAlphabetQuestion(marks[q], collects[q], allocations[q]))
        
class Question():
    """小問"""
    def __init__(self, marks, collect, allocation):
        self.marks = marks
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
            pass
        

class SingleAlphabetQuestion(Question):
    """
    アルファベット単一回答の小門
    """
    def __init__(self, marks, collect, allocation):
        super().__init__(marks, collect, allocation)
        self.answers.append(SingleAlphabetAnswer(marks, collect, allocation))
    

class Answer():
    """回答"""
    def __init__(self, marks, correct, allocation, partial_score_ratio = 0.5):
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

        

class Number(Answer):
    """数値回答"""
    def __init__(self, array):
        pass
    def scoring():
        pass

class Equation(Answer):
    """数式回答"""
    def __init__(self, array):
        pass
    def scoring():
        pass

