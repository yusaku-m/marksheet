import os
import cv2
import math
import pandas as pd
import numpy as np

from .Image import Image
from .Marksheet import Marksheet
from . import Unit

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
    def __init__(self, marks = [], 
                 variables = [], 
                 unit_classes = [],
                 corrects = [], 
                 allocations = [], 
                 questions_classes = [], 
                 partial_score_ratio = 0
                 ):

        self.marks = marks
        self.variables = variables
        self.unit_classes = unit_classes
        self.corrects = corrects
        self.allocations = allocations
        self.questions_classes = questions_classes
        self.partial_score_ratio = partial_score_ratio
        self.questions = []
        self.score = 0

        if len(self.questions_classes) > 0:
            self.make_questions()
    
    def make_questions(self):
        """
        与えられた質問クラスリストで質問を作成する。
        """
        self.questions = []
        question_rownumber = len(self.marks) / len(self.question_classes)
        qr = question_rownumber

        for i, question_class in enumerate(self.question_classes):
            self.questions.append(
                question_class(
                    marks =      self.marks[qr*i : qr*(i+1)],
                    variables =  self.variables,
                    correct =    self.corrects[i],
                    allocation = self.allocations[i]
                    )
                )
            
    def update_variables_and_corrects(self):
        """
        {name, value, unit, first difine}の辞書の不足要素を補う。
        """

        # 変数名とその値，式，単位を辞書に格納
        value_dict = {}; equation_dict = {}; unit_dict = {} 
        for var in self.variables:
            if 'value' in var:
                value_dict[var['name']] = var['value']
            if 'equation' in var:
                equation_dict[var['name']] = var['equation']
            if 'unit' in var:
                unit_dict[var['name']] = var['unit']
        print(value_dict)
        print(equation_dict)
        print(unit_dict)

        # valueのない要素にequationを基にした値を追加
        for var in self.variables:
            if 'value' not in var and 'equation' in var:
                
                # equationを評価するために必要な変数を準備
                equation = var['equation']
                for key, value in value_dict.items():
                    equation = equation.replace(f" {key} ", str(value))

                # equationを評価して値を求める
                try:
                    var['value'] = eval(equation)

                except Exception as e:
                    print(f"Error evaluating equation for {var['name']}: {e}")

                # unitを評価して単位を求める
                unit = var["equation"]
                for key, value in unit_dict.items():
                    unit = unit.replace(f" {key} ", f"Unit.{str(value)}()")

                try:
                    var['unit'] = eval(unit)

                except Exception as e:
                    print(f"Error evaluating unit for {var['name']}. you can check or add unit list on Unit.py: {e}")
                    

        # correctsの計算
        for correct in self.corrects:
            equation = correct['equation']

            #数式へ置換
            for key, value in equation_dict.items():
                equation = equation.replace(f" {key} ", f"({str(value)})")

            #値へ置換
            for key, value in value_dict.items():
                equation = equation.replace(f" {key} ", str(value))

            try:
                correct['value'] = eval(equation)
                

            except Exception as e:
                print(f"Error evaluating equation for {correct['name']}: {e}")

            # unitを評価して単位を求める
            unit = correct["equation"]
            # ()は除去
            unit = unit.replace(f"(", f"")
            unit = unit.replace(f")", f"")

            #単位が存在する部分を置換
            for key, value in unit_dict.items():
                unit = unit.replace(f" {key} ", f"Unit.{str(value)}()")
            print(unit)

            try:
                correct['unit'] = eval(unit)

            except Exception as e:
                print(f"Error evaluating unit for {correct['name']}. you can check or add unit list on Unit.py: {e}")

            print(f"{correct['name']}: {str(correct['value'])[:30]} [{correct['unit'].value}] ({equation})")            

    def scoring(self):

        self.score=0

        try:
            for question in self.questions:
                question.scoring()
                self.score += question.score

        except:
            print("part scoring error")

class Question():
    """小問"""
    def __init__(self, marks = [], 
                 variables = [], 
                 unit_classes = [],
                 correct = [], 
                 allocation = [],
                 partial_score_ratio = 0):
        self.marks = marks
        self.variables = variables
        self.unit_classes = unit_classes
        self.correct = correct
        self.allocation = allocation
        self.partial_score_ratio = partial_score_ratio
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

class Answer():
    """回答"""
    def __init__(self, marks = [], 
                 variables = [], 
                 unit_classes = [],
                 correct = [], 
                 allocation = [], 
                 partial_score_ratio = 0):
        self.marks = marks
        self.variables = variables
        self.correct = correct
        self.allocation = allocation
        self.score = 0
        self.partial_score_ratio = partial_score_ratio

    def scoring(self):
        pass


"""
Parts=============================================================================================================================
"""

class SingleAlphabetPart(Part):
    """
    アルファベット単一回答の大問
    """

    def __init__(self, marks, corrects, allocations):
        super().__init__(marks=marks, corrects=corrects, allocations=allocations)
        for q in range(marks.shape[0]):
            self.questions.append(SingleAlphabetQuestion(marks=marks[q], correct=corrects[q], allocation=allocations[q]))

class DualNumberPart(Part):
    """
    二つの数値を回答する大問
    """

    def __init__(self, marks, corrects, allocations):
        super().__init__(marks=marks, corrects=corrects, allocations=allocations)
        for q in range(len(allocations)):
            self.questions.append(DualNumberQuestion(marks=marks[q*4:q*4+4], correct=corrects[q], allocation=allocations[q]))

"""
questions=============================================================================================================================
"""

class SingleAlphabetQuestion(Question):
    """
    アルファベット単一回答の小門
    """
    def __init__(self, marks, correct, allocation):
        super().__init__(marks=marks, correct=correct, allocation=allocation)
        self.answers.append(SingleAlphabetAnswer(marks=marks, correct=correct, allocation=allocation))
    
class DualNumberQuestion(Question):
    """
    2つの数値を回答する小問，順序自由
    """
    def __init__(self, marks, correct, allocation):
        super().__init__(marks=marks, correct=correct, allocation=allocation)
        self.answers.append(NumberAnswer(marks=marks[0:2], correct=correct[0], allocation=allocation/2))
        self.answers.append(NumberAnswer(marks=marks[2:4], correct=correct[1], allocation=allocation/2))
        self.answers.append(NumberAnswer(marks=marks[0:2], correct=correct[1], allocation=allocation/2))
        self.answers.append(NumberAnswer(marks=marks[2:4], correct=correct[0], allocation=allocation/2))

    def scoring(self):
        self.score=0

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

class EquationAndNumberQuestion(Question):
    """
    variables: 変数一覧，マークシートのマーク順に並べる，マークにない変数はそれ以降に，
    変数名，単位，最初に定義される小問番号（本文の場合0）で構成
    
    correct:数式の文字列，動的な変数名は必ず両側へスペースを加える
    """
    def __init__(self, marks = [], variables = [], correct = [], allocation = []):
        super().__init__(marks=marks, variables=variables, corrects=correct, allocations=allocation)
        self.answers.append(EquationAnswer(marks=marks[0:3], correct=correct[0], allocation=allocation/2))
        self.answers.append(NumberAnswer(marks=marks[3:5], correct=correct[1], allocation=allocation/2))

"""
answers=============================================================================================================================
"""

class SingleAlphabetAnswer(Answer):
    """
    単一アルファベット回答
    correct:単一文字列
    """
    def __init__(self, marks, correct, allocation):
        super().__init__(marks=marks, correct=correct, allocation=allocation)

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
        super().__init__(marks=marks, correct=correct, allocation=allocation, partial_score_ratio=partial_score_ratio)

    def scoring(self):
        self.score = 0

        #print("correct:", self.correct)

        prefixes = [-9, -6, -3, 3, 6, 9]
        unit_classes = [Unit.N(), Unit.m(), Unit.m2(), Unit.m3(), Unit.m4(), Unit.pers(), Unit.Pa(), Unit.K()]


        #正答の取得
        # 浮動小数点型の値を1桁の有効桁数で文字列に変換
        formatted_value = format(self.correct[0], ".0e")
        #print(formatted_value)

        # フォーマットされた文字列から数値部分を抽出
        correct_number = int(formatted_value.split('e')[0])
        correct_exponent = int(formatted_value.split('e')[1])

        correct_unit = self.correct[1]

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
        #単位が一致していれば正解
        unit_answers = self.marks[1, 8:24:2]
        
        unit= Unit.Unit()
        for i, unit_answer in enumerate(unit_answers):
            
            if unit_answer == 1:
                unit = unit * unit_classes[i]

        #採点
        #print(unit.value, correct_unit.value)
        if unit == correct_unit:
            self.score += self.allocation / 3

        #print("number:", number,  "exponent", exponent)
        #print("answer:", number * 10 ** exponent , unit)

class EquationAnswer(Answer):
    """
    数式回答。正解要素は以下ｘ要素，各要素で配点を当分
    ・文字式（値が一致する組み合わせ）
    ・係数
    
    部分点は，以下場合に対してつける
    ・文字式の次元が一致


    variables:名前，数式，値，単位を含む辞書型
    correct:名前，数式，値，単位を含む辞書型
    """
    def __init__(self, marks, variables, correct, allocation, partial_score_ratio = 0.5):
        super().__init__(marks=marks, variables=variables, correct=correct, allocation=allocation, partial_score_ratio=partial_score_ratio)

    def scoring(self):
        self.score = 0

        print("correct:", self.correct)




        #数式指定の変数定義
