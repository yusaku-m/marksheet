import os
import re
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
                 unit_instances = [],
                 corrects = [], 
                 allocations = [], 
                 question_classes = [], 
                 partial_score_ratio = 0
                 ):

        self.marks = marks
        self.variables = variables
        self.unit_instances = unit_instances
        self.corrects = corrects
        self.allocations = allocations
        self.question_classes = question_classes
        self.partial_score_ratio = partial_score_ratio
        self.questions = []
        self.score = 0


        self.update_variables_and_corrects()

        if len(self.question_classes) > 0:
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
                    marks =      self.marks[int(qr*i) : int(qr*(i+1))],
                    variables =  self.variables,
                    unit_instances = self.unit_instances,
                    correct =    self.corrects[i],
                    allocation = self.allocations[i]
                    )
                )
            
    def update_variables_and_corrects(self):
        """
        {name, value, unit, first difine}の辞書の不足要素を補う。
        """

        # valueのない要素にequationを基にした値を追加
        for var in self.variables:

            if ('value' not in var or 'unit' not in var) and ('equation' in var):

                var['value'], unit = eval_equation(var['equation'], self.variables)

                if unit is not None:
                    var['unit'] = unit
                    

        # correctsの計算
        for correct in self.corrects:
            if ('value' not in correct or 'unit' not in correct) and ('equation' in correct):
                correct['value'], unit = eval_equation(correct['equation'], self.variables)
                if unit is not None:
                    correct['unit'] = unit
                        
            #print(f"{correct['name']}: {str(correct['value'])[:30]} [{correct['unit'].value}] ({correct['equation']})")            

    def scoring(self):

        self.score=0

        try:
            for question in self.questions:
                question.scoring()
                self.score += question.score

        except Exception as e:
            print(f"part scoring error: {e}")

class Question():
    """小問"""
    def __init__(self, marks = [], 
                 variables = [], 
                 unit_instances = [],
                 correct = [], 
                 allocation = [],
                 partial_score_ratio = 0):
        
        self.marks = marks
        self.variables = variables
        self.unit_instances = unit_instances
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

        except Exception as e:
                print(f"question scoring error: {e}")

class Answer():
    """回答"""
    def __init__(self, marks = [], 
                 variables = [], 
                 unit_instances = [],
                 correct = [], 
                 allocation = [], 
                 partial_score_ratio = 0):
        self.marks = marks
        self.variables = variables
        self.unit_instances = unit_instances
        self.correct = correct
        self.allocation = allocation
        self.partial_score_ratio = partial_score_ratio

        #内部のステータス変数
        self.score = 0
        self.string = "" #回答を示す文字列

    def __str__(self):
        if self.string == "":
            self.string = self.get_answer_string()
        
        return self.string
    
    def __repr__(self):
        return self.__str__()
    
    def scoring(self):
        """
        採点
        """
        pass

    def get_answer_string(self):
        """
        回答の文字列を取得
        """
        return ""


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

    def __init__(self, marks = [], unit_instances = [], corrects = [], allocations = []):
        super().__init__(marks=marks, unit_instances=unit_instances, corrects=corrects, allocations=allocations)
        for q in range(len(allocations)):
            self.questions.append(DualNumberQuestion(marks=marks[q*4:q*4+4], unit_instances = unit_instances, correct=corrects[q], allocation=allocations[q]))

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
    def __init__(self, marks = [], unit_instances = [], correct = [], allocation = []):
        super().__init__(marks=marks, unit_instances=unit_instances, correct=correct, allocation=allocation)
        self.answers.append(NumberAnswer(marks=marks[0:2], unit_instances=unit_instances, correct=correct[0], allocation=allocation/2))
        self.answers.append(NumberAnswer(marks=marks[2:4], unit_instances=unit_instances, correct=correct[1], allocation=allocation/2))
        self.answers.append(NumberAnswer(marks=marks[0:2], unit_instances=unit_instances, correct=correct[1], allocation=allocation/2))
        self.answers.append(NumberAnswer(marks=marks[2:4], unit_instances=unit_instances, correct=correct[0], allocation=allocation/2))

    def scoring(self):
        self.score = 0

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

class EquationQuestion(Question):
    """
    variables: 変数一覧，マークシートのマーク順に並べる，マークにない変数はそれ以降に，
    変数名，単位，最初に定義される小問番号（本文の場合0）で構成
    
    correct:数式の文字列，動的な変数名は必ず両側へスペースを加える
    """
    def __init__(self, marks = [], variables = [], unit_instances = [], correct = [], allocation = []):
        super().__init__(marks=marks, variables=variables, unit_instances=unit_instances ,correct=correct, allocation=allocation)
        self.answers.append(EquationAnswer(marks=marks[0:3], variables=variables, correct=correct, allocation=allocation))

class EquationAndNumberQuestion(Question):
    """
    variables: 変数一覧，マークシートのマーク順に並べる，マークにない変数はそれ以降に，
    変数名，単位，最初に定義される小問番号（本文の場合0）で構成
    
    correct:数式の文字列，動的な変数名は必ず両側へスペースを加える
    """
    def __init__(self, marks = [], variables = [], unit_instances = [], correct = [], allocation = []):
        super().__init__(marks=marks, variables=variables, unit_instances=unit_instances, correct=correct, allocation=allocation)
        self.answers.append(EquationAnswer(marks=marks[0:3], variables=variables, unit_instances=unit_instances, correct=correct, allocation=allocation/2))
        self.answers.append(NumberAnswer(marks=marks[3:5], variables=variables, unit_instances=unit_instances, correct=correct, allocation=allocation/2))

"""
answers=============================================================================================================================
"""

class SingleAlphabetAnswer(Answer):
    """
    単一アルファベット回答
    correct:単一文字列
    """

    def scoring(self):
        
        correct = ord(self.correct) - ord('A')
        number = np.array(range(len(self.marks)))
        answer = np.sum(number * np.array(self.marks))
        #print(answer, correct)
        
        if answer == correct:
            self.score = self.allocation
        else:
            self.score = 0
    
    def get_answer_string(self):
        """
        回答の文字列を取得
        """
        number = np.array(range(len(self.marks)))
        answer = chr(np.sum(number * np.array(self.marks)) + 65)
        return answer

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

    def scoring(self):
        self.score = 0
        answer_string = str(self)

        number = int(answer_string.split('e')[0])
        exponent = int(answer_string.split('e')[1].split(' ')[0])
        unit =  self.get_unit()

        #正答の取得
        # 浮動小数点型の値を1桁の有効桁数で文字列に変換
        formatted_value = format(self.correct['value'], ".0e")

        # フォーマットされた文字列から数値部分を抽出
        correct_number = int(formatted_value.split('e')[0])
        correct_exponent = int(formatted_value.split('e')[1])

        correct_unit = self.correct['unit']

        #数値採点
        if number == correct_number:
            self.score += self.allocation / 3

        elif abs(number) == abs(correct_number):
            self.score += self.allocation / 3 * self.partial_score_ratio

        #桁採点
        if exponent == correct_exponent:
            self.score += self.allocation / 3

        elif (exponent  <= correct_exponent + 1) and (exponent  >= correct_exponent - 1):
            self.score += self.allocation / 3 * self.partial_score_ratio

        #単位採点
        if unit == correct_unit:
            self.score += self.allocation / 3

        print("correct:", float(correct_number) * 10 ** float(correct_exponent) , correct_unit," ===== answer:", float(number) * 10 ** float(exponent) , unit)

    def get_answer_string(self):
        """
        化数
        """
        #符号の取得
        if self.marks[0, 2] == 1:
            number_sign = -1
        else:
            number_sign = 1
        
        #値の取得
        number = sum(self.marks[0, 3:13] * np.array(range(10))) * number_sign

        """
        指数
        """
        #符号の取得
        if self.marks[0,15] == 1:
            exponent_sign = -1
        else:
            exponent_sign = 1
        #値の取得
        prefixes = [-9, -6, -3, 3, 6, 9]
        exponent = sum(self.marks[0, 16:25] * np.array(range(1,10))) * exponent_sign + sum(self.marks[1, 2:8] * np.array(prefixes))

        return f"{number}e{exponent} {self.get_unit()}"
    
    def get_unit(self):

        unit_answers = self.marks[1, 10:27:2]
        unit= Unit.Unit()

        for i, unit_answer in enumerate(unit_answers):
            
            if unit_answer == 1:
                unit = unit * self.unit_instances[i]

        return unit

class EquationAnswer(Answer):
    """
    数式回答。正解要素は以下ｘ要素，各要素で配点を当分
    ・文字式（値が一致する組み合わせ）
    
    部分点は，以下場合に対してつける
    ・文字式の次元が一致


    variables:名前，数式，値，単位を含む辞書型
    correct:名前，数式，値，単位を含む辞書型
    """

    def scoring(self):
        self.score = 0

        answer_string = str(self)
        
        value, unit = eval_equation(answer_string, self.variables)
        
        #数値一致？（完答）
        if np.allclose(np.array(self.correct['value']), np.array(value)):
            self.score += self.allocation
        elif unit == self.correct['unit']: #次元一致？（部分点）
            self.score += self.allocation * self.partial_score_ratio


    def get_answer_string(self):

        """回答の文字列取得"""
        #符号取得
        if self.marks[0, 3] == 1:
            number_sign = -1
        else:
            number_sign = 1
        
        #根号取得
        if self.marks[0, 4] == 1:
            number_sqrt = True
        else:
            number_sqrt = False

        #係数（分子）取得
        number_child = (sum(self.marks[0, 5:13] * np.array(range(1,9)))+1) * number_sign
        if number_sqrt is True:
            number_child = np.sqrt(number_child)

        #係数（分母）取得
        number_mother = (sum(self.marks[0, 17:25] * np.array(range(1,9)))+1)

        #係数（範囲上限フラグ）取得
        number_error = (self.marks[0, 13] == 1) or (self.marks[0, 25] == 1)

        #文字式（分子）取得
        if number_error == 1:
            number_child, number_mother= get_coefficient(self.correct['equation'],self.variables)

        equation = f"{number_child} / {number_mother} "

        for i, char in enumerate(self.marks[1, 4:26:2]):
            if char == 1:
                equation += f"* {self.variables[i]['name']} "

        #文字式（分母）取得
        for i, char in enumerate(self.marks[2, 4:26:2]):
            if char == 1:
                equation += f"/ {self.variables[i]['name']} "  
        
        return equation.replace("1 /","").replace("1 *","")

def eval_equation(equation_string, variables):
    """
    文字列型で入力された数式から値と単位を返す
    """
    equation = equation_string
    #print(equation)
    
    # 変数名とその値，式，単位を辞書に格納
    value_dict = {}; equation_dict = {}; unit_dict = {} 
    for var in variables:
        if 'value' in var:
            value_dict[var['name']] = var['value']
        if 'equation' in var:
            equation_dict[var['name']] = var['equation']
        if 'unit' in var:
            unit_dict[var['name']] = var['unit']

    #数式へ置換
    for key, value in equation_dict.items():
        equation = equation.replace(f" {key} ", f"({str(value)})")

    #値へ置換
    for key, value in value_dict.items():
        equation = equation.replace(f" {key} ", str(value))

    # unitを評価して単位を求める
    unit = equation_string
    for key, value in unit_dict.items():
        unit = unit.replace(f" {key} ", f"Unit.{str(value.__class__.__name__)}()")

    #print(equation)
    value = eval(equation)

    if "," not in unit:
        unit = eval(unit)
    else:
        unit = None      

    return value, unit

def get_coefficient(equation_string, variables):
    equation = equation_string

    # 変数名を正規表現で抽出
    var_pattern = r'\b(?:{})\b'.format('|'.join([re.escape(var["name"]) for var in variables]))
    
    # 式の中から変数名を置き換える
    equation = re.sub(var_pattern, '1', equation)
    
    # 分母と分子を分離
    parts = re.split(r'/', equation)
    
    # 分子を評価
    numerator = eval(parts[0])
    
    # 分母を評価（存在する場合）
    denominator = 1
    if len(parts) > 1:
        denominator = eval('/'.join(parts[1:]))
    
    return int(numerator), int(denominator)