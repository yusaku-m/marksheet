import numpy as np
import sys
import inspect

class Unit():
    """
    次元解析を想定したクラス。
    引数無しで初期化した場合は無次元
    値は基本単位次数のリスト形式
    各基本単位はsi基本単位の順番に対応する固有インデックスを持つ。
    インデックスの値が各基本単位の指数に相当する。
    """

    def __init__(self, index = -1, value = []):
        self.__value = []

        if value is None:
            value = []

        if index > -1:
            for i in range(index+1):
                if i == index:
                    self.__value.append(1)
                else:
                    self.__value.append(0)
        
        elif len(value) > 0:
            self.__value = value

        else:
            self.__value.append(0)

    @property
    def value(self):
        """
        自分自身の値を返す
        """
        return  self.__value

    def __str__(self):

        #単位定義一覧を取得
        module = sys.modules[__name__]

        class_names = [
            name for name, obj in inspect.getmembers(module) if inspect.isclass(obj)
            ]

        #定義のある単位名であれば該当名を返す        
        for class_name in class_names:
            if class_name != "Unit":
                instance = getattr(module, class_name)()

                if instance == self:
                    return instance.__class__.__name__.replace("non","-").replace("_per_","/").replace("per_","/").replace("per","/")

        #定義のない単位名であれば基本単位の組み合わせを返す 
        result = ""
        units = ["m","kg","s","A","K","mol","cd"]
        for unit, power in zip(units, self.value):
            if power != 0:
                if power == 1:
                    result += f"{unit}·"
                else:
                    result += f"{unit}^{power}·"

        return result
    
    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):
        """
        単位の積を出力
        """
        try:
            sv, ov = self.convert_to_same_size_numpy(other)      
            return Unit(value=list(sv + ov))
        except:
            return Unit(value=self.value)

    def __rmul__(self, other):
        """
        単位の積を出力（逆版）
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        単位の商を出力
        """

        try:
            sv, ov = self.convert_to_same_size_numpy(other)
            return Unit(value=list(sv - ov))
        except:
            return Unit(value=self.value)

    def __rtruediv__(self, other):
        """
        単位の商を出力（逆方向対応）
        """

        try:
            sv, ov = self.convert_to_same_size_numpy(other)
            return Unit(value=list(ov - sv))
        except:
            return Unit(value=self.value)

    def __pow__(self, value):
        """
        単位の累乗を出力
        """
        sv, __ov = self.convert_to_same_size_numpy(self)
        return Unit(value=list(sv * value))
    
    def __eq__(self, other):
        """
        単位同士が等しいか比較する
        """	
        sv, ov = self.convert_to_same_size_numpy(other)
        return np.allclose(sv, ov)

    def convert_to_same_size_numpy(self, other):
        """
        演算用のnumpy配列を得る
        """
        #一旦numpy配列に
        sv = np.array(self.value)
        ov = np.array(other.value)

        #サイズ合わせ
        pad  =  sv.shape[0] - ov.shape[0]
        if pad > 0:
            ov = np.pad(ov, (0, pad), mode='constant', constant_values=0)
        elif pad < 0:
            sv = np.pad(sv, (0, -pad), mode='constant', constant_values=0)

        return sv, ov

class non(Unit):
    """無次元の単位"""
    def __init__(self):
        super().__init__()

class m(Unit):
    """長さの単位"""
    def __init__(self):
        super().__init__(0)

class kg(Unit):
    """質量の単位"""
    def __init__(self):
        super().__init__(1)

class s(Unit):
    """時間の単位"""
    def __init__(self):
        super().__init__(2)

class K(Unit):
    """温度の単位"""
    def __init__(self):
        super().__init__(4)

class N(Unit):
    """力の単位"""
    def __init__(self):
        unit = kg() * m() / (s() ** 2)
        super().__init__(value=unit.value)

class Nm(Unit):
    """モーメントの単位"""
    def __init__(self):
        unit = N() * m()
        super().__init__(value=unit.value)

class Nm2(Unit):
    """曲げ剛性の単位"""
    def __init__(self):
        unit = N() * m() ** 2
        super().__init__(value=unit.value)

class N_per_m(Unit):
    """分布荷重の単位"""
    def __init__(self):
        unit = N() / m()
        super().__init__(value=unit.value)

class Pa(Unit):
    """圧力の単位"""
    def __init__(self):
        unit = N() / (m() ** 2)
        super().__init__(value=unit.value)

class m2(Unit):
    """面積の単位"""
    def __init__(self):
        unit = m() ** 2
        super().__init__(value=unit.value)

class m3(Unit):
    """体積の単位"""
    def __init__(self):
        unit = m() ** 3
        super().__init__(value=unit.value)

class m4(Unit):
    """断面二次モーメントの単位"""
    def __init__(self):
        unit = m() ** 4
        super().__init__(value=unit.value)
        
class pers(Unit):
    """毎秒の単位"""
    def __init__(self):
        unit = Unit() / s()
        super().__init__(value=unit.value)
