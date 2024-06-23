import ctypes
from math import sqrt
from enum import Enum, auto
from typing import Optional, Tuple, Union

lalg = ctypes.CDLL('./liblalg.so')


class c_array(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                ("shape", ctypes.POINTER(ctypes.c_int)),
                ("n_dims", ctypes.c_int),
                ("size", ctypes.c_int)]
    
# Function prototypes
lalg.create_array.restype = ctypes.POINTER(c_array)
lalg.create_array.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lalg.sum_arrays.restype = ctypes.POINTER(c_array)
lalg.sum_arrays.argtypes = [ctypes.POINTER(c_array), ctypes.POINTER(c_array)]
lalg.sum_scalar.restype = ctypes.POINTER(c_array)
lalg.sum_scalar.argtypes = [ctypes.POINTER(c_array), ctypes.c_float]
lalg.fill_array.argtypes = [ctypes.POINTER(c_array), ctypes.c_float]
lalg.prod_scalar.restype = ctypes.POINTER(c_array)
lalg.prod_scalar.argtypes = [ctypes.POINTER(c_array), ctypes.c_float]
lalg.pow_scalar.restype = ctypes.POINTER(c_array)
lalg.pow_scalar.argtypes = [ctypes.POINTER(c_array), ctypes.c_float]
lalg.exp_array.restype = ctypes.POINTER(c_array)
lalg.exp_array.argtypes = [ctypes.POINTER(c_array)]
lalg.multiply_matrix.restype = ctypes.POINTER(c_array)
lalg.multiply_matrix.argtypes = [ctypes.POINTER(c_array), ctypes.POINTER(c_array)]
lalg.multiply_vector.restype = ctypes.POINTER(c_array)
lalg.multiply_vector.argtypes = [ctypes.POINTER(c_array), ctypes.POINTER(c_array)]
lalg.sum_array_elements.restype = ctypes.c_float
lalg.sum_array_elements.argtypes = [ctypes.POINTER(c_array)]

class Ops(Enum):
    add_scalar = auto()
    add_tensors = auto()
    mult_scalar = auto()
    mult_matrix = auto()
    mult_vector = auto()
    pow = auto()
    exp = auto()
    
unary_ops = [Ops.pow, Ops.exp]
binary_ops = [Ops.add_scalar, Ops.add_tensors, Ops.mult_scalar, Ops.mult_matrix, Ops.mult_vector]
binary_scalar = [Ops.add_scalar, Ops.mult_scalar]

class Tensor:
    def __init__(self, shape, _c_array=None, _requires_grad=True, _op: Optional[Ops] = None, _from: Optional[Tuple["Tensor", Optional[Union["Tensor", float]]]] = None):
        # Array stuff
        self._shape = shape
        self._n_dims = len(self._shape)
        c_shape = (ctypes.c_int * len(self._shape))(*self._shape)
        self._array = lalg.create_array(c_shape, self._n_dims) if _c_array is None else _c_array

        # Autograd stuff
        self._requires_grad = _requires_grad
        self._grad = None
        self._op = None
        self._from = None
        if _requires_grad:
            self._op = _op
            if self._op is None:
                return
            assert _from is not None
            if self._op in unary_ops:
                self._from = [_from[0]]
            else:
                assert _from[1] is not None
                assert isinstance(_from[1], float) if _op in binary_scalar else isinstance(_from[1], Tensor)
                self._from = [_from[0], _from[1]]
                
        

    @staticmethod
    def _from_c_array(arr) -> "Tensor":
        shape = [arr[0].shape[i] for i in range(arr[0].n_dims)]
        return Tensor(shape, arr)
    
    @staticmethod
    def zeros(shape):
        t = Tensor(shape)
        lalg.fill_array(t._array, 0.0)
        return t
    
    @staticmethod
    def ones(shape):
        t = Tensor(shape)
        lalg.fill_array(t._array, 1.0)
        return t
        
    
    def __del__(self):
        lalg.cleanup_array(self._array)
        
    def __repr__(self):
        contents = [float(self._array[0].data[i]) for i in range(self._array[0].size)]
        return f"Tensor({contents})"
    
    def add(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            assert all([self._shape[i] == other._shape[i] for i in range(self._n_dims)]), "Shapes do not match"
            result = lalg.sum_arrays(self._array, other._array)
        elif isinstance(other, float) or isinstance(other, int):
            result = lalg.sum_scalar(self._array, other)
        else:
            raise ValueError("Invalid type for sum")
        
        if not bool(result):
            raise ValueError("Error in summing arrays")

        return Tensor._from_c_array(result)

    def multiply(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            if self._n_dims > 2 or other._n_dims > 2:
                raise NotImplementedError("Multiplication of two tensors is not implemented")
            
            if self._n_dims == 2 and other._n_dims == 2:
                result = lalg.multiply_matrix(self._array, other._array)
            elif self._n_dims == 2 and other._n_dims == 1:
                result = lalg.multiply_vector(self._array, other._array)
            
        elif isinstance(other, float) or isinstance(other, int):
            result = lalg.prod_scalar(self._array, other)
        else:
            raise ValueError("Invalid type for product")
        
        if not bool(result):
            raise ValueError("Error in multiplying arrays")

        return Tensor._from_c_array(result)
    
    def sub(self, other) -> "Tensor":
        return self.add(other.multiply(-1))
    
    def pow(self, other: float) -> "Tensor":
        result = lalg.pow_scalar(self._array, other)
        if not bool(result):
            raise ValueError("Error in raising array to power")

        return Tensor._from_c_array(result)
    
    def exp(self) -> "Tensor":
        result = lalg.exp_array(self._array)
        if not bool(result):
            raise ValueError("Error in exponentiating array")

        return Tensor._from_c_array(result)
    
    def sum(self) -> float:
        return lalg.sum_array_elements(self._array)
    
    def backwards(self, _first=True, ops=Optional[Ops], result=Optional["Tensor"]):
        if _first:
            self._grad = Tensor.ones(self._shape)
        else:
            pass



def mse_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return sqrt(y_true.sub(y_pred).pow(2).sum()) / y_true._shape[0]

if __name__ == "__main__":
    t1 = Tensor.ones((5,))
    t2 = Tensor.zeros((5,))
    t2.backwards()
    print(t2._grad)