
from __future__ import annotations
from typing import Optional

class Tensor:
  
    def __init__(self, value: float, name: str = None):
        self.value = float(value)
        
        #store args references when they become available, using built-in Python tuple, to build up the computation graph
        self.args: tuple[Tensor] = None # old version was ()  

        #store partial derivatives of the inputs to use in derivative calculations on the backprob path
        self.local_derivatives: tuple[Tensor] = None #old version was ()

        #placeholder to store the derivative of the final output wrt this tensor, to be computed during backpropagation
        self.derivative: Optional[Tensor] = None
        #self.derivative = Tensor(0.0)  # possible swap to try later later to try to eliminate Optional

        #Optional name for this Tensor
        self.name = name


    # build topological order
    #
    # requires as input:
    # order: list[Tensor] = [] # will contain the Tensors in topological order
    # visited: set[Tensor] = set() # used to track seen Tensors while building the topological list
 
    def build_topo(self, order:list[Tensor], visited:set[Tensor]):
        if self in visited:
            return
        visited.add(self)

        if self.args is not None:
            for arg in self.args:
                arg.build_topo(order, visited)

        order.append(self)  
 

    def backward(self):
        self.derivative = 1.0

        order: list[Tensor] = []
        visited: set[Tensor] = set()
        self.build_topo(order,visited)

        for node in reversed(order):
            if node.args is not None:
                for arg, local_derivative in zip(node.args, node.local_derivatives):
                    contrib = node.derivative * local_derivative.value
                    if arg.derivative is None:
                        arg.derivative = contrib
                    else:
                        arg.derivative += contrib

    def clear_derivatives(self):
        order: list[Tensor] = []
        visited: set[Tensor] = set()
        self.build_topo(order, visited)
    
        for node in order:
            node.derivative = None


    def __repr__(self) -> str:
        return f"Tensor(name={self.name}, value={self.value})"

    def __add__(self, x: Tensor |int|float):
        if isinstance(x, Tensor):
            return _add(self,x)
        elif isinstance(x, (int, float)):
            return _add(self,Tensor(x))
        else:
            return NotImplemented

    def __radd__(self, x:Tensor|int|float):
        if isinstance(x, Tensor):
            return _add(x,self)
        elif isinstance(x, (int, float)):
            return _add(Tensor(x), self)
        else:
            return NotImplemented

    def __sub__(self, x:Tensor|int|float):
        if isinstance(x, Tensor):
            return _sub(self,x)
        elif isinstance(x, (int, float)):
            return _sub(self,Tensor(x))
        else:
            return NotImplemented

    def __rsub__(self, x:Tensor|int|float):
        if isinstance(x, Tensor):
            return _sub(x,self)
        elif isinstance(x, (int, float)):
            return _sub(Tensor(x), self)
        else:
            return NotImplemented

    def __mul__(self, x:Tensor|int|float):
        if isinstance(x, Tensor):
            return _mul(self,x)
        elif isinstance(x, (int, float)):
            return _mul(self,Tensor(x))
        else:
            return NotImplemented

    def __rmul__(self, x:Tensor|int|float):
        if isinstance(x, Tensor):
            return _mul(x,self)
        elif isinstance(x, (int, float)):
            return _mul(Tensor(x), self)
        else:
            return NotImplemented


def _add(a:Tensor, b:Tensor):
    result = Tensor(a.value + b.value)
    result.local_derivatives = (Tensor(1), Tensor(1))
    result.args = (a,b)   
    return result

def _sub(a:Tensor, b:Tensor):
    result = Tensor(a.value - b.value)
    result.local_derivatives = (Tensor(1), Tensor(-1))
    result.args = (a,b)                           
    return result

def _mul(a:Tensor, b:Tensor):
    result = Tensor(a.value * b.value)
    result.local_derivatives = (b,a)
    result.args = (a,b)
    return result




def test(want: any, got: any):
    indicator = "✅" if want == got else "❌" 
    print(f"{indicator}: want {want}, got {got}")

    


    