from tensor import Tensor

x = Tensor(5, name="x")
c = Tensor(3, name="c")

y = 5 + x * (x + c)    # y = x^2 + c x + 5

y.backward()

print("y:", y.value)           # 45.0
print("dy/dx:", x.derivative)  # 2*5 + 3 = 13.0
print("dy/dc:", c.derivative)  # 5.0