import inspect

func = lambda num1, num2: num1 + num2
print(inspect.getsource(lambda num1, num2: num1 + num2).strip())  # Output: 'lambda num1, num2: num1 + num2'   