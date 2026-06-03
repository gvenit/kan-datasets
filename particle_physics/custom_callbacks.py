class UpdatableFloat(float):
    def __new__(cls, x = 0):
        inst = super().__new__(cls, x)
        inst._value = [x]
        return inst

    def __iadd__(self, other):
        self._value[0] += other
        return self

    def __imul__(self, other):
        self._value[0] *= other
        return self

    def __float__(self):
        return float(self._value[0])
    
    def __repr__(self):
        return f"UpdatableFloat({self._value[0]})"
    
    def set(self, x):
        self._value[0] = x
        
    def value(self):
        return self._value[0]