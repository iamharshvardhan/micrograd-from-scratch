import math

class Value():

    """ stores a single scalar value, its label and its gradient """
    
    # Initialize the data using 'Value()' data structure
    def __init__(self, data, sub_value=(), op="", label=""):
        self.data = data
        self.prev = set(sub_value)
        self._backward = lambda: None
        self.op = op
        self.label = label
        self.grad = 0  # no effect
    

    # Add two distinct 'Value()' data types
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)    # add any int values to Value
        out = Value(self.data + other.data, (self, other), op="+")
        
        # Backpropogate addition function to find the gradient of the previous nodes
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    

    # Multiply two distinct 'Value()' data types
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)    # multiply any int values to Value
        out = Value(self.data * other.data, (self, other), op="*")
        
        # Backpropogate addition function to find the gradient of the previous nodes
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        
        return out
    

    # Raise the data to the power
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        # Backpropogate power function to find the gradient of the previous nodes
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    

    # Find the tanh() of the neuron [gives the activation number]
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        # Backpropogate tanh function to find the gradient of the previous nodes
        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
        
        return out  
    

    def backward(self):
        # implementing topoligical sort
        topo = []
        visited = ()
        def build_topo(v):
            if v not in visited:
                for sub in v.prev:
                    build_topo(sub)
                topo.append(v)
        build_topo(self)
        
        # Base case
        self.grad = 1.0
        # calling it for all the previous values for backpropogating
        for node in reversed(topo):
            node._backward()

    
    # Reverse the data and add when necessary (primary data type + Value() data type)
    def __radd__(self, other):    # other + self
        return self + other
    

    # Subtract two distinct 'Value()' data types
    def __sub__(self, other): # self - other
        return self + (-other)


    # Reverse the data and subtract when necessary (primary data type - Value() data type)
    def __rsub__(self, other):    # other - self
        return self - other

    
    # Reverse the data and multiply when necessary (primary data type * Value() data type)
    def __rmul__(self, other):    # other * self
        return self * other
    
    
    # Divide two distinct 'Value()' data types
    def __truediv__(self, other):
        return self * other**-1


    # Divide two distinct 'Value()' data types
    def __rtruediv__(self, other):
        return other * self**-1
    

    # Negation of the 'Value()' data type
    def __neg__(self):
        return self * -1
    
    
    # Output the value when called
    def __repr__(self):
        return f"Value(data={self.data})"
