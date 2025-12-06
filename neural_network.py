
import numpy as np
from micrograd_vector_module import Vector

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad=Vector([0.0]*len(p.data),requires_grad=False)
    def parameters(self):
        return []
class Neuron(Module):
    def __init__(self,nin,nonlin=True):
        self.nin=nin
        self.weight=Vector([np.random.uniform(-1,1) for _ in range(nin)])
        self.bias=Vector(0.0)
        self.nonlin=nonlin
    def __call__(self,xinput):
        act=self.weight.dot(xinput)+self.bias
        if self.nonlin:
            return act.sigmoid()
        return act
    def parameters(self):
        return [self.weight,self.bias]
    def __repr__(self,):
        return f"{'Sigmoid' if self.nonlin else 'linear'} with {self.nin} inputs"

class Layer(Module):
    def __init__(self,nin,nout,**kwargs):
        self.neurons=[Neuron(nin,**kwargs) for _ in range(nout)]
        self.nin=nin
        self.nout=nout
    def __call__(self,xinput):
        out_vectors=[n(xinput) for n in self.neurons]
        out_list=[o.data[0] for o in out_vectors]
        act=Vector(out_list,_children=tuple(out_vectors),_op='concat')
        def _backward():
            if act.requires_grad:
                for i, child in enumerate(out_vectors):
                    if child.requires_grad:
                        child.grad.data[0] += act.grad.data[i]                  
        act._backward = _backward
        return act
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    def __repr__(self):
        return f'layer:{self.nin}inputs --> {self.nout}outputs'


class MLP(Module):
    def __init__(self,nin,nouts):
        self.nin=nin
        self.nouts=nouts
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
    def __call__(self,xinput):
        x=xinput
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def __repr__(self):
        return f'Input layer:{self.nin}:{[nout for nout in self.nouts]}'
