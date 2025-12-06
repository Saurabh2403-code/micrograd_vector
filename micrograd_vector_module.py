import numpy as np


class Vector:
    def __init__(self,data=[],_children=(),_op='',requires_grad=True):
        self.data=data if isinstance(data,list) else [data]
        self._prev=set(_children)
        self._op=_op
        self.requires_grad=requires_grad
        if requires_grad:
            self.grad=Vector([0.0 for data in self.data],requires_grad=False)
        else:
            self.grad=None
        self._backward=lambda:None

    def __repr__(self):
        return f'{self.data}'


    def __add__(self,other):
        other=other if isinstance(other,Vector) else Vector(other)
        self_data=self.data
        other_data=other.data
        if len(self_data)!=len(other_data):
            if len(self_data)==1:
                self_data=[self_data[0]]*len(other_data)
            if len(other_data)==1:
                other_data=[other_data[0]]*len(self_data)
        out=Vector([x1+x2 for x1,x2 in zip(self_data,other_data)],(self,other),'+')
        def _backward():
            if self.requires_grad:
                if len(self.data)==1 and len(out.data)>1:
                    self.grad+=out.grad.sum()  #[sum(1.0*out.grad) for _ in range(len(self.data))]
                else:
                    self.grad+=out.grad
            if other.requires_grad:
                if len(other.data)==1 and len(out.data)>1:
                    other.grad+=out.grad.sum()   #[sum(1.0*out.grad) for _ in range(len(self.data))]
                else:
                    other.grad+=out.grad
        out._backward=_backward
        return out


    def __mul__(self,other):
        other=other if isinstance(other,Vector) else Vector(other)
        self_data=self.data
        other_data=other.data
        if len(self_data)!=len(other_data):
            if len(self_data)==1:
                self_data=[self_data[0]]*len(other_data)
            if len(other_data)==1:
                other_data=[other_data[0]]*len(self_data)
        out=Vector([x1*x2 for x1,x2 in zip(self_data,other_data)],(self,other),'elementwise product')
        def _backward():
            if self.requires_grad:
                if len(self.data)==1 and len(out.data)>1:
                    self.grad+=(out.grad*other).sum()
                else:
                    self.grad+=out.grad*other
            if other.requires_grad:
                if len(other.data)==1 and len(out.data)>1:
                    other.grad+=(out.grad*self.data).sum()
                else:
                    other.grad+=out.grad*self.data
        out._backward=_backward
        return out




        # if len(self.data)==len(other.data):
        #     out=Vector([x1*x2 for x1,x2 in zip(self.data,other.data)],(self,other),'elementwise product')
        #     def _backward():
        #         if self.requires_grad:
        #             self.grad+=out.grad*other
        #         if other.requires_grad:
        #             other.grad+=out.grad*self.data
        #     out._backward=_backward
        #     return out
        # elif len(other.data)==1:
        #     other=Vector([other.data[0] for _ in range(len(self.data))])
        #     out=self*other#Vector([x1*x2 for x1,x2 in zip(self.data,[other.data[0] for _ in range(len(self.data))])],(self,other),'scalar product')
        #     def _backward():
        #         if self.requires_grad:
        #             self.grad+=out.grad*other
        #         if other.requires_grad:
        #             other.grad+=out.grad*self.data
        #     return out


    def __pow__(self,other):
        out= Vector([data**other for data in self.data],(self,),f'**{other}')
        def _backward():
            if self.requires_grad:
                term1=(self**(other-1))*other
                self.grad+=term1*out.grad
        out._backward=_backward
        return out


    def relu(self):
        out=Vector([self.data[i]*(self.data[i]>0) for i in range(len(self.data))],(self,),'ReLu')
        def _backward():
            if self.requires_grad:
                self.grad+=Vector([1 if self.data[i]>0 else 0 for i in range(len(self.data))],requires_grad=False)*(out.grad)
        out._backward=_backward
        return out


    def sigmoid(self):
        out=Vector([1/(1+np.exp(-(self.data[i]))) for i in range(len(self.data))],(self,),'Sigmoid')
        sigmoid_prime=Vector([np.exp(-(self.data[i]))/((1+np.exp(-(self.data[i])))**2) for i in range(len(self.data))],(self,),'Sigmoid')
        def _backward():
            if self.requires_grad:
                self.grad+=sigmoid_prime*out.grad
        out._backward=_backward
        return out


    def exp(self):
        out=Vector([np.exp(data) for data in self.data],(self,),'e**x')
        def _backward():
            if self.requires_grad:
                self.grad+=(out)*out.grad
        out._backward=_backward
        return out


    def ln(self):
        out=Vector([np.log(self.data[i]) for i in range(len(self.data))],(self,),'ln(x)')
        def _backward():
            if self.requires_grad:
                self.grad+=(self**-1)*out.grad
        out._backward=_backward
        return out


    def sum(self):
        sum=0
        for i in range(len(self.data)):
            sum+=self.data[i]
        out=Vector([sum],(self,),'sum')
        def _backward():
            if self.requires_grad:
                self.grad+=out.grad
        out._backward=_backward
        return out
    def __getitem__(self,index):
        out=Vector([self.data[index]],(self,),f'Indexing:{index}')
        def _backward():
            if self.requires_grad:
                self.grad+=out.grad*Vector([1 if i==index else 0 for i in range(len(self.data))],requires_grad=False)
        out._backward=_backward
        return out
    def __len__(self):
        return len(self.data)

    def dot(self,other):
        other=other if isinstance(other,Vector) else Vector(other)
        if len(self.data)==len(other.data):
            prod=self*other 
            out=prod.sum()
            # def _backward():
            #     if self.requires_grad:
            #         self.grad+=other*out.grad
            #     if other.requires_grad:
            #         other.grad+=self*out.grad
            # out._backward=_backward
            return out
        print("Array must be of same length")
    def __rmul__(self,other):
        return self*other
    def __radd__(self,other):
        return self+other
    def softmax(self):
        counts = self.exp()
        denominator = counts.sum()
        out = counts*(denominator**-1)
        return out


    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = Vector([1.0]*len(self.data),requires_grad=False)
        for v in reversed(topo):
            v._backward()    





