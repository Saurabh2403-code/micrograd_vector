# --- VERIFICATION ---
x = Vector([0.0, 3.0, -2.0, 1.0]) 
w = Vector([0.5, 0.5, 0.5, 0.5]) 
b = Vector([1.0]) 

# 1. Forward
val = x*w+b

# 2. Softmax & Loss
probs = val.softmax()
loss = probs[2].ln() * Vector([-1]) 

# 3. Backward
loss.backward()

print("Loss:", loss.data)
print("Bias Grad", b.grad.data) 
print("X Grad:", x.grad.data)
