import numpy as np
from StackGP import *
import StackGP as sgp
import time
from collections import deque

def evModHelper2(varStack, opStack, tempStack, data):
    stack1 = deque(reversed(varStack))
    stack2 = deque(reversed(opStack))
    stack3 = deque(reversed(tempStack))

    while stack2:
        op = stack2.pop()
        if callable(op):
            patt = getArity(op)
            while patt > len(stack3):
                stack3.append(stack1.pop()) 
            try:
            #print(list(stack3)[-patt:][::-1], op, patt, stack3,"\n")
                temp = op(*varReplace(list(stack3)[-patt:][::], data))
            #print("Temp: ", temp, "\n")
            except (TypeError, OverflowError):
                temp = np.nan
            #    print("Error", list(stack3)[patt:][::-1], op, patt, stack3)
            #    list(stack3)[patt:][::-1]

            [stack3.pop() for i in range(patt)]
            stack3.append(temp)
        else:
            if len(stack1)>0:
                stack3.append(*varReplace([stack1.pop()],data))
    return [stack1, stack2, stack3]

def evModHelper(varStack, opStack, tempStack, data):
    stack1 = varStack
    stack2 = opStack
    stack3 = tempStack

    while len(stack2)>0:
        op = stack2[0]
        stack2 = stack2[1:]

        if callable(op):
            patt = getArity(op)
            while patt > len(stack3):
                stack3=[stack1[0]]+stack3
                stack1=stack1[1:]
            try:
                #print(list(stack3)[-patt:][::-1], op, patt, stack3,"\n")
                temp = op(*varReplace(stack3[:patt][::-1], data))
            except (TypeError, OverflowError):
                temp = np.nan
            stack3 = stack3[patt:]
            stack3.insert(0, temp)
        else:
            if len(stack1)>0:
                stack3=varReplace([stack1[0]],data)+stack3
                stack1=stack1[1:]
    return [stack1, stack2, stack3]

def evaluateGPModel(model,inputData): #Evaluates a model numerically
    response=evModHelper(model[1],model[0],[],np.array(inputData).astype(float))[2][0]
    if not type(response)==np.ndarray and inputLen(inputData)>1:
        response=np.array([response for i in range(inputLen(inputData))])
    return response

def evaluateGPModel2(model,inputData): #Evaluates a model numerically
    response=evModHelper2(model[1],model[0],[],np.array(inputData).astype(float))[2][0]
    if not type(response)==np.ndarray and inputLen(inputData)>1:
        response=np.array([response for i in range(inputLen(inputData))])
    return response

def printGPModel(mod,inputData=symbols(["x"+str(i) for i in range(100)])): #Evaluates a model numerically
    def inv1(a):
        return a**(-1)
    from sympy import tan as tan1, exp as exp1, sqrt as sqrt1, sin as sin1, cos as cos1, acos, asin, atan, tanh as tanh1, log as log1
    def sqrt2(a):
        return sqrt1(a)
    def log2(a):
        return log1(a)
    model = copy.deepcopy(mod)
    model[0] = replaceFunc(model[0],exp,exp1)
    model[0] = replaceFunc(model[0],tan,tan1)
    model[0] = replaceFunc(model[0],sqrt,sqrt2)
    model[0] = replaceFunc(model[0],inv,inv1)
    model[0] = replaceFunc(model[0],sin,sin1)
    model[0] = replaceFunc(model[0],cos,cos1)
    model[0] = replaceFunc(model[0],arccos,acos)
    model[0] = replaceFunc(model[0],arcsin,asin)
    model[0] = replaceFunc(model[0],arctan,atan)
    model[0] = replaceFunc(model[0],tanh,tanh1)
    model[0] = replaceFunc(model[0],log,log2)
    response=evModHelper(model[1],model[0],[],np.array(inputData))[2][0]
    return response

def printGPModel2(mod,inputData=symbols(["x"+str(i) for i in range(100)])): #Evaluates a model numerically
    def inv1(a):
        return a**(-1)
    from sympy import tan as tan1, exp as exp1, sqrt as sqrt1, sin as sin1, cos as cos1, acos, asin, atan, tanh as tanh1, log as log1
    def sqrt2(a):
        return sqrt1(a)
    def log2(a):
        return log1(a)
    model = copy.deepcopy(mod)
    model[0] = replaceFunc(model[0],exp,exp1)
    model[0] = replaceFunc(model[0],tan,tan1)
    model[0] = replaceFunc(model[0],sqrt,sqrt2)
    model[0] = replaceFunc(model[0],inv,inv1)
    model[0] = replaceFunc(model[0],sin,sin1)
    model[0] = replaceFunc(model[0],cos,cos1)
    model[0] = replaceFunc(model[0],arccos,acos)
    model[0] = replaceFunc(model[0],arcsin,asin)
    model[0] = replaceFunc(model[0],arctan,atan)
    model[0] = replaceFunc(model[0],tanh,tanh1)
    model[0] = replaceFunc(model[0],log,log2)
    response=evModHelper2(model[1],model[0],[],np.array(inputData))[2][0]
    return response

input=np.array([[np.random.normal(1,5) for i in range(100)],[np.random.normal(1,5) for i in range(100)],[np.random.normal(1,5) for i in range(100)],[np.random.normal(1,5) for i in range(100)]])
response=np.array([np.sqrt((input[1][i]-input[2][i])**2+(input[3][i]-input[2][i])**2) for i in range(len(input[0]))])
st=time.time()
models=sgp.evolve(input,response,popSize=2000,generations=40,ops=sgp.allOps())
print(time.time()-st)
st=time.time()
models2=evolve(input,response,popSize=2000,generations=40,ops=allOps())
print(time.time()-st)
mod=models[0]
input2=np.array([[np.random.normal(1,5) for i in range(20000)],[np.random.normal(1,5) for i in range(20000)],[np.random.normal(1,5) for i in range(20000)],[np.random.normal(1,5) for i in range(20000)]])
#print(mod)
print(sgp.printGPModel(mod),"\n")
print(printGPModel(mod),"\n")
print("New\n")
print(printGPModel2(mod),"\n")
#quit()
st=time.time()
for i in range(100):
    [sgp.evaluateGPModel(models[i],input2) for j in range(2000)]
print(time.time()-st)

st=time.time()
for i in range(100):
    [evaluateGPModel(models[i],input2) for j in range(2000)]
print(time.time()-st)

st=time.time()
for i in range(100):
    [evaluateGPModel2(models[i],input2) for j in range(2000)]
print(time.time()-st)

out1=sgp.evaluateGPModel(mod,input2)
out2=evaluateGPModel(mod,input2)
out3=evaluateGPModel2(mod,input2)
out1=np.array(out1)
out2=np.array(out2)
print(np.array_equal(out1,out2))
print(np.array_equal(out1,out3))