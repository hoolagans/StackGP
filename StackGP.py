#!/usr/bin/env python
# coding: utf-8

import random
import matplotlib.pyplot as plt 
import numpy as np
import math
import copy
import sys
from scipy.stats import pearsonr # for computing correlation
from functools import reduce, cache #for flattening lists and caching
from operator import concat  #for flattening lists
from scipy.stats import trim_mean # for ensemble evaluation
from scipy.stats import differential_entropy
import warnings
import time
import dill
import os
from sklearn.cluster import KMeans #for clustering in ensemble definition
from scipy.optimize import minimize, differential_evolution #for uncertainty maximization
from sympy import symbols, simplify, expand
import sympy as sym
try:
    from IPython.display import display, clear_output
except:
    pass

import signal #for timing out functions
from contextlib import contextmanager #for timing out functions

warnings.filterwarnings('ignore', '.*invalid value.*' )
warnings.filterwarnings('ignore', '.*overflow.*' )
warnings.filterwarnings('ignore', '.*divide by.*' )
warnings.filterwarnings('ignore', '.*is constant.*' )
warnings.filterwarnings('ignore', '.*nearly constant.*' )
warnings.filterwarnings('ignore', '.*Polyfit may be.*' )
warnings.filterwarnings('ignore', '.*Number of.*')
def protectDiv(a,b):
    
    if (type(b)==int or type(b)==float or type(b)==np.float64) and b==0:
        return a/math.nan
    if (type(b)==np.ndarray) and (0 in b):
        return a/np.where(b==0,math.nan,b)
    return a/b
def add(a,b):
    return a+b
def sub(a,b):
    return a-b
def mult(a,b):
    return a*b
def exp(a):
    return np.exp(a)
# def sine(a,b):
#     return np.sin(a)
def power(a,b):
    if (type(a)==int or type(a)==float or type(a)==np.float64) and a==0:
        return a/math.nan
    if (type(a)==np.ndarray) and (0 in a):
        return a/np.where(a==0,math.nan,a)
    return a**b
def sqrt(a):
    return np.sqrt(a)
def sqrd(a):
    return a**2
def inv(a):
    return np.array(a).astype(float)**(-1)
def neg(a):
    return -a
def sin(a):
    return np.sin(a)
def cos(a):
    return np.cos(a)
def tan(a):
    return np.tan(a)
def arccos(a):
    return np.arccos(a)
def arcsin(a):
    return np.arcsin(a)
def arctan(a):
    return np.arctan(a)
def tanh(a):
    return np.tanh(a)
def log(a):
    return np.log(a)
def log10(a):
    return np.log10(a)
def log2(a):
    return np.log2(a)
def abs1(a):
    return np.abs(a)

def and1(a,b):
    return np.logical_and(a,b)
def or1(a,b):
    return np.logical_or(a,b)
def xor1(a,b):
    return np.logical_xor(a,b)
def nand1(a,b):
    return np.logical_not(np.logical_and(a,b))
def nor1(a,b):
    return np.logical_not(np.logical_or(a,b))
def xnor1(a,b):
    return np.logical_not(np.logical_xor(a,b))
def not1(a):
    return np.logical_not(a)


def defaultOps():
    return [protectDiv,add,sub,mult,exp,sqrd,sqrt,inv,neg,"pop","pop","pop","pop","pop","pop"]
def allOps():
    return [protectDiv,add,sub,mult,exp,sqrd,sqrt,inv,neg,cos,sin,tan,arccos,arcsin,arctan,tanh,log,"pop","pop","pop","pop","pop","pop","pop","pop","pop","pop"]
def booleanOps():
    return [and1,or1,xor1,nand1,nor1,xnor1,not1,"pop","pop","pop","pop","pop","pop","pop"]
def randomInt(a=-3,b=3):
    return random.randint(a,b)
def defaultConst():
    return [np.pi, np.e, randomInt,ranReal ]
def booleanConst():
    return [1,0]
def ranReal(a=20,b=-10):
    return random.random()*a-b


############################
#Data Subsampling Methods
############################
def randomSubsample(x,y, *args, **kwargs):
    n=max(int(np.ceil(len(y)**(3/5))),3)
    idx=np.random.choice(range(x.shape[1]),n,replace=False)
    return np.array([i[idx] for i in x]),y[idx]

def generationProportionalSample(x,y,generation=100,generations=100):
    n=max(int(np.ceil(len(y)*(generation/generations)**(3/5))),3)
    idx=np.random.choice(range(x.shape[1]),n,replace=False)
    return np.array([i[idx] for i in x]),y[idx]

def ordinalSample(x,y,generation=100,generations=100):
    n=max(int(len(y)*generation/generations),3)
    sortedIdx=np.argsort(y)
    step=len(y)/(n-1)
    idx=[sortedIdx[max(int(i*step)-1,0)] for i in range(n)]
    return np.array([i[idx] for i in x]),y[idx]

def orderedSample(x,y,generation=100,generations=100):
    n=max(int(len(y)*generation/generations),3)
    idx=[i for i in range(n)]
    return np.array([i[idx] for i in x]),y[idx]

def ordinalBalancedSample(x,y,generation=100,generations=100):
    n=max(int(len(y)*generation/generations),3)
    numBins=int(max(np.ceil(np.sqrt(n)),3))
    bins=np.linspace(min(y),max(y),numBins+1)
    binIdx=np.digitize(y,bins)-1
    samplesPerBin=max(int(n/numBins),1)
    idx=[]
    for i in range(numBins):
        binMembers=[j for j in range(len(y)) if binIdx[j]==i]
        if len(binMembers)>0:
            chosen=np.random.choice(binMembers,min(samplesPerBin,len(binMembers)),replace=False)
            idx=idx+chosen.tolist()
    return np.array([i[idx] for i in x]),y[idx]

def balancedSample(x,y, *args, **kwargs):
    n=int(np.ceil(len(y)**(3/5)))
    numBins=max(round(n**(2/5)),3)
    bins=np.linspace(min(y),max(y),numBins+1)
    binIdx=np.digitize(y,bins)-1
    samplesPerBin=max(int(n/numBins),1)
    idx=[]
    for i in range(numBins):
        binMembers=[j for j in range(len(y)) if binIdx[j]==i]
        if len(binMembers)>0:
            chosen=np.random.choice(binMembers,min(samplesPerBin,len(binMembers)),replace=False)
            idx=idx+chosen.tolist()
    return np.array([i[idx] for i in x]),y[idx]

import inspect
@cache
def getArity(func): #Returns the arity of a function: used for model evaluations
    if func=="pop":
        return 1
    return len(inspect.signature(func).parameters)

getArity.__doc__ = "getArity(func) takes a function and returns the function arity"
def modelArity(model): #Returns the total arity of a model
    return 1+sum([getArity(i)-1 for i in model[0]])

modelArity.__doc__ = "modelArity(model) returns the total arity of a model"
def listArity(data): #Returns arity of evaluating a list of operators
    if len(data)==0:
        return 0
    return 1+sum([getArity(i)-1 for i in data])
listArity.__doc__ = "listArity(list) returns the arity of evaluating a list of operators"
def buildEmptyModel(): # Generates an empty model
    return [[],[],[]]
buildEmptyModel.__doc__ = "buildEmptyModel() takes no inputs and generates an empty GP model"
def variableSelect(num): #Function that creates a function to select a specific variable
    return lambda variables: variables[num]
variableSelect.__doc__ = "variableSelect(n) is a function that creates a function to select the nth variable"
def modelToListForm(model):
    model[0]=model[0].tolist()
def modelRestoreForm(model):
    model[0]=np.array(model[0],dtype=object)

def generateRandomModel(variables,ops,const,maxLength):  #Generates a random GP model
    prog = buildEmptyModel()                             #Generate an empty model with correct structure
    varChoices=[variableSelect(i) for i in range(variables)]+const                           #All variable and constants choices
    prog[0]=np.array(np.random.choice(ops,random.randint(1,maxLength)),dtype=object) #Choose random operators
    countVars=modelArity(prog)    #Count how many variables/constants are needed
    prog[1]=np.random.choice(varChoices,countVars)       #Choose random variables/constants
    # if all vars are constants then replace one random term
    if all(t in const for t in prog[1]):
        replace_idx = random.randrange(countVars)
        prog[1][replace_idx] = random.choice(varChoices[:variables]) #Replace with a variable
    prog[1]=[i() if (callable(i) and i.__name__!='<lambda>' )else i for i in prog[1]] #If function then evaluate
    return prog
generateRandomModel.__doc__ = "generateRandomModel() takes as input the variables, operators, constants, and max program length and returns a random program"
def initializeGPModels(variables,ops=defaultOps(),const=defaultConst(),numberOfModels=100,maxLength=10): # generate random linear program
    prog=[[],[],[]]
    # prog stores [Operators, VarConst, QualityMetrics]
    
    models=[generateRandomModel(variables,ops,const,maxLength) for i in range(numberOfModels)] #Generate models
    
    return models
initializeGPModels.__doc__ = "initializeGPModels(countOfVariables, operators, constants, numberOfModels=100, maxLength=10) returns a set of randomly generated models"

def reverseList(data): #Returns a list reversed
    return [i for i in reversed(data)]
reverseList.__doc__ = "reverseList(data) returns the data list reversed"
def varReplace(data,variables): #Replaces variable references with data during model evaluation
    return [i(variables) if callable(i) else i for i in data]
varReplace.__doc__ = "varReplace(data,variables) replaces references to variables in data with actual values"
def inputLen(data): #Returns the number of data records in a data set
    el1=data[0]
    if type(el1)==list or type(el1)==np.ndarray:
        return len(el1)
    else:
        return 1
inputLen.__doc__ = "inputLen(data) determines the number of data records in a data set"
def varCount(data): #Returns the number of variables in a data set
    return len(data)
varCount.__doc__ = "varCount(data) determines the number of variables in a data set"
def evaluateGPModel(model,inputData): #Evaluates a model numerically
    response=evModHelper(model[1],model[0],[],np.array(inputData).astype(float))[2][0]
    if not type(response)==np.ndarray and inputLen(inputData)>1:
        response=np.array([response for i in range(inputLen(inputData))])
    return response
evaluateGPModel.__doc__ = "evaluateGPModel(model,data) numerically evaluates a model using the data stored in inputData"
def evModHelper(varStack,opStack,tempStack,data): #Recursive helper function for evaluateGPModel
    stack1=varStack
    stack2=opStack
    stack3=tempStack
    
    if len(stack2)==0:
        return [stack3,stack2,stack1]
    op=stack2[0]
    stack2=stack2[1:]
    
    if callable(op):
        
        patt=getArity(op)
        while patt>len(stack3):
            stack3=[stack1[0]]+stack3
            stack1=stack1[1:]
        try:
            temp=op(*varReplace(reverseList(stack3[:patt]),data))
        except TypeError:
            print("stack3: ", stack3, " patt: ", patt, " data: ", data)
            temp=np.nan
        except OverflowError:
            temp=np.nan
        stack3=stack3[patt:]
        stack3=[temp]+stack3
        
    else:
        if len(stack1)>0:
            stack3=varReplace([stack1[0]],data)+stack3
            stack1=stack1[1:]
    if len(stack2)>0:
        stack1,stack2,stack3=evModHelper(stack1,stack2,stack3,data)
        
    return [stack1,stack2,stack3]
evModHelper.__doc__ = "evModHelper(varStack,opStack,tempStack,data) is a helper function for evaluateGPModel"
def rmse(model, inputData, response):
    predictions = evaluateGPModel(model, inputData)
    if not all(np.isfinite(predictions)) or any(np.iscomplex(predictions)):
        return np.nan
    return np.sqrt(np.mean((predictions - response) ** 2))
rmse.__doc__ = "rmse(model, input, response) is a fitness objective that evaluates the root mean squared error"
def binaryError(model, input, response):
    prediction=evaluateGPModel(model,input)
    error=np.mean(np.abs(prediction-response))
    if np.isnan(error) or np.isinf(error) or error > 1 or error < 0:
        return 0.5
    return min(error,1 - error)
def fitness(prog,data,response): # Fitness function using correlation
    predicted=evaluateGPModel(prog,np.array(data))
    if type(predicted)!=list and type(predicted)!=np.ndarray:
        predicted=np.array([predicted for i in range(inputLen(data))])
    try:    
        if np.isnan(predicted).any() or np.isinf(predicted).any():
            return np.nan
    except TypeError:
        #print(predicted)
        return np.nan
    except OverflowError:
        return np.nan
    if (not all(np.isfinite(np.array(predicted,dtype=np.float32)))) or np.all(predicted==predicted[0]):
        return np.nan
    try:
        fit=1-pearsonr(predicted,np.array(response))[0]**2  # 1-R^2
    except ValueError:
        return 1
    if math.isnan(fit):
        return 1 # If nan return 1 as fitness
    return fit   # Else return actual fitness 1-R^2
fitness.__doc__ = "fitness(program,data,response) returns the 1-R^2 value of a model"
def stackGPModelComplexity(model,*args):
    return len(model[0])+len(model[1])-model[0].tolist().count("pop")
stackGPModelComplexity.__doc__ = "stackGPModelComplexity(model) returns the complexity of the model"

###################### Timeout function for model complexity ######################
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
####################################################################################

# Compute Hess
def ComputeSymbolicHess(model,vars):
    printedModel=sym.simplify(printGPModel(model))
    if type(printedModel)==float:
        return sym.matrices.dense.MutableDenseMatrix(np.zeros((vars,vars)))
    hess=sym.hessian(printedModel, [symbols('x'+str(i)) for i in range(vars)])
    return hess

def EvaluateHess(hess,vars,values):
    numHess=hess.subs({symbols('x'+str(j)):values[j] for j in range(vars)})
    hessN = np.array(numHess).astype(float)
    rankN=np.linalg.matrix_rank(hessN,tol=0.0001*0.0001*10)
    return rankN

def Approx2Deriv(model,values,diff1,diff2,positions): #maybe diff should be relative to the variation of each feature
    term1=[values[i]+diff1 if i == positions[0] else values[i] for i in range(len(values))]
    term1=[term1[i]+diff2 if i == positions[1] else term1[i] for i in range(len(term1))]
    term2=[values[i]-diff1 if i == positions[0] else values[i] for i in range(len(values))]
    term2=[term2[i]+diff2 if i == positions[1] else term2[i] for i in range(len(term2))]
    term3=[values[i]+diff1 if i == positions[0] else values[i] for i in range(len(values))]
    term3=[term3[i]-diff2 if i == positions[1] else term3[i] for i in range(len(term3))]
    term4=[values[i]-diff1 if i == positions[0] else values[i] for i in range(len(values))]
    term4=[term4[i]-diff2 if i == positions[1] else term4[i] for i in range(len(term4))]
    return ((evaluateGPModel(model,term1)-evaluateGPModel(model,term2))/((2*diff1))
            -(evaluateGPModel(model,term3)-evaluateGPModel(model,term4))/((2*diff1)))/(2*diff2)

def ApproxHessRank(model,vars,values,diff1=0.001,diff2=0.001):
    hess=[[Approx2Deriv(model,values,diff1,diff2,[i,j]) for i in range(vars)] for j in range(vars)]
    hessN = np.array(hess).astype(float)
    rankN=np.linalg.matrix_rank(hessN,tol=0.0001*0.0001*10)
    return rankN

#def HessRank(model,vars,values):
#    try: 
#        with time_limit(.01):
#            hess=ComputeSymbolicHess(model,vars)
#            hess = EvaluateHess(hess,vars,values)
#            #print(hess)
#            return hess
#    except TimeoutException as e:
#        hess=ApproxHessRank(model,vars,values)
        #print(hess)
#        return hess

def HessRank(model,vars,values):
    hess=ApproxHessRank(model,vars,values)
    return hess





# Counts basis terms in a model
def count_basis_terms(equation, expand=False):
    try:
        with time_limit(2):


            if expand:
                # Simplify the equation to standardize the expression
                simplified_eq = simplify(equation)
                # Expand the expression to identify additive terms clearly
                expanded_eq = expand(simplified_eq)
            
                # Separate the terms of the expression
                terms = expanded_eq.as_ordered_terms()
            else:
                terms = equation.as_ordered_terms()
            #print(terms)
            
    except TimeoutException as e:
        return 1000
    return len(terms)

# Determines the number of basis functions in a model by counting +s and -s
def basisFunctionComplexity(model,vars, values,*args):
    try: # values should be max, min, and median with respect to response variable
        return HessRank(model,vars,values)#count_basis_terms(printGPModel(model))
    except:
        return 1000

# Creates a lambda function to be used as a complexity metric when given a target dimensionality and deviation
def basisFunctionComplexityDiff(target, deviation, vars, low, mid, high):
    return lambda model,*args: max(np.mean([abs(basisFunctionComplexity(model,vars,low)-target),abs(basisFunctionComplexity(model,vars,mid)-target) ,abs(basisFunctionComplexity(model,vars,high)-target)] ),(deviation))-deviation


def setModelQuality(model,inputData,response,modelEvaluationMetrics=[fitness,stackGPModelComplexity]):
    model[2]=[i(model,inputData,response) for i in modelEvaluationMetrics]

    
setModelQuality.__doc__ = "setModelQuality(model, inputdata, response, metrics=[r2,size]) is an inplace operator that sets a models quality"
def stackPass(model,pt):
    i=0
    t=0
    p=0
    s=model[0]
    if i <pt:
        t+=1
    while i<pt:
        if s[i]=="pop":
            t+=1
            p+=1
        else:
            p+=max(0,getArity(s[i])-t)
            t=max(1,t-getArity(s[i])+1)
        i+=1
    stack1=model[1][p:]
    stack2=reverseList(model[1][:p])[:t+1]
    return [stack1,stack2]
def stackGrab(stack1, stack2, num):
    tStack1=copy.deepcopy(stack1)
    tStack2=copy.deepcopy(stack2)
    newStack=[]
    if len(stack2)<num:
        newStack=stack2+stack1[:(num-len(stack2))]
        tStack1=tStack1[num-len(tStack2):]
        tStack2=[]
    else:
        newStack=stack2[:num]
        tStack2=tStack2[num:]
    return [newStack,tStack1,tStack2]
def fragmentVariables(model,pts):
    stack1,stack2=stackPass(model,pts[0])
    opStack=model[0]
    newStack=[]
    i=pts[0]
    while i<=pts[1]:
        if opStack[i]=="pop" and len(stack1)>0:
            stack2=[stack1[0]]+stack2
            stack1=stack1[1:]
        else:
            if len(newStack)==0 and pts[0]==0:
                tStack,stack1,stack2=stackGrab(stack1,stack2,getArity(opStack[i]))
            else:
                tStack,stack1,stack2=stackGrab(stack1,stack2,getArity(opStack[i])-1)
            newStack=newStack+tStack
        i+=1
    return newStack
                                            
def recombination2pt(model1,model2): #2 point recombination
    pts1=np.sort(random.sample(range(0,len(model1[0])+1),2))
    pts2=np.sort(random.sample(range(0,len(model2[0])+1),2))
    #pts1=[4,5]
    #pts2=[2,4]
    #pts1=[0,3]
    #pts2=[1,3]
    #print(pts1,pts2)
    child1=buildEmptyModel()
    child2=buildEmptyModel()
    
    parent1=copy.deepcopy(model1)
    parent2=copy.deepcopy(model2)
    parent1[0]=np.array(parent1[0],dtype=object).tolist()
    parent2[0]=np.array(parent2[0],dtype=object).tolist()
    
    child1[0]=np.array(parent1[0][0:pts1[0]]+parent2[0][pts2[0]:pts2[1]]+parent1[0][pts1[1]:],dtype=object)
    child2[0]=np.array(parent2[0][0:pts2[0]]+parent1[0][pts1[0]:pts1[1]]+parent2[0][pts2[1]:],dtype=object)
        
    varPts1=[listArity(parent1[0][:(pts1[0])])+0,listArity(parent2[0][:(pts2[0])])+0,listArity(parent2[0][pts2[0]:pts2[1]]),listArity(parent1[0][pts1[0]:pts1[1]])]
    if pts1[0]==0:
        varPts1[0]+=1
    if pts2[0]==0:
        varPts1[1]+=1
    child1[1]=parent1[1][:varPts1[0]]+parent2[1][varPts1[1]:(varPts1[1]+varPts1[2]-1)]+parent1[1][(varPts1[0]+varPts1[3]-1):]
    
    varPts2=[listArity(parent2[0][:(pts2[0])])+0,listArity(parent1[0][:(pts1[0])])+0,listArity(parent1[0][pts1[0]:pts1[1]]),listArity(parent2[0][pts2[0]:pts2[1]])]
    if pts1[0]==0:
        varPts2[1]+=1
    if pts2[0]==0:
        varPts2[0]+=1
    child2[1]=parent2[1][:varPts2[0]]+parent1[1][varPts2[1]:(varPts2[1]+varPts2[2]-1)]+parent2[1][(varPts2[0]+varPts2[3]-1):]
    #print(varPts1,varPts2)
    
    return [child1,child2]
recombination2pt.__doc__ = "recombination2pt(model1,model2) does 2 point crossover and returns two children models"

def get_numeric_indices(l): #Returns indices of list that are numeric
    return [i for i in range(len(l)) if type(l[i]) in [int,float]]


def mutate(model,variables,ops=defaultOps(),const=defaultConst(),maxLength=10):
    newModel=copy.deepcopy(model)
    newModel[0]=np.array(newModel[0],dtype=object).tolist()
    mutationType=random.randint(0,7) 
    varChoices=[variableSelect(i) for i in range(variables)]+const
    opChoice=0
    varChoice=0
    
    tmp=0
    
    if mutationType==0: #single operator mutation
        opChoice=random.randint(0,len(newModel[0])-1)
        if len(newModel[0])>0:
            newModel[0][opChoice]=np.random.choice([i for i in ops] )
               
    elif mutationType==1: #single variable mutation
        varChoice=np.random.choice(varChoices)
        if callable(varChoice) and varChoice.__name__!='<lambda>':
            varChoice=varChoice()
        newModel[1][random.randint(0,len(newModel[1])-1)]=varChoice
    
    elif mutationType==2: #insertion mutation to top of stack
        opChoice=np.random.choice(ops)
        newModel[0]=[opChoice]+newModel[0]
        while modelArity(newModel)>len(newModel[1]):
            varChoice=np.random.choice(varChoices)
            if callable(varChoice) and varChoice.__name__!='<lambda>':
                varChoice=varChoice()
            newModel[1]=[varChoice]+newModel[1]
        
    elif mutationType==3: #deletion mutation from top of stack
        if len(newModel[0])>1:
            opChoice=random.randint(1,len(newModel[0])-1)
            newModel[0]=newModel[0][-opChoice:]
            newModel[1]=newModel[1][-listArity(newModel[0]):]
            
    elif mutationType==4: #insertion mutation to bottom of stack
        opChoice=np.random.choice([i for i in ops])
        newModel[0].append(opChoice)
        
    elif mutationType==5: #mutation via crossover with random model
        newModel=recombination2pt(newModel,generateRandomModel(variables,ops,const,maxLength))[0]
            
    elif mutationType==6: #single operator insertion mutation
        singleOps=[op for op in ops if getArity(op)==1 and op!='pop']
        singleOps.append('pop')
        pos=random.randint(0,len(newModel[0])-1)
        newModel[0].insert(pos,np.random.choice(singleOps))

    elif mutationType==7: #nudge numeric constant
        pos=get_numeric_indices(newModel[1])
        if(len(pos)>0): #If there are numeric constants
            pos=random.choice(pos)
            newModel[1][pos]=newModel[1][pos]+np.random.normal(-1,1) 
            
    if modelArity(newModel)<len(newModel[1]):
        newModel[1]=newModel[1][:modelArity(newModel)]
    elif modelArity(newModel)>len(newModel[1]):
        newModel[1]=newModel[1]+[np.random.choice(varChoices) for i in range(modelArity(newModel)-len(newModel[1]))]
    newModel[1]=[varChoice() if callable(varChoice) and varChoice.__name__!='<lambda>' else varChoice for varChoice in newModel[1]]         
    newModel[0]=np.array(newModel[0],dtype=object)
    return newModel
    
mutate.__doc__ = "mutate(model,variableCount,ops,constants,maxLength) mutates a model"
def paretoFront(fitValues): #Returns Boolean list of Pareto front elements
    onFront = np.ones(fitValues.shape[0], dtype = bool)
    for i, j in enumerate(fitValues):
        if onFront[i]:
            onFront[onFront] = np.any(fitValues[onFront]<j, axis=1)  
            onFront[i] = True  
    return onFront
def paretoTournament(pop): # selects the Pareto front of a model set
    fitnessValues=np.array([mod[2] for mod in pop])
    return (np.array(pop,dtype=object)[paretoFront(fitnessValues)]).tolist()
def tournamentModelSelection(models, popSize=100,tourneySize=5):
    selectedModels=[]
    selectionSize=popSize
    while len(selectedModels)<popSize:
        tournament=random.sample(models,tourneySize)
        winners=paretoTournament(tournament)
        selectedModels=selectedModels+winners
    
    return selectedModels
paretoTournament.__doc__ = "paretoTournament(models, inputData, responseData) returns the Pareto front of a model set"
def modelSameQ(model1,model2): #Checks if two models are the same
    return len(model1[0])==len(model2[0]) and len(model1[1]) == len(model2[1]) and all(model1[0]==model2[0]) and model1[1]==model2[1]
modelSameQ.__doc__ = "modelSameQ(model1,model2) checks if model1 and model2 are the same and returns True if so, else False"
def deleteDuplicateModels(models): #Removes any models that are the same, does not consider simplified form
    uniqueMods = [models[0]]
     
    for mod in models:
        test=False
        for checkMod in uniqueMods:
            if modelSameQ(mod,checkMod):
                test=True
        if not test:
            uniqueMods.append(mod)
    
    return uniqueMods
deleteDuplicateModels.__doc__ = "deleteDuplicateModels(models) deletes models that have the same form without simplifying"

def deleteDuplicateModelsPhenotype(models): #Removes any models that are the same regarding phenotype, does not consider simplified form
    uniqueMods = [printGPModel(models[0])]
    remainingMods=[printGPModel(mod) for mod in models[1:]]
    uniquePos = [0]
    currPos=1
    for mod in remainingMods:
        test=False
        for checkMod in uniqueMods:
            if mod==checkMod:
                test=True
        if not test:
            uniqueMods.append(mod)
            uniquePos.append(currPos)
        currPos+=1
    
    return [models[i] for i in uniquePos]

def removeIndeterminateModels(models): #Removes models from the population that evaluate to nonreal values
    return [i for i in models if (not any(np.isnan(i[2]))) and all(np.isfinite(np.isnan(i[2])))]
removeIndeterminateModels.__doc__ = "removeIndeterminateModels(models) removes models that have a fitness that results from inf or nan values"
def sortModels(models):
    return sorted(models, key=lambda m:m[2])
sortModels.__doc__ = "sortModels(models) sorts a model population by the models' accuracies"
def selectModels(models, selectionSize=0.5, thresholds=None):
    tMods=copy.deepcopy(models)
    [modelToListForm(mod) for mod in tMods]
    if thresholds is not None:
        tMods=[mod for mod in tMods if all([mod[2][i]<=thresholds[i] for i in range(len(thresholds))])]
    paretoModels=[]
    if selectionSize<=1:
        selection=selectionSize*len(models)
    else:
        selection=selectionSize
    
    while len(paretoModels)<selection and len(tMods)>0:
        front=paretoTournament(tMods)
        paretoModels=paretoModels+front
        for i in front:
            tMods.remove(i)
    [modelRestoreForm(mod) for mod in paretoModels]
    return paretoModels
selectModels.__doc__ = "selectModels(models, selectionSize=0.5) iteratively selects the Pareto front of a model population until n or n*popSize models are selected"
def stackVarUsage(opStack): #Counts how many variables are used by the operator stack
    pos=getArity(opStack[0])
    for j in range(1,len(opStack)):
        pos+=getArity(opStack[j])-1
        if opStack[j]=='pop':
            pos+=1
    return pos
stackVarUsage.__doc__ = "stackVarUsage(opStack) is a helper function that determines how many variables/constants are needed by the operator stack"
def trimModel(mod): #Removes extra pop operators that do nothing
    model=copy.deepcopy(mod)
    i=0
    varStack=len(mod[1])
    tempStack=0
    varStack-=getArity(model[0][i])
    tempStack+=1
    i+=1
    while varStack>0:
        if model[0][i]=='pop':
            varStack-=1
            tempStack+=1
        else:
            
            take=getArity(model[0][i])-tempStack
            if take>0:
                varStack-=take
                tempStack=1
            else:
                tempStack-=getArity(model[0][i])-1
        i+=1
    model[0]=np.array(model[0][:i].tolist()+[j for j in model[0][i:] if not j=='pop'],dtype=object)
    return model
trimModel.__doc__ = "trimModel(model) trims extra pop operators off the operator stack so that further modifications such as a model alignment aren't altered by those pop operators"
def alignGPModel(model, data, response): #Aligns a model
    prediction=evaluateGPModel(model,data)
    if (not all(np.isfinite(np.array(prediction)))) or np.all(prediction==prediction[0]):
        return model
    if np.isnan(np.array(prediction)).any() or np.isnan(np.array(response)).any() or not np.isfinite(np.array(prediction,dtype=np.float32)).all():
        return model
    # Variance guards
    if np.std(prediction) < 1e-12:
        return model
    if np.ptp(prediction) < 1e-12:
        return model
    try:
        align=np.polyfit(prediction,response,1,rcond=1e-16)#np.round(np.polyfit(prediction,response,1,rcond=1e-16),decimals=14)
    except np.linalg.LinAlgError:
        #print("Alignment failed for: ", model, " with prediction: ", prediction, "and reference data: ", response)
        return model
    newModel=trimModel(model)
    newModel[0]=np.array(newModel[0].tolist()+[mult,add],dtype=object)
    newModel[1]=newModel[1]+align.tolist()
    #setModelQuality(newModel,data,response)
    return newModel
alignGPModel.__doc__ = "alignGPModel(model, input, response) aligns a model such that response-a*f(x)+b are minimized over a and b"

def replaceEvaluate(model, newVec, inputData, responseData):
    modelCopy = copy.deepcopy(model)
    indices = get_numeric_indices(model[1])
    for i in range(len(indices)):
        modelCopy[1][indices[i]] = newVec[i]
    value = rmse(modelCopy, inputData, responseData)
    # if nan return infinity to avoid it being selected as the best solution
    if np.isnan(value):
        return np.inf
    return value

def optimizeModel(model, inputData, responseData, bounds=None, **kwargs):
    indices = get_numeric_indices(model[1])
    startingVals = [model[1][i] for i in indices]
    fnc = lambda x: replaceEvaluate(model, x, inputData, responseData)
    if bounds is None:
        bounds = [np.sort((-10*val, 10*val)) for val in startingVals]
    out = differential_evolution(fnc, bounds=bounds, x0=startingVals, **kwargs)
    newModel = copy.deepcopy(model)
    for i in range(len(indices)):
        newModel[1][indices[i]] = out.x.tolist()[i]
    setModelQuality(newModel, inputData, responseData)
    return newModel


def evolve(inputData, responseData, generations=100, ops=defaultOps(), const=defaultConst(), variableNames=[], mutationRate=79, crossoverRate=11, spawnRate=10, extinction=False,extinctionRate=10,elitismRate=10,popSize=300,maxComplexity=100,align=True,initialPop=[],timeLimit=300,capTime=False,tourneySize=5,tracking=False,returnTracking=False,liveTracking=False,liveTrackingInterval=1,modelEvaluationMetrics=[fitness,stackGPModelComplexity],dataSubsample=False,samplingMethod=randomSubsample,alternateObjectives=[],alternateObjFrequency=10,allowEarlyTermination=False,earlyTerminationThreshold=0):
    
    alternatingFlag = False
    if callable(modelEvaluationMetrics):
        metrics=[modelEvaluationMetrics]
        allMetrics=[modelEvaluationMetrics]+alternateObjectives
    elif isinstance(modelEvaluationMetrics, list) and callable(modelEvaluationMetrics[0]):
        metrics=modelEvaluationMetrics
        allMetrics=modelEvaluationMetrics+alternateObjectives
    elif isinstance(modelEvaluationMetrics, list) and isinstance(modelEvaluationMetrics[0], list):
        metrics=modelEvaluationMetrics[0]
        allMetrics=[item for sublist in modelEvaluationMetrics for item in sublist]+alternateObjectives
        alternatingFlag = True
    else:
        raise ValueError("modelEvaluationMetrics must be a function, list of functions, or a list of lists of functions")

    fullInput,fullResponse=copy.deepcopy(inputData),copy.deepcopy(responseData)
    inData=copy.deepcopy(fullInput)
    resData=copy.deepcopy(fullResponse)
    variableCount=varCount(inData)
    models=initializeGPModels(variableCount,ops,const,popSize)
    models=models+initialPop
    startTime=time.perf_counter()
    bestFits=[]
    if liveTracking:
        fig, ax = plt.subplots(figsize=(20,10))
        ckTime=time.perf_counter()
    for i in range(generations):
        if capTime and time.perf_counter()-startTime>timeLimit:
            break
        if len(alternateObjectives)>0 and (i+1)%alternateObjFrequency==0:
            metrics=modelEvaluationMetrics[:1]+alternateObjectives
        else:
            if alternatingFlag:
                metrics=modelEvaluationMetrics[i%len(modelEvaluationMetrics)]
            else:
                metrics=modelEvaluationMetrics
        if dataSubsample:
            inData,resData=samplingMethod(fullInput,fullResponse,generations=generations,generation=i)
        for mods in models:
            setModelQuality(mods,inData,resData,modelEvaluationMetrics=metrics)
        models=removeIndeterminateModels(models)
        if allowEarlyTermination and min([mods[2][0] for mods in models])<=earlyTerminationThreshold:
            print("Early termination at generation ", i)
            break
        if tracking or liveTracking or returnTracking:
            bestFits.append(min([mods[2][0] for mods in paretoTournament(models)]))
        if liveTracking and time.perf_counter()-ckTime>liveTrackingInterval:
            ax.clear()
            ax.plot(bestFits)
            ax.set_title(f"Best Model: {bestFits[-1]:.2f} at Generation {(i+1)}")
            ax.set_xlabel("Generations")
            ax.set_ylabel("Fitness")
            clear_output(wait=True) 
            display(fig)    
            #plt.show()        
            plt.close(fig)
            ckTime=time.perf_counter()        

        #paretoModels=paretoTournament(models)
        paretoModels=selectModels(models,elitismRate/100*popSize if elitismRate/100*popSize<len(models) else len(models))
        if extinction and i%extinctionRate==0 and i>0:
            models=initializeGPModels(variableCount,ops,const,popSize)
            for mods in models:
                setModelQuality(mods,inData,resData,modelEvaluationMetrics=metrics)
        
        models=tournamentModelSelection(models,popSize,tourneySize)
        
        crossoverPairs=random.sample(models,round(crossoverRate/100*popSize))
        toMutate=random.sample(models,round(mutationRate/100*popSize))
        
        childModels=paretoModels
        
        for j in range(round(len(crossoverPairs)/2)-1):
            childModels=childModels+recombination2pt(crossoverPairs[j],crossoverPairs[j+round(len(crossoverPairs)/2)])
        
        for j in toMutate:
            childModels=childModels+[mutate(j,variableCount,ops,const)]
        
        childModels=childModels+initializeGPModels(variableCount,ops,const,round(spawnRate/100*popSize))
        
        childModels=deleteDuplicateModels(childModels)
        childModels=[model for model in childModels if stackGPModelComplexity(model)<maxComplexity]
        
        #for mods in childModels:
        #    setModelQuality(mods,inData,resData,modelEvaluationMetrics=modelEvaluationMetrics)
        #childModels=removeIndeterminateModels(childModels)
        
        if len(childModels)<popSize:
            childModels=childModels+initializeGPModels(variableCount,ops,const,popSize-len(childModels))
        
        models=copy.deepcopy(childModels)
        
    
    for mods in models:
        setModelQuality(mods,fullInput,fullResponse,modelEvaluationMetrics=allMetrics)
    models=[trimModel(mod) for mod in models]
    models=deleteDuplicateModels(models)
    models=removeIndeterminateModels(models)
    models=sortModels(models)
    if align:
        models=[alignGPModel(mods,fullInput,fullResponse) for mods in models]
        for mods in models:
            setModelQuality(mods,fullInput,fullResponse,modelEvaluationMetrics=allMetrics)
    
    if tracking or returnTracking:
        bestFits.append(min([mods[2][0] for mods in paretoTournament(models)]))
        if returnTracking:
            return models, bestFits
        plt.figure()
        plt.plot(bestFits)
        plt.title("Fitness over Time")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.show()
    
    return models
    

def replaceFunc(stack,f1,f2):
    return [i if i!=f1 else f2 for i in stack]
def printGPModel(mod,inputData=symbols(["x"+str(i) for i in range(100)])): #Evaluates a model algebraically
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
    try:
        response=evModHelper(model[1],model[0],[],np.array(inputData))[2][0]
    except:
        return np.nan
    return response

def ensembleSelect(models, inputData, responseData, numberOfClusters=10): #Generates a model ensemble using input data partitions
    data=np.transpose(inputData)
    if len(data)<numberOfClusters:
        numberOfClusters=len(data)
    clusters=KMeans(n_clusters=numberOfClusters).fit_predict(data)
    if numberOfClusters>len(set(clusters)):
        numberOfClusters=len(set(clusters))
        clusters=KMeans(n_clusters=numberOfClusters).fit_predict(data)
    dataParts=[]
    partsResponse=[]
    for i in range(numberOfClusters):
        dataParts.append([])
        partsResponse.append([])
    
    for i in range(len(clusters)):
        dataParts[clusters[i]].append(data[i])
        partsResponse[clusters[i]].append(responseData[i])
        
    modelResiduals=[]
    
    for i in range(len(models)):
        modelResiduals.append([])
    for i in range(len(models)):
        for j in range(numberOfClusters):
            modelResiduals[i].append(fitness(models[i],np.transpose(dataParts[j]),partsResponse[j]))
    
    best=[]
    for i in range(numberOfClusters):
        ordering=np.argsort(modelResiduals[i])
        j=0
        while ordering[j] in best:
            j+=1
        best.append(ordering[j])
    ensemble=[models[best[i]] for i in range(numberOfClusters)]
    
    return ensemble
def uncertainty(data,trim=0.3):
    wl=None
    if len(data)<=4:
        wl=1
    h=differential_entropy(data,window_length=wl)
    if np.isfinite(h):
        return h
    else:
        return 0

def evaluateModelEnsemble(ensemble, inputData):
    responses=[evaluateGPModel(mod, inputData) for mod in ensemble]
    if type(responses[0])==np.ndarray:
        responses=np.transpose(responses)
        predictions=[np.median(res) for res in responses]
    else:
        
        predictions=[np.median(responses)]
    
    return predictions
    
def evaluateModelEnsembleUncertainty(ensemble, inputData):
    responses=[evaluateGPModel(mod, inputData) for mod in ensemble]
    if type(responses[0])==np.ndarray:
        responses=np.transpose(responses)
        uncertainties=[uncertainty(res,0) for res in responses]
    else:
        
        uncertainties=[uncertainty(responses,0)]
    return uncertainties

def relativeEnsembleUncertainty(ensemble,inputData):
    output=evaluateModelEnsembleUncertainty(ensemble,inputData)
    return np.array(output)
    
def createUncertaintyFunc(ensemble):
    return lambda x: -relativeEnsembleUncertainty(ensemble,x)
    
def maximizeUncertainty(ensemble,varCount,bounds=[]): #Used to select a new point of maximum uncertainty
    func=createUncertaintyFunc(ensemble)
    x0=[np.mean(bounds[i]) for i in range(varCount)]
    if bounds==[]:
        pt=minimize(func,x0).x
    else:
        pt=minimize(func,x0,bounds=bounds).x
    return pt
def extendData(data,newPoint):
    return np.concatenate((data.T,np.array([newPoint]))).T

def activeLearningCheckpoint(eqNum,version,i,inputData,response,testInput,testResponse,errors,models,minerr):
    path=os.path.join(str(eqNum),str(version))
    file=open(path,"wb+")
    dill.dump([i,inputData,response,testInput,testResponse,errors,models,minerr],file)
    file.close()
def activeLearningCheckpointLoad(eqNum,version,i,inputData,response,testInput,testResponse,errors,models,minerr):
    path=os.path.join(str(eqNum),str(version))
    try:
        with open(path,'rb') as f:
            i,inputData,response,testInput,testResponse,errors,models,minerr=dill.load(f)
    except FileNotFoundError:
        return i,inputData,response,testInput,testResponse,errors,models,minerr
    return i,inputData,response,testInput,testResponse,errors,models,minerr
def subSampleSpace(space):
    newSpace=copy.deepcopy(space)
    newSpace=list(newSpace)
    for i in range(len(newSpace)):
        pts=sorted([np.random.uniform(newSpace[i][0],newSpace[i][1]),np.random.uniform(newSpace[i][0],newSpace[i][1])])
        newSpace[i]=tuple(pts)
    return tuple(newSpace)

def activeLearning(func, dims, ranges,rangesP,eqNum=1,version=1,iterations=100): #func should be a lamda function of form lambda data: f(data[0],data[1],...)
    try:
        with open(os.path.join(str(eqNum),str(version))+".txt",'rb') as f:
            return -1
    except FileNotFoundError:
        pass
    inputData=[]
    testInput=[]
    found=False
    for i in range(dims):
        inputData.append(np.random.uniform(ranges[i][0],ranges[i][1],3))
        testInput.append(np.random.uniform(ranges[i][0],ranges[i][1],200))
    inputData=np.array(inputData)
    testInput=np.array(testInput)
    response=func(inputData)
    testResponse=func(testInput)
    errors=[]
    models=[]
    minerr=1
    for i in range(iterations):
        print("input: ",inputData)
        print("\n response: ",response)
        i,inputData,response,testInput,testResponse,errors,models,minerr=activeLearningCheckpointLoad(eqNum,version,i,inputData,response,testInput,testResponse,errors,models,minerr)
        if i>iterations-1:
            break
        i+=1            
        models1=evolve(inputData,response,initialPop=models,generations=1000,tracking=False,popSize=300,ops=allOps(),timeLimit=120,capTime=True,align=False,elitismRate=10)
        models2=evolve(inputData,response,initialPop=models,generations=1000,tracking=False,popSize=300,ops=allOps(),timeLimit=120,capTime=True,align=False,elitismRate=10)
        models3=evolve(inputData,response,initialPop=models,generations=1000,tracking=False,popSize=300,ops=allOps(),timeLimit=120,capTime=True,align=False,elitismRate=10)
        models4=evolve(inputData,response,initialPop=models,generations=1000,tracking=False,popSize=300,ops=allOps(),timeLimit=120,capTime=True,align=False,elitismRate=10)
        models=models1+models2+models3+models4
        models=selectModels(models,20)
        alignedModels=[alignGPModel(mods,inputData,response) for mods in models]
        ensemble=ensembleSelect(alignedModels,inputData,response)
        out=maximizeUncertainty(ensemble,dims,rangesP)
        while out in inputData.T:
            out=maximizeUncertainty(ensemble,dims,subSampleSpace(rangesP))
        inputData=extendData(inputData,out)
        response=func(inputData)
        fitList=np.array([fitness(mod,testInput,testResponse) for mod in alignedModels])
        errors.append(min(fitList[np.logical_not(np.isnan(fitList))]))
        minerr=errors[-1]
        if minerr<1e-14:
            #print("Points needed in round", j,": ",3+i, " Time needed: ", time.perf_counter()-roundTime)
            if not os.path.exists(str(eqNum)):
                os.makedirs(str(eqNum))
            path=os.path.join(str(eqNum),str(version))
            file=open(path,"wb+")
            dill.dump([i,inputData,response,testInput,testResponse,errors,models,minerr],file)
            file.close()
            file=open(path+'.txt','w+')
            file.write(str(i+3)+'\n')
            file.write(str(errors))
            file.close()
            return 3+i
            found=True
            ptsNeeded.append(3+i)
            break
        activeLearningCheckpoint(eqNum,version,i,inputData,response,testInput,testResponse,errors,models,minerr)
    if found==False:
        #print("Points needed in round",j,": NA (model not found)")
        path=os.path.join(str(eqNum),str(version))
        file=open(path,"wb")
        dill.dump([-1,inputData,response,testInput,testResponse,errors,models,minerr],file)
        file.close()
        file=open(path+'.txt',"w+")
        file.write(str(i+3)+"\n")
        file.write(str(errors))
        file.close()
        return -1

def plotModels(models, modelExpression=False):
    tMods=copy.deepcopy(models)
    if len(tMods[0][2])<2:
        # add complexity as second value
        for mod in tMods:
            mod[2]=[mod[2][0],stackGPModelComplexity(mod)]
    [modelToListForm(mod) for mod in tMods]
    paretoModels=paretoTournament(tMods)
    for i in paretoModels:
        tMods.remove(i)
    [modelRestoreForm(mod) for mod in paretoModels]
    [modelRestoreForm(mod) for mod in tMods]
    
    pAccuracies=[mod[2][0] for mod in paretoModels]
    pComplexities=[mod[2][1] for mod in paretoModels]
    
    accuracies=[mod[2][0] for mod in tMods]+pAccuracies
    complexities=[mod[2][1] for mod in tMods]+pComplexities
    colors=['blue' for i in range(len(tMods))]+['red' for i in range(len(pAccuracies))]
    
    fig,ax = plt.subplots()

    sc=plt.scatter(complexities,accuracies,color=colors)
    plt.xlabel("Complexity")
    plt.ylabel("1-R**2")

    if modelExpression:
        names=[str(printGPModel(mod)) for mod in tMods]+[str(printGPModel(mod)) for mod in paretoModels]
    else:
        names = [str(mod) for mod in tMods]+[str(mod) for mod in paretoModels]
    
    label = ax.annotate("", xy=(0,0), xytext=(np.min(complexities),np.mean([np.max(accuracies),np.min(accuracies)])),
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    label.set_visible(False)
    
    def update_labels(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        label.xy = pos
        text = "{}".format(" ".join([names[n] for n in [ind["ind"][0]]]))
        label.set_text(text)
        label.get_bbox_patch().set_facecolor('grey')
        label.get_bbox_patch().set_alpha(0.9)


    def hover(event):
        vis = label.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_labels(ind)
                label.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    label.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

def plotModelResponseComparison(model,inputData,response,sort=False):
    plt.scatter(range(len(response)),response,label="True Response")
    plt.scatter(range(len(response)),evaluateGPModel(model,inputData),label="Model Prediction")
    plt.legend()
    plt.xlabel("Data Index")
    plt.ylabel("Response Value")
    plt.show()
def plotPredictionResponseCorrelation(model,inputData,response):
    plt.scatter(response,evaluateGPModel(model,inputData),label="Model")
    plt.plot(response,response,label="Perfect Correlation",color='green')
    plt.xlabel("True Response")
    plt.ylabel("Predicted Response")
    plt.legend()
    plt.show()
#Plot model complexity distribution
def plotModelComplexityDistribution(models):
    tMods=copy.deepcopy(models)
    [modelToListForm(mod) for mod in tMods]
    paretoModels=paretoTournament(tMods)
    for i in paretoModels:
        tMods.remove(i)
    [modelRestoreForm(mod) for mod in paretoModels]
    [modelRestoreForm(mod) for mod in tMods]
    pComplexities=[mod[2][1] for mod in paretoModels]
    tComplexities=[mod[2][1] for mod in tMods]
    plt.hist(tComplexities,label="Non-Pareto Models")
    plt.hist(pComplexities,label="Pareto Models")
    plt.xlabel("Model Complexity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
#Plot model accuracy distribution
def plotModelAccuracyDistribution(models):
    tMods=copy.deepcopy(models)
    [modelToListForm(mod) for mod in tMods]
    paretoModels=paretoTournament(tMods)
    for i in paretoModels:
        tMods.remove(i)
    [modelRestoreForm(mod) for mod in paretoModels]
    [modelRestoreForm(mod) for mod in tMods]
    pAccuracies=[mod[2][0] for mod in paretoModels]
    tAccuracies=[mod[2][0] for mod in tMods]
    plt.hist(tAccuracies,label="Non-Pareto Models")
    plt.hist(pAccuracies,label="Pareto Models")
    plt.xlabel("Model Accuracy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
#Plot model residuals relative to response
def plotModelResiduals(model,input,response):
    plt.scatter(response,evaluateGPModel(model,input)-response)
    plt.xlabel("Response")
    plt.ylabel("Residual")
    plt.show()
#Plot model residual distribution
def plotModelResidualDistribution(model,input,response):
    plt.hist(evaluateGPModel(model,input)-response)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()
#Plot the presence of variables in a model population
def plotVariablePresence(models,variables=["x"+str(i) for i in range(100)],sort=False):
    vars=[varReplace(model[1],variables) for model in models]
    #Remove all numeric entries in vars
    vars=[[i for i in var if type(i)!=int and type(i)!=float] for var in vars]
    #Merge into one list
    vars=[j for i in vars for j in i]
    #Count frequency of each variable in vars
    varFreqs=[vars.count(i) for i in variables]
    #Keep only variables that appear at least once
    variablesUsed=[variables[i] for i in range(len(varFreqs)) if varFreqs[i]>0]
    varFreqs=[varFreqs[i] for i in range(len(varFreqs)) if varFreqs[i]>0]
    if sort:
        order=np.argsort(varFreqs)[::-1]
        variablesUsed=[variablesUsed[i] for i in order]
        varFreqs=[varFreqs[i] for i in order]
    #Plot variable frequency
    plt.bar(variablesUsed,varFreqs)
    plt.xlabel("Variable")
    plt.ylabel("Frequency")
    plt.show()
def replaceOpsWithStrings(opStack):
    model = copy.deepcopy(opStack)
    model = replaceFunc(model,exp,str("exp"))
    model = replaceFunc(model,tan,str("tan"))
    model = replaceFunc(model,sqrt,str("sqrt"))
    model = replaceFunc(model,inv,str("1/#"))
    model = replaceFunc(model,sin,str("sin"))
    model = replaceFunc(model,cos,str("cos"))
    model = replaceFunc(model,arccos,str("acos"))
    model = replaceFunc(model,arcsin,str("asin"))
    model = replaceFunc(model,arctan,str("atan"))
    model = replaceFunc(model,tanh,str("tanh"))
    model = replaceFunc(model,log,str("log"))
    model = replaceFunc(model,add,"+")
    model = replaceFunc(model,mult,"*")
    model = replaceFunc(model,sub,"-")
    model = replaceFunc(model,protectDiv,"/")
    model = replaceFunc(model,sqrd,"^2")
    return model
#Plot the presence of operators in a model population
def plotOperatorPresence(models,sort=False,excludePop=True):
    ops=[replaceOpsWithStrings(model[0]) for model in models]
    #Merge into one list
    ops=[j for i in ops for j in i]
    #Remove duplicates in ops
    uniqueOps=list(set(ops))
    if excludePop:
        #Remove pop operator
        uniqueOps.remove('pop')
    #Count frequency of each operator in ops
    opFreqs=[ops.count(i) for i in uniqueOps]
    #Keep only operators that appear at least once
    opsUsed=[str(uniqueOps[i]) for i in range(len(opFreqs)) if opFreqs[i]>0]
    opFreqs=[opFreqs[i] for i in range(len(opFreqs)) if opFreqs[i]>0]
    if sort:
        order=np.argsort(opFreqs)[::-1]
        opsUsed=[opsUsed[i] for i in order]
        opFreqs=[opFreqs[i] for i in order]
    #Plot operator frequency
    plt.bar(opsUsed,opFreqs)
    #Rotate x axis labels
    plt.xticks(rotation=0)
    plt.xlabel("Operator")
    plt.ylabel("Frequency")
    plt.show()

############################
#Sharpness Computations
############################

def sharpnessConstants(model,inputData,responseData,numPerturbations=10,percentPerturbation=0.2):

    fits=[]

    #For each model parameter, if numeric, randomly perturb by x% and see how much the model changes
    for i in range(numPerturbations):
        tempModel=copy.deepcopy(model)
        newParameters=[param if callable(param) else param*(1+percentPerturbation*(np.random.uniform()-0.5)) for param in model[1]]
        tempModel[1]=newParameters
        fits.append(fitness(tempModel,inputData,responseData))
    return np.std(fits)

def sharpnessData(model,inputData,responseData,numPerturbations=10,percentPerturbation=0.2,preserveSign=False):

    fits=[]

    #For each vector, randomly perturb by x% of the standard deviation and see how much the model fitness changes
    for i in range(numPerturbations):
        tempData=copy.deepcopy(inputData)
        tempData=np.array([(vec+percentPerturbation*np.std(vec)*(np.random.uniform(size=len(vec))-0.5)) for vec in tempData])
        if preserveSign:
            signs=[np.unique(var) for var in np.sign(inputData)]
            tempData=[signs[i]*abs(tempData[i]) if len(signs[i])==1 else tempData[i] for i in range(len(signs))]
        fits.append(fitness(model,tempData,responseData))
    return np.std(fits)

def totalSharpness(model,inputData,responseData,numPerturbations=10,percentPerturbation=0.2,preserveSign=False):

    return sharpnessConstants(model,inputData,responseData,numPerturbations=numPerturbations,percentPerturbation=percentPerturbation)+sharpnessData(model,inputData,responseData,numPerturbations=numPerturbations,percentPerturbation=percentPerturbation,preserveSign=preserveSign)

############################
#Multiple Independent Searches
############################
def runEpochs(x,y,epochs=5,**kwargs):
    models=[]
    for i in range(epochs):
        models+=evolve(x,y,**kwargs)

    return sortModels(models)


############################
#Parallelization
############################
from joblib import Parallel, delayed
def parallelEvolve(*args,n_jobs=-1,avail_cores=-1, cascades=False, cascadeCount=10, exchangeCount=5, **kwargs):
    capTime = False
    liveTracking = False
    if avail_cores==-1:
        try:
            avail_cores=len(os.sched_getaffinity(0))
        except:
            avail_cores=os.cpu_count()
    if n_jobs==-1:
        try:
            n_jobs=len(os.sched_getaffinity(0))
        except:
            n_jobs=os.cpu_count()
    bestFits = [[] for _ in range(n_jobs)]
    if "liveTracking" in kwargs and kwargs["liveTracking"]:
        liveTracking = True
    if "tracking" in kwargs and kwargs["tracking"]:
        kwargs["returnTracking"]=True
        

    print(f"Running parallel evolution with {n_jobs} jobs.")
    if "liveTracking" in kwargs and kwargs["liveTracking"] and cascades==False:
        print("Live tracking is not supported in parallel evolution without cascades, disabling live tracking.")
        kwargs["liveTracking"]=False

    if cascades:
        if "capTime" in kwargs and kwargs["capTime"]:
            startTime = time.perf_counter()
            capTime = True
        if liveTracking:
                fig, ax = plt.subplots(figsize=(20,10))
                kwargs["returnTracking"]=True
        argList = [copy.deepcopy(kwargs) for _ in range(n_jobs)]
        for i in range(n_jobs):
            argList[i]["tracking"]=False
            argList[i]["liveTracking"]=False
        for cs in range(cascadeCount):
            #print(f"Starting cascade {cs+1}/{cascadeCount}")
            if cs==0:
                runs = Parallel(n_jobs=avail_cores, backend="loky")(delayed(evolve)(*args, **argList[1]) for _ in range(n_jobs))
            else:
                for i in range(n_jobs):
                    exchangeOrder = random.sample(range(n_jobs), n_jobs)
                    argList[i]["initialPop"]=runs[i]+copy.deepcopy(random.sample(runs[exchangeOrder[i]], min(exchangeCount, len(runs[exchangeOrder[i]]))))
                runs = Parallel(n_jobs=avail_cores, backend="loky")(delayed(evolve)(*args, **argList[i]) for i in range(n_jobs))
            if liveTracking or ("returnTracking" in kwargs and kwargs["returnTracking"]): 
                runs, tracking = zip(*runs)
                for i in range(n_jobs):
                    bestFits[i].extend(tracking[i])
                if liveTracking:
                    ax.clear()
                    for i in range(n_jobs):
                        ax.plot(bestFits[i], label=f'Run {i+1}')
                    ax.set_title(f"Best Model: {min([bst[-1] for bst in bestFits]):.2f} at Cascade {(cs+1)}")
                    ax.set_xlabel("Generations")
                    ax.set_ylabel("Fitness")
                    if n_jobs <= 16:  # Only show legend if there are a reasonable number of jobs
                        ax.legend()
                    clear_output(wait=True) 
                    display(fig)       
                    plt.close(fig)
            if capTime: 
                for i in range(n_jobs):
                    argList[i]["timeLimit"]=kwargs["timeLimit"]-(time.perf_counter()-startTime)
                if kwargs["timeLimit"]-(time.perf_counter()-startTime)  <= 0:
                    break 
        if "returnTracking" in kwargs and kwargs["returnTracking"]:
            if liveTracking:
                clear_output(wait=True)
            plt.figure(figsize=(12, 6))
            for i in range(n_jobs):
                plt.plot(bestFits[i], label=f'Job {i+1}')
            plt.title('Best Fitness Over Generations for Each Parallel Run')
            plt.xlabel('Generations')
            plt.ylabel('Best Fitness')
            if n_jobs <= 16:  # Only show legend if there are a reasonable number of jobs
                plt.legend()
            plt.show()

    else:    
        runs = Parallel(n_jobs=avail_cores, backend="loky")(delayed(evolve)(*args, **kwargs) for _ in range(n_jobs))
        if ("tracking" in kwargs and kwargs["tracking"]):
            runs, tracking = zip(*runs)
            # plot tracking for each job
            plt.figure(figsize=(12, 6))
            for i, track in enumerate(tracking):
                plt.plot(track, label=f'Job {i+1}')
            plt.title('Best Fitness Over Generations for Each Parallel Run')
            plt.xlabel('Generations')
            plt.ylabel('Best Fitness')
            if n_jobs <= 16:  # Only show legend if there are a reasonable number of jobs
                plt.legend()
            plt.show()
    flat = [model for sublist in runs for model in sublist]
    return sortModels(flat)


############################
#Benchmarking
############################
def generateRandomBenchmark(numVars=5, numSamples=100, noiseLevel=0, opsChoices=defaultOps(), constChoices=defaultConst(), maxLength=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    # Generate random input data
    inputData = np.random.rand(numVars, numSamples)

    # Generate a random target function
    randomModel = generateRandomModel(numVars, opsChoices, constChoices, maxLength)

    # Evaluate the model to get response data
    responseData = evaluateGPModel(randomModel, inputData)

    # Add noise if specified
    if noiseLevel > 0:
        noise = np.random.normal(0, noiseLevel, size=responseData.shape)
        responseData += noise

    return inputData, responseData, randomModel


############################
# Feature 1: Multi-Output / Vector Symbolic Regression
############################

def evolveMultiOutput(inputData, responseData, **kwargs):
    """Evolve one GP model population per output row in responseData.

    inputData    : array of shape (n_features, n_samples)
    responseData : array of shape (n_outputs, n_samples) or (n_samples,) for
                   a single output.

    Returns a list of model populations, one per output.
    """
    responseData = np.array(responseData)
    if responseData.ndim == 1:
        responseData = responseData.reshape(1, -1)
    numOutputs = responseData.shape[0]
    populations = []
    for i in range(numOutputs):
        populations.append(evolve(inputData, responseData[i], **kwargs))
    return populations

evolveMultiOutput.__doc__ = ("evolveMultiOutput(inputData, responseData, **kwargs) evolves one GP "
                              "model population per output row and returns a list of populations")


def evaluateMultiOutputModel(populations, inputData):
    """Evaluate the best model from each population.

    Returns an array of shape (n_outputs, n_samples).
    """
    return np.array([evaluateGPModel(pop[0], inputData) for pop in populations])

evaluateMultiOutputModel.__doc__ = ("evaluateMultiOutputModel(populations, inputData) evaluates the "
                                     "best model from each population and returns a (n_outputs, n_samples) array")


############################
# Feature 3: Island-Model / Migration GP (explicit wrapper around parallelEvolve)
############################

def islandEvolve(inputData, responseData, numIslands=4, totalGenerations=100,
                  migrationInterval=10, migrationCount=5, **kwargs):
    """Island-model GP: independent sub-populations that periodically share individuals.

    numIslands         : number of independent sub-populations (islands).
    totalGenerations   : total number of generations to run.
    migrationInterval  : number of generations between migrations (cascade length).
    migrationCount     : number of Pareto-front individuals to send between islands.
    **kwargs           : additional arguments forwarded to evolve().

    Internally uses parallelEvolve with cascades=True.
    """
    numCascades = max(1, totalGenerations // migrationInterval)
    return parallelEvolve(
        inputData, responseData,
        n_jobs=numIslands,
        cascades=True,
        cascadeCount=numCascades,
        exchangeCount=migrationCount,
        generations=migrationInterval,
        **kwargs
    )

islandEvolve.__doc__ = ("islandEvolve(inputData, responseData, numIslands=4, totalGenerations=100, "
                         "migrationInterval=10, migrationCount=5, **kwargs) runs island-model GP "
                         "with periodic migration between sub-populations")


############################
# Feature 4: Gradient-Based Constant Optimization
############################

def optimizeModelGradient(model, inputData, responseData, method='L-BFGS-B',
                           bounds=None, eps=1e-5, **kwargs):
    """Optimize numeric constants in a model using gradient-based L-BFGS-B.

    Uses numerical finite differences for the Jacobian.  Faster than
    differential_evolution for models with a small number of constants.

    method : scipy.optimize.minimize method string (default 'L-BFGS-B').
    bounds : list of (lo, hi) pairs, one per constant.  Defaults to (-1e6, 1e6).
    eps    : finite-difference step size for gradient estimation.
    """
    indices = get_numeric_indices(model[1])
    if not indices:
        return model
    startingVals = np.array([model[1][i] for i in indices], dtype=float)
    fnc = lambda x: replaceEvaluate(model, x, inputData, responseData)

    def grad(x):
        g = np.zeros_like(x)
        for j in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[j] += eps
            xm[j] -= eps
            fp = fnc(xp)
            fm = fnc(xm)
            g[j] = (fp - fm) / (2.0 * eps) if (np.isfinite(fp) and np.isfinite(fm)) else 0.0
        return g

    if bounds is None:
        bounds = [(-1e6, 1e6) for _ in startingVals]
    out = minimize(fnc, startingVals, jac=grad, method=method, bounds=bounds, **kwargs)
    newModel = copy.deepcopy(model)
    for i in range(len(indices)):
        newModel[1][indices[i]] = float(out.x[i])
    setModelQuality(newModel, inputData, responseData)
    return newModel

optimizeModelGradient.__doc__ = ("optimizeModelGradient(model, inputData, responseData, method='L-BFGS-B', "
                                   "bounds=None, eps=1e-5, **kwargs) optimizes numeric constants with gradient descent")


############################
# Feature 5: Grammar / Type Constraints on Operator Selection
############################

def makeTypeConstrainedOps(ops, variableTypes, typeOpConstraints):
    """Return a filtered ops list compatible with the given variable types.

    variableTypes     : list of type strings, one per variable
                        e.g. ['angle', 'positive', 'real']
    typeOpConstraints : dict mapping type string -> list of allowed operators
                        e.g. {'angle': [sin, cos, tan],
                               'positive': [log, sqrt],
                               'real': allOps()}

    Only operators that are allowed for at least one variable type present in
    variableTypes (plus 'pop') are included in the returned list.
    """
    allowedOps = set()
    for vtype in set(variableTypes):
        if vtype in typeOpConstraints:
            allowedOps.update(typeOpConstraints[vtype])
    return [op for op in ops if op == 'pop' or op in allowedOps]

makeTypeConstrainedOps.__doc__ = ("makeTypeConstrainedOps(ops, variableTypes, typeOpConstraints) returns "
                                   "a filtered operator list respecting per-variable type constraints")


############################
# Feature 6: Symbolic Simplification-Aware Duplicate Removal
############################

def deleteDuplicateModelsSimplified(models, timeout=2):
    """Remove semantically equivalent models by comparing simplified SymPy expressions.

    More thorough than deleteDuplicateModelsPhenotype because it calls
    sympy.simplify before comparing, catching algebraically equivalent forms.
    Falls back to treating a model as unique if simplification times out.
    """
    simplified = []
    kept = []
    for mod in models:
        try:
            with time_limit(timeout):
                expr = sym.simplify(printGPModel(mod))
                sexpr = str(expr)
        except (TimeoutException, Exception):
            sexpr = "_unique_" + str(len(kept))
        if sexpr not in simplified:
            simplified.append(sexpr)
            kept.append(mod)
    return kept

deleteDuplicateModelsSimplified.__doc__ = ("deleteDuplicateModelsSimplified(models, timeout=2) removes "
                                            "semantically equivalent models using sympy.simplify")


############################
# Feature 7: Niching / Diversity Preservation (NSGA-II Crowding Distance)
############################

def crowdingDistance(fitValues):
    """Compute NSGA-II crowding distances for a 2-D array of fitness values.

    fitValues : numpy array of shape (n_models, n_objectives)
    Returns   : 1-D array of crowding distances (length n_models).
                Boundary individuals receive distance inf.
    """
    n = len(fitValues)
    if n == 0:
        return np.array([])
    nObj = fitValues.shape[1]
    distances = np.zeros(n)
    for obj in range(nObj):
        order = np.argsort(fitValues[:, obj])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        objRange = fitValues[order[-1], obj] - fitValues[order[0], obj]
        if objRange == 0:
            continue
        for i in range(1, n - 1):
            distances[order[i]] += (
                (fitValues[order[i + 1], obj] - fitValues[order[i - 1], obj]) / objRange
            )
    return distances

crowdingDistance.__doc__ = ("crowdingDistance(fitValues) computes NSGA-II crowding distances "
                              "for a (n_models, n_objectives) array of fitness values")


def nichingSelect(models, popSize):
    """Select models using Pareto rank + crowding distance (NSGA-II style).

    Iteratively extracts Pareto fronts; within the last front that would
    overflow the budget, individuals are ranked by crowding distance so
    that structurally diverse models are preferred.
    """
    tMods = copy.deepcopy(models)
    [modelToListForm(mod) for mod in tMods]
    selected = []
    remaining = tMods[:]

    while len(selected) < popSize and remaining:
        fitVals = np.array([mod[2] for mod in remaining], dtype=float)
        front_mask = paretoFront(fitVals)
        front = [remaining[i] for i in range(len(remaining)) if front_mask[i]]

        if len(selected) + len(front) <= popSize:
            selected.extend(front)
            remaining = [remaining[i] for i in range(len(remaining)) if not front_mask[i]]
        else:
            need = popSize - len(selected)
            frontFits = np.array([mod[2] for mod in front], dtype=float)
            dists = crowdingDistance(frontFits)
            order = np.argsort(dists)[::-1]
            selected.extend([front[i] for i in order[:need]])
            break

    [modelRestoreForm(mod) for mod in selected]
    return selected

nichingSelect.__doc__ = ("nichingSelect(models, popSize) selects models using Pareto rank and "
                          "crowding distance to preserve diversity in the population")


############################
# Feature 8: Constraint-Satisfaction / Physics-Informed Loss
############################

def makeConstrainedFitness(baseFitness, constraints):
    """Create a physics-informed fitness function with soft constraint penalties.

    baseFitness : base fitness function with signature f(model, inputData, response)
    constraints : list of (constraint_fn, weight) tuples where
                  constraint_fn(model, inputData, response) -> float (0 = fully satisfied)

    Returns a new fitness function that adds weighted penalty terms to the base fitness.
    """
    def constrainedFitness(model, inputData, response):
        base = baseFitness(model, inputData, response)
        if np.isnan(base):
            return np.nan
        penalty = sum(w * cf(model, inputData, response) for cf, w in constraints)
        return base + penalty
    constrainedFitness.__name__ = getattr(baseFitness, '__name__', 'fitness') + "_constrained"
    return constrainedFitness

makeConstrainedFitness.__doc__ = ("makeConstrainedFitness(baseFitness, constraints) wraps a fitness "
                                   "function with soft constraint penalties for physics-informed SR")


############################
# Feature 9: Save/Load Model Populations (Serialization API)
############################

STACKGP_VERSION = "1.0"


def savePopulation(models, path, metadata=None):
    """Save a model population to disk with versioning metadata.

    models   : list of GP models to save.
    path     : file path (string) to write to.
    metadata : optional dict of user-supplied metadata (e.g. variable names, fitness).
    """
    data = {
        'stackgp_version': STACKGP_VERSION,
        'timestamp': time.time(),
        'models': models,
        'metadata': metadata or {}
    }
    with open(path, 'wb') as f:
        dill.dump(data, f)

savePopulation.__doc__ = ("savePopulation(models, path, metadata=None) serialises a model "
                           "population to disk with version and timestamp metadata")


def loadPopulation(path):
    """Load a model population from disk.

    Returns a (models, metadata) tuple.
    For legacy files that stored a raw model list, metadata is an empty dict.
    """
    with open(path, 'rb') as f:
        data = dill.load(f)
    if isinstance(data, dict) and 'models' in data:
        return data['models'], data.get('metadata', {})
    return data, {}

loadPopulation.__doc__ = ("loadPopulation(path) deserialises a model population from disk "
                           "and returns (models, metadata)")


############################
# Feature 10: Sklearn-Compatible Estimator Interface
############################

try:
    from sklearn.base import BaseEstimator, RegressorMixin as _RegressorMixin

    class StackGPRegressor(BaseEstimator, _RegressorMixin):
        """Scikit-learn compatible wrapper for StackGP symbolic regression.

        Implements fit(X, y) / predict(X) / score(X, y) following the sklearn
        BaseEstimator / RegressorMixin interface so that StackGPRegressor can
        be used in sklearn pipelines, cross-validation, and GridSearchCV.

        Parameters
        ----------
        generations : int
            Number of evolutionary generations.
        popSize : int
            Population size.
        ops : list or None
            Operator set.  Defaults to defaultOps() when None.
        align : bool
            Whether to align models after evolution.
        elitismRate : int
            Percentage of population preserved as elites each generation.
        timeLimit : float
            Maximum wall-clock time in seconds (used when capTime=True).
        capTime : bool
            Whether to cap evolution by wall-clock time.
        """

        def __init__(self, generations=100, popSize=300, ops=None, align=True,
                     elitismRate=10, timeLimit=300, capTime=False):
            self.generations = generations
            self.popSize = popSize
            self.ops = ops
            self.align = align
            self.elitismRate = elitismRate
            self.timeLimit = timeLimit
            self.capTime = capTime

        def fit(self, X, y):
            """Fit the GP model to training data.

            X : array-like of shape (n_samples, n_features)
            y : array-like of shape (n_samples,)
            """
            X = np.array(X)
            y = np.array(y)
            inputData = X.T  # StackGP expects (n_features, n_samples)
            ops = self.ops if self.ops is not None else defaultOps()
            self.models_ = evolve(
                inputData, y,
                generations=self.generations,
                popSize=self.popSize,
                ops=ops,
                align=self.align,
                elitismRate=self.elitismRate,
                timeLimit=self.timeLimit,
                capTime=self.capTime,
            )
            self.best_model_ = self.models_[0]
            return self

        def predict(self, X):
            """Predict using the best evolved model.

            X : array-like of shape (n_samples, n_features)
            """
            X = np.array(X)
            pred = evaluateGPModel(self.best_model_, X.T)
            return np.array(pred)

        def get_expression(self):
            """Return the symbolic expression string of the best model."""
            return str(printGPModel(self.best_model_))

except ImportError:
    pass


############################
# Feature 11: Categorical / Mixed-Type Variable Support
############################

class CategoricalEncoder:
    """Encode categorical variables for use with StackGP.

    Numeric columns are passed through unchanged.  Categorical columns are
    encoded using either one-hot ('onehot') or ordinal ('ordinal') encoding.

    Parameters
    ----------
    encoding            : 'onehot' or 'ordinal'
    categorical_indices : list of int column indices to treat as categorical,
                          or None to auto-detect.
    threshold           : maximum number of unique values for auto-detection.

    Input arrays are expected in StackGP's (n_features, n_samples) layout.
    """

    def __init__(self, encoding='onehot', categorical_indices=None, threshold=10):
        self.encoding = encoding
        self.categorical_indices = categorical_indices
        self.threshold = threshold
        self.categories_ = {}
        self._cat_idx = []

    def fit(self, X):
        """Fit encoder to X (shape: n_features × n_samples or n_samples × n_features)."""
        X = np.array(X, dtype=object)
        if X.shape[0] > X.shape[1]:
            X = X.T  # ensure (n_features, n_samples)
        n_features = X.shape[0]
        if self.categorical_indices is None:
            self._cat_idx = [
                j for j in range(n_features)
                if len(np.unique(X[j])) <= self.threshold
            ]
        else:
            self._cat_idx = list(self.categorical_indices)
        for j in self._cat_idx:
            self.categories_[j] = sorted(set(X[j]))
        return self

    def transform(self, X):
        """Encode X and return array of shape (n_features_out, n_samples)."""
        X = np.array(X, dtype=object)
        if X.shape[0] > X.shape[1]:
            X = X.T  # ensure (n_features, n_samples)
        n_features = X.shape[0]
        out_cols = []
        for j in range(n_features):
            if j in self._cat_idx:
                cats = self.categories_[j]
                if self.encoding == 'onehot':
                    for c in cats:
                        out_cols.append((X[j] == c).astype(float))
                else:
                    mapping = {c: i for i, c in enumerate(cats)}
                    out_cols.append(np.array([mapping.get(v, -1) for v in X[j]], dtype=float))
            else:
                out_cols.append(X[j].astype(float))
        return np.array(out_cols)

    def fit_transform(self, X):
        """Fit and transform X in one step."""
        return self.fit(X).transform(X)


############################
# Feature 12: Adaptive Operator Probabilities
############################

def adaptiveEvolve(inputData, responseData, generations=100, ops=None,
                   const=None, popSize=300, mutationRate=79, crossoverRate=11,
                   spawnRate=10, elitismRate=10, maxComplexity=100, align=True,
                   initialPop=None, tourneySize=5,
                   modelEvaluationMetrics=None):
    """Evolve with adaptive operator probabilities using credit assignment.

    Operators that appear in children with fitness improvements receive a
    higher selection probability in subsequent generations.  Credit is
    computed as the mean fitness improvement (lower fitness is better) of
    children containing each operator relative to their parents.

    All parameters mirror those of evolve().
    """
    if ops is None:
        ops = defaultOps()
    if const is None:
        const = defaultConst()
    if initialPop is None:
        initialPop = []
    if modelEvaluationMetrics is None:
        metrics = [fitness, stackGPModelComplexity]
    elif callable(modelEvaluationMetrics):
        metrics = [modelEvaluationMetrics]
    else:
        metrics = list(modelEvaluationMetrics)

    fullInput = copy.deepcopy(inputData)
    fullResponse = copy.deepcopy(responseData)
    variableCount = varCount(fullInput)

    uniqueOps = list(dict.fromkeys(ops))
    opWeights = {op: 1.0 for op in uniqueOps}

    def weightedOpList():
        wts = np.array([opWeights[op] for op in uniqueOps], dtype=float)
        wts = wts / wts.sum()
        counts = np.maximum(1, np.round(wts * len(uniqueOps) * 2).astype(int))
        return [op for op, cnt in zip(uniqueOps, counts) for _ in range(cnt)]

    models = initializeGPModels(variableCount, ops, const, popSize)
    models = models + copy.deepcopy(initialPop)

    for gen in range(generations):
        for mod in models:
            setModelQuality(mod, fullInput, fullResponse, modelEvaluationMetrics=metrics)
        models = removeIndeterminateModels(models)
        if not models:
            models = initializeGPModels(variableCount, ops, const, popSize)
            continue

        prevFits = {id(m): m[2][0] for m in models if m[2] and not np.isnan(m[2][0])}
        paretoModels = selectModels(
            models,
            elitismRate / 100 * popSize if elitismRate / 100 * popSize < len(models) else len(models)
        )
        models = tournamentModelSelection(models, popSize, tourneySize)

        currentOps = weightedOpList()
        toMutate = random.sample(models, min(round(mutationRate / 100 * popSize), len(models)))
        childModels = paretoModels[:]
        opCredits = {op: [] for op in uniqueOps}

        for parent in toMutate:
            child = mutate(parent, variableCount, currentOps, const)
            setModelQuality(child, fullInput, fullResponse, modelEvaluationMetrics=metrics)
            if child[2] and not np.isnan(child[2][0]):
                parentFit = prevFits.get(id(parent), child[2][0])
                improvement = parentFit - child[2][0]
                for op in child[0]:
                    if op in opCredits:
                        opCredits[op].append(improvement)
            childModels.append(child)

        crossoverPairs = random.sample(models, min(round(crossoverRate / 100 * popSize), len(models)))
        for j in range(round(len(crossoverPairs) / 2) - 1):
            childModels = childModels + recombination2pt(
                crossoverPairs[j],
                crossoverPairs[j + round(len(crossoverPairs) / 2)]
            )

        for op in uniqueOps:
            if opCredits[op]:
                opWeights[op] = max(0.05, opWeights[op] + 0.1 * np.mean(opCredits[op]))
        total = sum(opWeights.values())
        for op in uniqueOps:
            opWeights[op] = opWeights[op] / total * len(uniqueOps)

        childModels += initializeGPModels(variableCount, ops, const, round(spawnRate / 100 * popSize))
        childModels = deleteDuplicateModels(childModels)
        childModels = [m for m in childModels if stackGPModelComplexity(m) < maxComplexity]
        if len(childModels) < popSize:
            childModels += initializeGPModels(variableCount, ops, const, popSize - len(childModels))
        models = copy.deepcopy(childModels)

    for mod in models:
        setModelQuality(mod, fullInput, fullResponse, modelEvaluationMetrics=metrics)
    models = [trimModel(m) for m in models]
    models = deleteDuplicateModels(models)
    models = removeIndeterminateModels(models)
    models = sortModels(models)
    if align:
        models = [alignGPModel(m, fullInput, fullResponse) for m in models]
        for mod in models:
            setModelQuality(mod, fullInput, fullResponse, modelEvaluationMetrics=metrics)
    return models

adaptiveEvolve.__doc__ = ("adaptiveEvolve(inputData, responseData, generations=100, ops=None, ...) "
                            "evolves with adaptive operator probabilities via credit assignment")


############################
# Feature 13: Pareto Front Export / Reporting
############################

def summarizeModels(models, inputData, response, variableNames=None):
    """Return a pandas DataFrame summarising the Pareto-front models.

    Columns: Expression, R2, RMSE, Complexity, Variables.

    models        : list of GP models (typically the output of evolve()).
    inputData     : array of shape (n_features, n_samples).
    response      : array of shape (n_samples,).
    variableNames : list of variable name strings.  Auto-generated if None.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for summarizeModels. Install with: pip install pandas")

    n_vars = varCount(inputData)
    if variableNames is None:
        variableNames = ["x" + str(i) for i in range(n_vars)]

    tMods = copy.deepcopy(models)
    [modelToListForm(mod) for mod in tMods]
    paretoMods = paretoTournament(tMods)
    [modelRestoreForm(mod) for mod in paretoMods]

    rows = []
    symVars = symbols(variableNames[:n_vars])
    for mod in paretoMods:
        expr = str(printGPModel(mod, inputData=symVars))
        pred = evaluateGPModel(mod, inputData)
        try:
            pred = np.array(pred, dtype=float)
            resp = np.array(response, dtype=float)
            if np.all(np.isfinite(pred)) and not np.all(pred == pred[0]):
                r2_val = float(pearsonr(pred, resp)[0] ** 2)
                rmse_val = float(np.sqrt(np.mean((pred - resp) ** 2)))
            else:
                r2_val, rmse_val = np.nan, np.nan
        except Exception:
            r2_val, rmse_val = np.nan, np.nan

        complexity = stackGPModelComplexity(mod)
        varList = varReplace(mod[1], variableNames[:n_vars])
        used = sorted(set(v for v in varList if isinstance(v, str)))
        rows.append({
            'Expression': expr,
            'R2': r2_val,
            'RMSE': rmse_val,
            'Complexity': complexity,
            'Variables': ', '.join(used)
        })
    return pd.DataFrame(rows)

summarizeModels.__doc__ = ("summarizeModels(models, inputData, response, variableNames=None) returns "
                            "a pandas DataFrame with expression, R², RMSE, complexity, and variables "
                            "for each Pareto-front model")


############################
# Feature 14: Interactive Model Exploration Dashboard
############################

def modelDashboard(models, inputData, response, variableNames=None):
    """Interactive model exploration dashboard using ipywidgets.

    Displays a dropdown of Pareto-front models.  Selecting a model renders:
      - a Pareto front scatter with the selected model highlighted,
      - a prediction-vs-truth scatter plot, and
      - a residual plot.

    Requires ipywidgets and an IPython / Jupyter environment.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display as ipy_display
    except ImportError:
        print("ipywidgets is required for modelDashboard. Install with: pip install ipywidgets")
        return

    n_vars = varCount(inputData)
    if variableNames is None:
        variableNames = ["x" + str(i) for i in range(n_vars)]

    tMods = copy.deepcopy(models)
    [modelToListForm(mod) for mod in tMods]
    paretoMods = paretoTournament(tMods)
    [modelRestoreForm(mod) for mod in paretoMods]

    exprs = [str(printGPModel(m, inputData=symbols(variableNames[:n_vars]))) for m in paretoMods]
    complexities = [
        mod[2][1] if len(mod[2]) > 1 else stackGPModelComplexity(mod)
        for mod in paretoMods
    ]
    accuracies = [mod[2][0] for mod in paretoMods]

    options = [
        (f"#{i}: {exprs[i][:60]}..." if len(exprs[i]) > 60 else f"#{i}: {exprs[i]}", i)
        for i in range(len(paretoMods))
    ]
    dropdown = widgets.Dropdown(options=options, description='Model:',
                                layout=widgets.Layout(width='80%'))
    output = widgets.Output()

    def on_change(change):
        idx = change['new']
        mod = paretoMods[idx]
        pred = evaluateGPModel(mod, inputData)
        with output:
            output.clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].scatter(complexities, accuracies, color='red', s=50)
            axes[0].scatter([complexities[idx]], [accuracies[idx]],
                            color='gold', s=150, zorder=5, label='Selected')
            axes[0].set_xlabel("Complexity")
            axes[0].set_ylabel("1-R\u00b2")
            axes[0].set_title("Pareto Front")
            axes[0].legend()

            resp_arr = np.array(response)
            pred_arr = np.array(pred)
            axes[1].scatter(resp_arr, pred_arr, alpha=0.6)
            mn, mx = resp_arr.min(), resp_arr.max()
            axes[1].plot([mn, mx], [mn, mx], 'g-', label='Perfect')
            axes[1].set_xlabel("True Response")
            axes[1].set_ylabel("Predicted")
            axes[1].set_title("Prediction vs Truth")
            axes[1].legend()

            residuals = pred_arr - resp_arr
            axes[2].scatter(resp_arr, residuals, alpha=0.6)
            axes[2].axhline(0, color='green', linestyle='--')
            axes[2].set_xlabel("True Response")
            axes[2].set_ylabel("Residual")
            axes[2].set_title("Residuals")

            plt.suptitle(f"Expression: {exprs[idx]}", fontsize=9)
            plt.tight_layout()
            plt.show()
            print(f"Expression: {exprs[idx]}")

    dropdown.observe(on_change, names='value')
    ipy_display(widgets.VBox([dropdown, output]))
    on_change({'new': 0})

modelDashboard.__doc__ = ("modelDashboard(models, inputData, response, variableNames=None) shows an "
                           "interactive Pareto-front exploration dashboard powered by ipywidgets")


############################
# Feature 15: Noise-Robust Fitness Metrics
############################

def huberFitness(model, inputData, response, delta=1.0):
    """Huber-loss fitness function, robust to outliers.

    Returns the mean Huber loss:
      L(e) = 0.5*e^2              if |e| <= delta
           = delta*(|e| - 0.5*delta)  otherwise
    """
    pred = evaluateGPModel(model, inputData)
    if pred is None:
        return np.nan
    pred = np.array(pred, dtype=float)
    resp = np.array(response, dtype=float)
    if not np.all(np.isfinite(pred)):
        return np.nan
    err = pred - resp
    loss = np.where(np.abs(err) <= delta,
                    0.5 * err ** 2,
                    delta * (np.abs(err) - 0.5 * delta))
    return float(np.mean(loss))

huberFitness.__doc__ = ("huberFitness(model, inputData, response, delta=1.0) returns mean Huber "
                         "loss, robust to outliers")


def madFitness(model, inputData, response):
    """Median Absolute Deviation fitness function, highly robust to outliers."""
    pred = evaluateGPModel(model, inputData)
    if pred is None:
        return np.nan
    pred = np.array(pred, dtype=float)
    resp = np.array(response, dtype=float)
    if not np.all(np.isfinite(pred)):
        return np.nan
    return float(np.median(np.abs(pred - resp)))

madFitness.__doc__ = ("madFitness(model, inputData, response) returns the Median Absolute "
                       "Deviation, highly robust to outliers")


def trimmedR2Fitness(model, inputData, response, trimFraction=0.1):
    """Trimmed R\u00b2 fitness: 1-R\u00b2 computed after removing the most extreme residuals.

    trimFraction : fraction of samples with the largest absolute residuals to
                   exclude before computing R\u00b2 (0 = no trimming, 0.5 = half removed).
    """
    pred = evaluateGPModel(model, inputData)
    if pred is None:
        return np.nan
    pred = np.array(pred, dtype=float)
    resp = np.array(response, dtype=float)
    if not np.all(np.isfinite(pred)):
        return np.nan
    n = len(resp)
    nTrim = int(np.floor(trimFraction * n))
    order = np.argsort(np.abs(pred - resp))
    idx = order[:n - nTrim] if nTrim > 0 else order
    try:
        r2 = pearsonr(pred[idx], resp[idx])[0] ** 2
    except Exception:
        return 1.0
    if np.isnan(r2):
        return 1.0
    return 1.0 - r2

trimmedR2Fitness.__doc__ = ("trimmedR2Fitness(model, inputData, response, trimFraction=0.1) returns "
                              "1-R\u00b2 computed on the central (1-2*trimFraction) fraction of samples, "
                              "down-weighting outliers")
