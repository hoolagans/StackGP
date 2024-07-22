#!/usr/bin/env python
# coding: utf-8

import random
import matplotlib.pyplot as plt 
import numpy as np
import math
import copy
import sys
from scipy.stats import pearsonr # for computing correlation
from functools import reduce #for flattening lists
from operator import concat  #for flattening lists
from scipy.stats import trim_mean # for ensemble evaluation
from scipy.stats import differential_entropy
import warnings
import time
import dill
import os
from sklearn.cluster import KMeans #for clustering in ensemble definition
from scipy.optimize import minimize #for uncertainty maximization
from sympy import symbols
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
    return a**b
def sqrt(a):
    return np.sqrt(a)
def sqrd(a):
    return a**2
def inv(a):
    return np.array(a).astype(float)**(-1)
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

def defaultOps():
    return [protectDiv,add,sub,mult,exp,sqrd,sqrt,inv,"pop","pop","pop","pop","pop","pop"]
def allOps():
    return [protectDiv,add,sub,mult,exp,sqrd,sqrt,inv,cos,sin,tan,arccos,arcsin,arctan,tanh,log,"pop","pop","pop","pop","pop","pop","pop","pop","pop","pop"]
def randomInt(a=-3,b=3):
    return random.randint(a,b)
def defaultConst():
    return [np.pi, np.e, randomInt,ranReal ]
def ranReal(a=20,b=-10):
    return random.random()*a-b


############################
#Data Subsampling Methods
############################
def randomSubsample(x,y):
    n=max(int(np.ceil(len(y)**(3/5))),3)
    idx=np.random.choice(range(x.shape[1]),n,replace=False)
    return np.array([i[idx] for i in x]),y[idx]

def generationProportionalSample(x,y,generation=100,generations=100):
    n=max(int(np.ceil(len(y)*(generation/generations)**(3/5))),3)
    idx=np.random.choice(range(x.shape[1]),n,replace=False)
    return np.array([i[idx] for i in x]),y[idx]

import inspect
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
def selectModels(models, selectionSize=0.5):
    tMods=copy.deepcopy(models)
    [modelToListForm(mod) for mod in tMods]
    paretoModels=[]
    if selectionSize<=1:
        selection=selectionSize*len(models)
    else:
        selection=selectionSize
    
    while len(paretoModels)<selection:
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
    try:
        align=np.round(np.polyfit(prediction,response,1,rcond=1e-16),decimals=14)
    except np.linalg.LinAlgError:
        #print("Alignment failed for: ", model, " with prediction: ", prediction, "and reference data: ", response)
        return model
    newModel=trimModel(model)
    newModel[0]=np.array(newModel[0].tolist()+[mult,add],dtype=object)
    newModel[1]=newModel[1]+align.tolist()
    setModelQuality(newModel,data,response)
    return newModel
alignGPModel.__doc__ = "alignGPModel(model, input, response) aligns a model such that response-a*f(x)+b are minimized over a and b"
def evolve(inputData, responseData, generations=100, ops=defaultOps(), const=defaultConst(), variableNames=[], mutationRate=79, crossoverRate=11, spawnRate=10, extinction=False,extinctionRate=10,elitismRate=50,popSize=300,maxComplexity=100,align=True,initialPop=[],timeLimit=300,capTime=False,tourneySize=5,tracking=False,modelEvaluationMetrics=[fitness,stackGPModelComplexity],dataSubsample=False,samplingMethod=randomSubsample):
    
    fullInput,fullResponse=copy.deepcopy(inputData),copy.deepcopy(responseData)
    inData=copy.deepcopy(fullInput)
    resData=copy.deepcopy(fullResponse)
    variableCount=varCount(inData)
    models=initializeGPModels(variableCount,ops,const,popSize)
    models=models+initialPop
    startTime=time.perf_counter()
    bestFits=[]
    for i in range(generations):
        if capTime and time.perf_counter()-startTime>timeLimit:
            break
        if dataSubsample:
            inData,resData=samplingMethod(fullInput,fullResponse)
        for mods in models:
            setModelQuality(mods,inData,resData,modelEvaluationMetrics=modelEvaluationMetrics)
        models=removeIndeterminateModels(models)
        if tracking:
            bestFits.append(min([mods[2][0] for mods in paretoTournament(models)]))

        #paretoModels=paretoTournament(models)
        paretoModels=selectModels(models,elitismRate/100*popSize if elitismRate/100*popSize<len(models) else len(models))
        if extinction and i%extinctionRate:
            models=initializeGPModels(variableCount,ops,const,popSize)
            for mods in models:
                setModelQuality(mods,inData,resData,modelEvaluationMetrics=modelEvaluationMetrics)
        
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
        setModelQuality(mods,fullInput,fullResponse,modelEvaluationMetrics=modelEvaluationMetrics)
    models=[trimModel(mod) for mod in models]
    models=deleteDuplicateModels(models)
    models=removeIndeterminateModels(models)
    models=sortModels(models)
    if align:
        models=[alignGPModel(mods,fullInput,fullResponse) for mods in models]
    
    if tracking:
        bestFits.append(min([mods[2][0] for mods in paretoTournament(models)])) 
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
    response=evModHelper(model[1],model[0],[],np.array(inputData))[2][0]
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
        uncertainties=[uncertainty(res,0) for res in responses]
    else:
        
        uncertainties=[uncertainty(responses,0)]
    
    return uncertainties
def relativeEnsembleUncertainty(ensemble,inputData):
    output=evaluateModelEnsemble(ensemble,inputData)
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

def plotModels(models):
    tMods=copy.deepcopy(models)
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
    names=[str(printGPModel(mod)) for mod in tMods]+[str(printGPModel(mod)) for mod in paretoModels]
    
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