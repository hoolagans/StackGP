from StackGP import *
import sympy as sym
import pandas as pd
import numpy as np
import sys
print(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
def runExperiment(file,targetID,IDrange,name):
    #Import data file
    data=pd.read_csv(file)#"/Users/nathanhaut/Downloads/pmlb/datasets/195_auto_price/195_auto_price.csv")
    
    #Extract data from file
    #vars=data.columns


    #Split train and test data
    trainSize=np.floor(len(data)*0.7)
    testSize=len(data)-trainSize

    trainIndices=np.random.choice(len(data),int(trainSize),replace=False)
    testIndices=np.setdiff1d(np.arange(len(data)),trainIndices)

    trainData=data.iloc[trainIndices]
    testData=data.iloc[testIndices]

    #Extract input and response data
    trainInput=np.array(trainData.T)[:-1]
    trainResponse=np.array(trainData.T)[-1]
    testInput=np.array(testData.T)[:-1]
    testResponse=np.array(testData.T)[-1]


    #Get position of max, min, and mean values of response
    maxPos=np.argmax(trainResponse)
    minPos=np.argmin(trainResponse)
    diff=np.abs(trainResponse-np.mean(trainResponse))
    meanPos=np.argmin(diff)
    

    #Create target basis set function
    func=basisFunctionComplexityDiff(targetID,IDrange,len(trainInput),np.array(trainData)[minPos][:-1],np.array(trainData)[maxPos][:-1],np.array(trainData)[meanPos][:-1])
    #Evolve models using three approaches: ID-informed, complexity-informed, and standard tournament
    ID3Omodels=evolve(trainInput,trainResponse,modelEvaluationMetrics=[fitness,stackGPModelComplexity],tourneySize=40,generations=100,align=False,elitismRate=10,popSize=300,alternateObjectives=[func],alternateObjFrequency=10)
    temp=[mods for mods in ID3Omodels if mods[2][2]<1]
    if len(temp)>0:
        ID3Omodels=temp
    print("ID3O models done", ID3Omodels[0][2])
    IDmodels=evolve(trainInput,trainResponse,modelEvaluationMetrics=[fitness],tourneySize=20,generations=100,align=False,elitismRate=10,popSize=300,alternateObjectives=[func],alternateObjFrequency=10)
    temp=[mods for mods in IDmodels if mods[2][1]<1]
    if len(temp)>0:
        IDmodels=temp
    print("ID models done", IDmodels[0][2])

    compModels=evolve(trainInput,trainResponse,tourneySize=20,generations=100,align=False,elitismRate=10,popSize=300)
    print("Complexity models done")
    tourneyModels=evolve(trainInput,trainResponse,modelEvaluationMetrics=[fitness],tourneySize=5,generations=100,align=False,elitismRate=10,popSize=300)
    print("Tourney models done")

    #Select target models from approaches
    IDmodel=IDmodels[0]
    ID3Omodel=ID3Omodels[0]
    compModel=compModels[0]
    tourneyModel=tourneyModels[0]

    #Train Stats
    IDStats=IDmodels[0][2]
    ID3OStats=ID3Omodels[0][2]
    compStats=compModels[0][2]
    tourneyStats=tourneyModels[0][2]

    #Align models
    IDmodel=alignGPModel(IDmodel,trainInput,trainResponse)
    ID3Omodel=alignGPModel(ID3Omodel,trainInput,trainResponse)
    compModel=alignGPModel(compModel,trainInput,trainResponse)
    tourneyModel=alignGPModel(tourneyModel,trainInput,trainResponse)

    #Evaluate models on test data
    IDfitness=fitness(IDmodel,testInput,testResponse)
    ID3Ofitness=fitness(ID3Omodel,testInput,testResponse)
    compFitness=fitness(compModel,testInput,testResponse)
    tourneyFitness=fitness(tourneyModel,testInput,testResponse)

    IDRMSE=np.linalg.norm(evaluateGPModel(IDmodel,testInput)-testResponse)
    ID3ORMSE=np.linalg.norm(evaluateGPModel(ID3Omodel,testInput)-testResponse)
    compRMSE=np.linalg.norm(evaluateGPModel(compModel,testInput)-testResponse)
    tourneyRMSE=np.linalg.norm(evaluateGPModel(tourneyModel,testInput)-testResponse)

    #Save results
    results=pd.DataFrame({'ID':[IDStats, printGPModel(IDmodel),IDfitness,IDRMSE],'ID3O':[ID3OStats, printGPModel(ID3Omodel),ID3Ofitness,ID3ORMSE],'Complexity':[compStats, printGPModel(compModel),compFitness,compRMSE],'Tourney':[tourneyStats, printGPModel(tourneyModel),tourneyFitness,tourneyRMSE]})
    results.to_csv('Results/'+name+'.csv')


    #Return target models and fitnesses on test data
    return results

def runTrials(file, target,dev,name,count):
    #Create variables to store output

    #Loop through trials
    for i in range(count):
        #Run experiments
        results=runExperiment(file,target,dev,name+str(i))

    return 0


runTrials("/Users/nathanhaut/Downloads/pmlb/datasets/"+str(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),str(sys.argv[4]),10)