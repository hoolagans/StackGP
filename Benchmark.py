import StackGP as sgp
import numpy as np
import time

#Performance Test
def test(func, dimensions, ranges, numberOfPoints=100, numberOfTestsPoints=200):
    inputData=[]
    testInput=[]
    for i in range(dimensions):
        inputData.append(np.random.uniform(ranges[i][0],ranges[i][1],numberOfPoints))
        testInput.append(np.random.uniform(ranges[i][0],ranges[i][1],numberOfTestsPoints))
    inputData=np.array(inputData)
    testInput=np.array(testInput)
    response=func(inputData)
    testResponse=func(testInput)
    errors=[]
    models=[]
    minerr=1
    models1=sgp.evolve(inputData,response,initialPop=models,generations=1000,tracking=False,popSize=300,ops=sgp.allOps(),timeLimit=120,capTime=True,align=False,elitismRate=10)
    models2=sgp.evolve(inputData,response,initialPop=models,generations=1000,tracking=False,popSize=300,ops=sgp.allOps(),timeLimit=120,capTime=True,align=False,elitismRate=10)
    models=models1+models2
    models=sgp.selectModels(models,20)
    alignedModels=[sgp.alignGPModel(mods,inputData,response) for mods in models]
    fitList=np.array([sgp.fitness(mod,testInput,testResponse) for mod in alignedModels])
    minerr=min(fitList[np.logical_not(np.isnan(fitList))])
    return minerr, fitList

#Speed Test
def speedTest(func, dimensions, ranges, numberOfPoints=100, numberOfTestsPoints=200):
    inputData=[]
    testInput=[]
    for i in range(dimensions):
        inputData.append(np.random.uniform(ranges[i][0],ranges[i][1],numberOfPoints))
        testInput.append(np.random.uniform(ranges[i][0],ranges[i][1],numberOfTestsPoints))
    inputData=np.array(inputData)
    testInput=np.array(testInput)
    response=func(inputData)
    testResponse=func(testInput)
    #Record start time
    start=time.time()
    models=sgp.evolve(inputData,response,generations=1000,popSize=300,ops=sgp.allOps(),capTime=False,align=True,elitismRate=10)
    #Record end time
    end=time.time()
    #Return time taken
    return end-start

def batches(func, dimensions, ranges, numberOfPoints=100, numberOfTestPoints=200, repeats=10):
    errs=[]
    for i in range(repeats):
        err,fit=test(func,dimensions,ranges,numberOfPoints,numberOfTestPoints)
        errs.append(err)
    return min(errs), np.median(errs), np.mean(errs), max(errs), np.std(errs)

def speedBatch(func,dimension,ranges,numberOfPoints=100,numberOfTestPoints=200,repeats=10):
    times=[]
    for i in range(repeats):
        times.append(speedTest(func,dimension,ranges,numberOfPoints,numberOfTestPoints))
    return min(times), np.median(times), np.mean(times), max(times), np.std(times)


minerrs=[]
mederrs=[]
meanerrs=[]
maxerrs=[]
std=[]
fits=[]

#Feynman EQ2
f1=lambda data: (np.exp((-((data[0])/data[1])**2)/2)/(np.sqrt(2*np.pi)*data[1]))
err=batches(f1,2,[[1,3],[1,3]],100,200)
minerrs.append(err[0])
mederrs.append(err[1])
meanerrs.append(err[2])
maxerrs.append(err[3])
std.append(err[4])
print("Feynman EQ2")
print("Error: "+str(err))

#Feynman EQ3
f2=lambda data: (np.exp((-((data[0]-data[1])/data[2])**2)/2)/(np.sqrt(2*np.pi)*data[2]))
err=batches(f2,3,[[1,3],[1,3],[1,3]],100,200)
minerrs.append(err[0])
mederrs.append(err[1])
meanerrs.append(err[2])
maxerrs.append(err[3])
std.append(err[4])
print("Feynman EQ3")
print("Error: "+str(err))

#Feynman EQ4
f3=lambda data: (np.sqrt((data[1]-data[2])**2+(data[3]-data[2])**2))
err=batches(f3,4,[[1,5],[1,5],[1,5],[1,5]],100,200)
minerrs.append(err[0])
mederrs.append(err[1])
meanerrs.append(err[2])
maxerrs.append(err[3])
std.append(err[4])
print("Feynman EQ4")
print("Error: "+str(err))

#Feynman EQ91
f4=lambda data: (data[0]*np.sqrt(data[1]**2+data[2]**2+data[3]**2))
err=batches(f4,4,[[1,5],[1,5],[1,5],[1,5]],100,200)
minerrs.append(err[0])
mederrs.append(err[1])
meanerrs.append(err[2])
maxerrs.append(err[3])
std.append(err[4])
print("Feynman EQ91")
print("Error: "+str(err))

#Feynman EQ27
f5=lambda data: (1/(1/data[0]+data[1]/data[2]))
err=batches(f5,3,[[1,5],[1,5],[1,5]],100,200)
minerrs.append(err[0])
mederrs.append(err[1])
meanerrs.append(err[2])
maxerrs.append(err[3])
std.append(err[4])
print("Feynman EQ27")
print("Error: "+str(err))

#Speed Test
print("Speed Test")
print("Feynman EQ91")
speed=speedBatch(f4,4,[[1,5],[1,5],[1,5],[1,5]],100,200)
print("Time: "+str(speed))


file=open("BenchmarkResults.txt","w+")
file.write("Feynman EQ2\n")
file.write("Min Error: "+str(minerrs[0])+"\n")
file.write("Median Error: "+str(mederrs[0])+"\n")
file.write("Mean Error: "+str(meanerrs[0])+"\n")
file.write("Max Error: "+str(maxerrs[0])+"\n")
file.write("Standard Deviation: "+str(std[0])+"\n")
file.write("Feynman EQ3\n")
file.write("Min Error: "+str(minerrs[1])+"\n")
file.write("Median Error: "+str(mederrs[1])+"\n")
file.write("Mean Error: "+str(meanerrs[1])+"\n")
file.write("Max Error: "+str(maxerrs[1])+"\n")
file.write("Standard Deviation: "+str(std[1])+"\n")
file.write("Feynman EQ4\n")
file.write("Min Error: "+str(minerrs[2])+"\n")
file.write("Median Error: "+str(mederrs[2])+"\n")
file.write("Mean Error: "+str(meanerrs[2])+"\n")
file.write("Max Error: "+str(maxerrs[2])+"\n")
file.write("Standard Deviation: "+str(std[2])+"\n")
file.write("Feynman EQ91\n")
file.write("Min Error: "+str(minerrs[3])+"\n")
file.write("Median Error: "+str(mederrs[3])+"\n")
file.write("Mean Error: "+str(meanerrs[3])+"\n")
file.write("Max Error: "+str(maxerrs[3])+"\n")
file.write("Standard Deviation: "+str(std[3])+"\n")
file.write("Feynman EQ27\n")
file.write("Min Error: "+str(minerrs[4])+"\n")
file.write("Median Error: "+str(mederrs[4])+"\n")
file.write("Mean Error: "+str(meanerrs[4])+"\n")
file.write("Max Error: "+str(maxerrs[4])+"\n")
file.write("Standard Deviation: "+str(std[4])+"\n")
file.write("Speed Test\n")
file.write("Min Time: "+str(speed[0])+"\n")
file.write("Median Time: "+str(speed[1])+"\n")
file.write("Mean Time: "+str(speed[2])+"\n")
file.write("Max Time: "+str(speed[3])+"\n")
file.write("Standard Deviation: "+str(speed[4])+"\n")
file.close()




