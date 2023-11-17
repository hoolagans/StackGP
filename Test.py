import numpy as np
import StackGP as sgp
pts=100
input=[[np.random.normal(1,5) for i in range(pts)],[np.random.normal(1,2) for i in range(pts)],[np.random.normal(3,10) for i in range(pts)]]
input=np.array(input)
response=input[0]/(1.3-input[1]/input[2])
models=sgp.evolve(input,response,generations=600,tracking=True,ops=sgp.allOps())
print(sgp.printGPModel(models[0]))
print(models[0])
