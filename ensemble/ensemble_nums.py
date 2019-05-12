import numpy as np
import random
import matplotlib.pyplot as plt
singleAccuracy = 0.55
samples = 200
ensembleAccuracy = []
classifierNum = np.arange(0, 300)
for i in range(len(classifierNum)):
    sampleCount = 0
    for sample in range(samples):
        numCount = 0
        for num in range(classifierNum[i]):
            if random.random() < singleAccuracy:
                numCount += 1
        if (float(numCount) / classifierNum[i]) > 0.5:
            sampleCount += 1
    accuracy = float(sampleCount) / samples
    ensembleAccuracy.append(accuracy)
plt.plot(classifierNum, ensembleAccuracy)
plt.xlabel('classifier number')
plt.ylabel('accuracy')
plt.show()