import numpy as np
import matplotlib.pyplot as plt
# M = [[0, 1/2, 1, 0],[1/3,0,0,1/2],[1/3,0,0,1/2],[1/3,1/2,0,0]]
# 等级泄露rank leak:一个网页只有入链，没有出链
# M = [[0, 0, 0, 1/2], [1,0,0,0], [0,0,0,1/2], [0,1,0,0]]
# 等级沉没rank sink:一个网页没有入链，只有出链
M = [[0,0,1/2,1],[1,0,0,0],[0,0,0,0],[0,1,1/2,0]]
M = np.array(M)
print(M)
w0 = np.ones((4,1)) / 4

A = []
B = []
C = []
D = []
times = 10
wi = w0
A.append(wi[0,0])
B.append(wi[1,0])
C.append(wi[2,0])
D.append(wi[3,0])
for i in range(times):
    wi = np.dot(M, wi)
    A.append(wi[0,0])
    B.append(wi[1,0])
    C.append(wi[2,0])
    D.append(wi[3,0])

print(wi)

plt.plot([i for i in range(times+1)], A, label='A')
plt.plot([i for i in range(times+1)], B, label='B')
plt.plot([i for i in range(times+1)], C, label='C')
plt.plot([i for i in range(times+1)], D, label='D')
plt.legend()
plt.show()