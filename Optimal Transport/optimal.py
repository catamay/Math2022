import numpy as np
import scipy as sp
import scipy.stats as st
import matplotlib.pyplot as plt
from WGPOT.wgpot import logmap
from WGPOT.utils import Plot_GP

ran = np.linspace(-10,20,num=1000)
norm1 = st.norm.pdf(ran)
norm2 = st.norm.pdf(ran,loc=10, scale=2)
print(st.wasserstein_distance(norm1,norm2))

norms = np.array([st.norm.pdf(ran, loc=i, scale=(1+3*i/10)**(1/2)) for i in range(11)]).T
plt.plot(ran,norms, 'g--')
plt.plot(ran,norm1, 'b')
plt.plot(ran,norm2, 'r')
plt.show()