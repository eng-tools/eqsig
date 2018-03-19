import numpy as np

noise = (np.random.random(size=4000) - 0.5) * 9.8
vals = ["dt=0.005", "acc: m/s2"]
for i in range(len(noise)):
    vals.append("%.4f" % noise[i])
para = "\n".join(vals)
a = open("noise_test_x.txt", "w")
a.write(para)
a.close()
