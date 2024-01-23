import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

is_visualize = False


for simnum in range(200):
    if is_visualize:
        pdf = PdfPages('../visualization/naca_ellipse_boundary_sim_{:06d}.pdf'.format(simnum))
    for i in range(1):
        f = open("../boundary_test/sim_{0}/boundary_{1}.txt".format(simnum, i))
        data = f.read()

        # replacing end splitting the text 
        # when newline ('\n') is seen.
        data_into_list = re.split("\n|[|[ |, | ]|]", data)
        print(data_into_list)

        a = []
        for w in data_into_list:
             if w:
                a.append(float(w))
 
        print(a)

        b = []
        for v in a:
            if v != 0.:
                b.append(v)

        npb = np.array(b)

        if is_visualize:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.plot(npb[0::2], npb[1::2], marker='o', linewidth=0, markersize=2)
            plt.xlim([0, 62])
            plt.ylim([0, 62])
            pdf.savefig()

        np.save("../dataset/naca_ellipse/test_trajectories/sim_{:06d}/boundary".format(simnum), np.stack([npb[0::2], npb[1::2]]))

    if is_visualize:
        pdf.close()
