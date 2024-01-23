import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os

is_visualize = False

for simnum in range(5):
    if is_visualize:
        pdf = PdfPages('../visualization/naca_ellipse_multiple_boundary_sim_{:06d}.pdf'.format(simnum))
    try:    
        os.makedirs("../np_boundary_multiple/sim_{:06d}".format(simnum))
    except Exception:
        pass   
    for i in range(3):
        f = open("../boundary_multiple/sim_{0}/boundary_{1}.txt".format(simnum, i))
        data = f.read()

        # replacing end splitting the text 
        # when newline ('\n') is seen.
        data_into_list = re.split("\n|[|[ |, | ]|]", data)
        print(data_into_list)

        a = []
        for w in data_into_list:
             if w:
                a.append(float(w))
 
        #print(a)

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

        np.save("../np_boundary_multiple/sim_{:06d}/boundary_{:06d}".format(simnum, i), np.stack([npb[0::2], npb[1::2]]))

    if is_visualize:
        pdf.close()
