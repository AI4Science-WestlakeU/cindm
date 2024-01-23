import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


def gradient_color(frame, frame_count, cmap='RdBu'):
    jet = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=0, vmax=frame_count)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    return scalarMap.to_rgba(frame)


def burgers_figure(title):
    plt.ylabel("$u(x)$")
    plt.xlabel("x")
    plt.grid(True, axis='y')
    plt.title(title)