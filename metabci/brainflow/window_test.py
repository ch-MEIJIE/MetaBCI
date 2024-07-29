# %%
import matplotlib.pyplot as plt
# %%
if __name__ == "__main__":
    STIM_POS = {
    "0": ((0.465, 0.149), (0.544, 0.273)),
    "1": ((0.383, 0.727), (0.453, 0.850)),
    "2": ((0.222, 0.329), (0.292, 0.455)),
    "3": ((0.708, 0.329), (0.778, 0.455)),
    "4": ((0.222, 0.547), (0.292, 0.669)),
    "5": ((0.708, 0.547), (0.778, 0.669)),
    "6": ((0.465, 0.547), (0.544, 0.669)),
    "7": ((0.546, 0.727), (0.616, 0.850)),
    "8": ((0.305, 0.305), (0.694, 0.696))
    }
    length = 16
    width = 9

    # create a figure and axis with the given length and width
    fig, ax = plt.subplots(figsize=(length, width))
    # Each value in STIM_POS represents a square on the screen,
    # so draw them
    for key, ((x1, y1), (x2, y2)) in STIM_POS.items():
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, key, ha='center', va='center')

    # display the plot
    plt.show()
# %%
