def make_L_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def clear_axes_lines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

def clear_axis_ticks(ax):
    clear_x_axis_ticks(ax)
    clear_y_axis_ticks(ax)

def clear_x_axis_ticks(ax):
    ax.set_xticks([])
    ax.set_xticklabels([])

def clear_y_axis_ticks(ax):
    ax.set_yticks([])
    ax.set_yticklabels([])

def clear_axes(ax):
    clear_axes_lines(ax)
    clear_x_axis_ticks(ax)
    clear_y_axis_ticks(ax)

def square_aspect(ax):
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    
    ar = abs(max_y - min_y)/abs(max_x - min_x)
    ax.set_aspect(1/ar)


