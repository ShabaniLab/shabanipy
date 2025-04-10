import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def pad(x, n=2):
    x = str(x)
    if len(x) < n:
        x = '0'*(n - len(x)) + x
    return x

def rgb_to_hex(color):
    return '#' + ''.join(pad(hex(i).lstrip('0x')) for i in color)

line_colors = [
   (12, 121, 192),
   (221, 102, 51),
   (140, 69, 153),
   (238, 184, 53),
   (135, 181, 73),
   (93, 196, 240),
   (171, 41, 66),
   (50, 190, 30),
   (255, 142, 255),
   (0, 0, 0),
]

line_colors = [rgb_to_hex(color) for color in line_colors]

def hex_color_to_rgb(hex_color):
    hex_color = hex_color.strip('#')
    return [
        int(hex_color[i*2:(i+1)*2], 16)/255 for i in range(3) 
    ]

def normalize_color(color):
    return [i/255 for i in color]

def make_color_dict(colors):
    color_dict = {
        'red': [],
        'blue': [],
        'green': [],
    }
    N = len(colors) - 1
    for i, color in enumerate(colors):
        if isinstance(color, str):
            color = hex_color_to_rgb(color)
        if any(i > 1 for i in color):
            color = normalize_color(color)
        for j, c in enumerate(['red', 'green', 'blue']):
            color_dict[c].append((i/N, color[j], color[j]))
    return color_dict

def register_color_map(cmap_name, colors):
    cdict = make_color_dict(colors)
    cmap = LinearSegmentedColormap(cmap_name, cdict)
    plt.register_cmap(cmap = cmap)

color_pts = [
    (45, 96, 114),
    (243, 210, 181),
    (242, 184, 164),
    (242, 140, 143),
    (208, 127, 127)
]

register_color_map('jy_pink', color_pts)

color_pts = [
    '582841',
    'EF4648',
    'F36F38',
    'F99E4C',
    'CC2A49',
]
register_color_map('jy_red', color_pts)

color_pts = [
    '325D79',
    '9BD7D1',
    'EFEEEE',
    'F9A26C',
    'F26627',
]
register_color_map('jy_orange', color_pts)

