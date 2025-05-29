import os
import sys

repo_dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_dirname)

support_dirname = os.path.join(repo_dirname, 'test', 'support')
temp_dirname = os.path.join(repo_dirname, 'test', '_temp')

from rayleigh.util import TicToc

tt = TicToc()


def save_synthetic_image(color, dirname, size=100):
    """
    Save a solid color image of the given hex color to the given directory.
    """
    # omit the # sign in the filename
    filename = os.path.join(dirname, color[1:] + '.png')
    cmd = "convert -size {size}x{size} 'xc:{color}' '{filename}'"
    os.system(cmd.format(**locals()))
    return filename
