# coding=utf-8
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from matplotlib import cm
from scipy._lib.six import xrange

from VISUALIZATION.channelLocation import CHANNEL_9_APPROX, get_channelpos


def ax_scalp(v, channels,ax=None, annotate=True,
             vmin=None, vmax=None, cmap=cm.coolwarm,
             scalp_line_width=1,
             scalp_line_style='solid',
             chan_pos_list=CHANNEL_9_APPROX,
             interpolation='bilinear',
             fontsize=8):
    """
    :param v: 输入通道对应的相关性值
    :param channels: 通道的坐标位置
    :param ax: 最后的存到ax这个框架里
    :param annotate:
    :param vmin:
    :param vmax:
    :param cmap:
    :param scalp_line_width:头皮轮廓的线宽
    :param scalp_line_style:头皮轮廓的线的形状
    :param chan_pos_list:整个电极通道的位置
    :param interpolation:双线性插值
    :param fontsize:字体大小
    :return: 一张头皮图
    """
    if ax is None:
        ax = plt.gca()
    assert len(v) == len(channels), "Should be as many values as channels"
    assert interpolation == 'bilinear' or interpolation == 'nearest'
    if vmin is None:
        # added by me (robintibor@gmail.com)
        assert vmax is None
        vmin, vmax = -np.max(np.abs(v)), np.max(np.abs(v))
    # what if we have an unknown channel?得到对应电极通道在列表中正确的位置
    points = [get_channelpos(c, chan_pos_list) for c in channels]
    for c in channels:
        assert get_channelpos(c, chan_pos_list) is not None, (
            "Expect " + c + " to exist in positions")
    z = [v[i] for i in range(len(points))]
    # calculate the interpolation
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    # interpolate the in-between values插值填充
    xx = np.linspace(min(x), max(x), 1000)
    yy = np.linspace(min(y), max(y), 1000)
    if interpolation == 'bilinear':
        xx_grid, yy_grid = np.meshgrid(xx, yy)#生成网格点坐标矩阵
        #对输入数据进行三角测量，并在每个执行线性重心插值的三角形上构造插值

        f = interpolate.LinearNDInterpolator(list(zip(x, y)), z)#

        zz = f(xx_grid, yy_grid)
    else:
        assert interpolation == 'nearest'
        f = interpolate.NearestNDInterpolator(list(zip(x, y)), z)
        assert len(xx) == len(yy)
        zz = np.ones((len(xx), len(yy)))
        for i_x in xrange(len(xx)):
            for i_y in xrange(len(yy)):
                # somehow this is correct. don't know why :(
                zz[i_y, i_x] = f(xx[i_x], yy[i_y])
                # zz[i_x,i_y] = f(xx[i_x], yy[i_y])
        assert not np.any(np.isnan(zz))

    # plot map开始画头皮图
    image = ax.imshow(zz, vmin=vmin, vmax=vmax, cmap=cmap,
                      extent=[min(x), max(x), min(y), max(y)], origin='lower',
                      interpolation=interpolation)

    if scalp_line_width > 0:
        # paint the head
        ax.add_artist(plt.Circle((0, 0), 1, linestyle=scalp_line_style,
                                 linewidth=scalp_line_width, fill=False))
        # add a nose
        ax.plot([-0.1, 0, 0.1], [1, 1.1, 1], color='black',
                linewidth=scalp_line_width, linestyle=scalp_line_style)
        # add ears
        _add_ears(ax, scalp_line_width, scalp_line_style)
    # add markers at channels positions
    # set the axes limits, so the figure is centered on the scalp
    ax.set_ylim([-1.05, 1.15])
    ax.set_xlim([-1.15, 1.15])

    # hide the frame and ticks
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # draw the channel names
    if annotate:
        for i in zip(channels, list(zip(x, y))):
            ax.annotate(" " + i[0], i[1], horizontalalignment="center",
                        verticalalignment='center', fontsize=fontsize)
    ax.set_aspect(1)
    plt.show()
    return image

#添加耳朵
def _add_ears(ax, linewidth, linestyle):
    start_x = np.cos(10 * np.pi / 180.0)
    start_y = np.sin(10 * np.pi / 180.0)
    end_x = np.cos(-15 * np.pi / 180.0)
    end_y = np.sin(-15 * np.pi / 180.0)
    verts = [
        (start_x, start_y),
        (start_x + 0.05, start_y + 0.05),  # out up
        (start_x + 0.1, start_y),  # further out, back down
        (start_x + 0.11, (end_y * 0.7 + start_y * 0.3)),  # midpoint
        (end_x + 0.14, end_y),  # down out start
        (end_x + 0.05, end_y - 0.05),  # down out further
        (end_x, end_y),  # endpoint
    ]
    codes = [Path.MOVETO] + [Path.CURVE3] * (len(verts) - 1)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none',
                              linestyle=linestyle, linewidth=linewidth)
    ax.add_patch(patch)
    verts_left = [(-x, y) for x, y in verts]
    path_left = Path(verts_left, codes)
    patch_left = patches.PathPatch(path_left, facecolor='none',
                                   linestyle=linestyle, linewidth=linewidth)
    ax.add_patch(patch_left)

corr_data = [-0.487487150695782,	-0.495225022786884,	0.150318922392934,	-0.216631134185799,	-0.132130062121033,	-0.169622996820690,	0.0592720733259083,	-0.232156092553636,	-0.150692721260417]
channel = ['FLV', 'FRV', 'BP', 'F3-C3', 'T3-P3', 'P3-O1', 'F4-C4', 'T4-P4', 'P4-O2']
x = ax_scalp(corr_data,channel)
print(x)
