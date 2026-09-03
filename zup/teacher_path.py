import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ============================================================
# 1. Data
#    Strictly from the LaTeX table provided by you.
#    Only mean values are plotted; ± std is not used as a
#    coordinate in this 3D visualization.
# ============================================================

teachers = {
    'RWKVFusion': {
        'Dl': 0.0157,
        'Ds': 0.0357,
        'HQNR': 0.9491,
        'Dl_std': 0.0058,
        'Ds_std': 0.0053,
        'HQNR_std': 0.0097,
    },

    'ARConv': {
        'Dl': 0.0146,
        'Ds': 0.0279,
        'HQNR': 0.9579,
        'Dl_std': 0.0059,
        'Ds_std': 0.0068,
        'HQNR_std': 0.0103,
    },

    'PanNet': {
        'Dl': 0.0169,
        'Ds': 0.0470,
        'HQNR': 0.9371,
        'Dl_std': 0.0074,
        'Ds_std': 0.0213,
        'HQNR_std': 0.0271,
    },

    'FusionMamba': {
        'Dl': 0.0192,
        'Ds': 0.0269,
        'HQNR': 0.9544,
        'Dl_std': 0.0081,
        'Ds_std': 0.0058,
        'HQNR_std': 0.0112,
    }
}


students = {
    'RWKVFusion': {
        'Dl': 0.0167,
        'Ds': 0.0213,
        'HQNR': 0.9624,
        'Dl_std': 0.0040,
        'Ds_std': 0.0042,
        'HQNR_std': 0.0068,
    },

    'ARConv': {
        'Dl': 0.0167,
        'Ds': 0.0204,
        'HQNR': 0.9632,
        'Dl_std': 0.0046,
        'Ds_std': 0.0037,
        'HQNR_std': 0.0073,
    },

    'PanNet': {
        'Dl': 0.0169,
        'Ds': 0.0261,
        'HQNR': 0.9575,
        'Dl_std': 0.0042,
        'Ds_std': 0.0092,
        'HQNR_std': 0.0125,
    },

    'FusionMamba': {
        'Dl': 0.0166,
        'Ds': 0.0175,
        'HQNR': 0.9662,
        'Dl_std': 0.0043,
        'Ds_std': 0.0072,
        'HQNR_std': 0.0101,
    }
}


# ============================================================
# 2. Teacher → Student correspondence
# ============================================================

pairs = [
    ('RWKVFusion', 'RWKVFusion'),
    ('ARConv', 'ARConv'),
    ('PanNet', 'PanNet'),
    ('FusionMamba', 'FusionMamba'),
]


# ============================================================
# 3. Marker mapping
#    Keep the original visual semantics.
# ============================================================

marker_map = {
    'FusionMamba': 'o',
    'ARConv': '^',
    'RWKVFusion': 'D',
    'PanNet': 's'
}


# ============================================================
# 4. Color palette
#    Muted / academic / publication-friendly
# ============================================================

teacher_color = '#7896B8'        # macaron dusty blue
student_color = '#E18478'        # macaron coral
path_color = '#7C8792'           # cool gray-blue
projection_color = '#A9AFB5'     # neutral projection gray
projection_fill = '#DEE1E3'
text_color = '#2F3439'
grid_color = '#ADB5BC'


# ============================================================
# 5. Figure
#    Compact layout, close to your original figure.
# ============================================================

fig = plt.figure(
    figsize=(10.0, 9.0),
    dpi=300,
    facecolor='#FCFCFB'
)

ax = fig.add_subplot(projection='3d')
ax.set_facecolor('#FCFCFB')


# ============================================================
# 6. Coordinate conversion
#
#    Original table:
#       Dl = 0.0157
#       Ds = 0.0357
#
#    Plot:
#       Dl × 100 → 1.57
#       Ds × 100 → 3.57
#
#    This matches:
#       ×10^-2
# ============================================================

def xyz(data):
    x = data['Dl'] * 100
    y = data['Ds'] * 100
    z = data['HQNR']
    return x, y, z


# Bottom plane
z0 = 0.935


# ============================================================
# 7. Draw gray projection elements
#
#    Every teacher and student point:
#       point → vertical gray projection → bottom plane
#
#    This is the gray projection element you requested.
# ============================================================

for teacher_name, student_name in pairs:

    tx, ty, tz = xyz(teachers[teacher_name])
    sx, sy, sz = xyz(students[student_name])

    # ----------------------------
    # Teacher projection
    # ----------------------------
    ax.plot(
        [tx, tx],
        [ty, ty],
        [z0, tz],
        color=projection_color,
        linestyle='-',
        linewidth=1.15,
        alpha=0.72
    )

    # Teacher bottom projection point
    ax.scatter(
        tx, ty, z0,
        marker=marker_map[teacher_name],
        s=38,
        facecolor=projection_fill,
        edgecolor=projection_color,
        linewidth=0.75,
        alpha=0.72,
        depthshade=False
    )

    # ----------------------------
    # Student projection
    # ----------------------------
    ax.plot(
        [sx, sx],
        [sy, sy],
        [z0, sz],
        color=projection_color,
        linestyle='-',
        linewidth=1.15,
        alpha=0.72
    )

    # Student bottom projection point
    ax.scatter(
        sx, sy, z0,
        marker=marker_map[student_name],
        s=38,
        facecolor=projection_fill,
        edgecolor=projection_color,
        linewidth=0.75,
        alpha=0.72,
        depthshade=False
    )


# ============================================================
# 8. Draw distillation paths + data points
# ============================================================

for teacher_name, student_name in pairs:

    tx, ty, tz = xyz(teachers[teacher_name])
    sx, sy, sz = xyz(students[student_name])

    marker = marker_map[teacher_name]

    # --------------------------------------------------------
    # Distillation path
    # Teacher → Student
    # --------------------------------------------------------
    ax.plot(
        [tx, sx],
        [ty, sy],
        [tz, sz],
        color=path_color,
        linestyle='--',
        linewidth=1.15,
        alpha=0.88
    )

    # --------------------------------------------------------
    # Teacher point
    # --------------------------------------------------------
    point_size = 165
    point_edge = 0.90

    ax.scatter(
        tx, ty, tz,
        marker=marker,
        s=point_size,
        color=teacher_color,
        edgecolor='white',
        linewidth=point_edge,
        depthshade=False
    )

    # --------------------------------------------------------
    # Student point
    # --------------------------------------------------------
    ax.scatter(
        sx, sy, sz,
        marker=marker,
        s=point_size,
        color=student_color,
        edgecolor='white',
        linewidth=point_edge,
        depthshade=False
    )


# ============================================================
# 9. Performance labels
#
#    Automatically calculate:
#
#    HQNR improvement =
#    (Student HQNR - Teacher HQNR)
#    / Teacher HQNR × 1000‰
#
#    DO NOT manually enter percentages.
# ============================================================

label_offsets = {

    # RWKVFusion
    'RWKVFusion': (
        0.000,     # x offset
        0.000,     # y offset
        0.0020     # z offset
    ),

    # ARConv
    'ARConv': (
        0.000,
        0.000,
        0.0045
    ),

    # PanNet
    'PanNet': (
        0.000,
        0.000,
        0.0015
    ),

    # FusionMamba
    'FusionMamba': (
        0.000,
        0.000,
        0.0020
    )
}

label_alignments = {
    'RWKVFusion': 'center',
    'ARConv': 'right',
    'PanNet': 'center',
    'FusionMamba': 'left',
}


for teacher_name, student_name in pairs:

    teacher_hqnr = teachers[teacher_name]['HQNR']
    student_hqnr = students[student_name]['HQNR']

    # Actual relative improvement
    improvement = (
        (student_hqnr - teacher_hqnr)
        / teacher_hqnr
        * 1000
    )

    sx, sy, sz = xyz(students[student_name])

    ox, oy, oz = label_offsets[teacher_name]

    ax.text(
        sx + ox,
        sy + oy,
        sz + oz,
        f'↑{improvement:.1f}‰',
        fontsize=10.0,
        fontweight='bold',
        color=text_color,
        ha=label_alignments[teacher_name],
        va='bottom'
    )

# ============================================================
# 10. Axis labels
#
#     Keep the original label formatting.
# ============================================================

ax.set_xlabel(
    r'Spatial Distortion $D_s$ ($\times 10^{-2}$)',
    labelpad=13,
    fontsize=13.5,
    fontweight='bold'
)

ax.set_ylabel(
    r'Spectral Distortion $D_\lambda$ ($\times 10^{-2}$)',
    labelpad=13,
    fontsize=13.5,
    fontweight='bold'
)

ax.set_zlabel(
    'HQNR (Performance)',
    labelpad=11,
    fontsize=13.5,
    fontweight='bold'
)


# ============================================================
# 11. Axis limits
#     Keep the compact range of the original figure.
# ============================================================

ax.set_xlim(1.3, 2.1)
ax.set_ylim(1.6, 4.8)
ax.set_zlim(0.935, 0.975)


# ============================================================
# 12. Ticks
# ============================================================

ax.set_xticks(
    np.arange(1.3, 2.11, 0.1)
)

ax.set_yticks(
    np.arange(1.6, 4.81, 0.4)
)

ax.set_zticks(
    np.arange(0.935, 0.976, 0.005)
)

ax.tick_params(
    axis='both',
    which='major',
    labelsize=10.5
)

ax.tick_params(
    axis='z',
    which='major',
    labelsize=10.5
)


# ============================================================
# 13. 3D viewing angle
#     Same basic perspective as your original.
# ============================================================

ax.view_init(
    elev=25,
    azim=-210
)

# ============================================================
# 14. Compact 3D box aspect ratio
# ============================================================

ax.set_box_aspect(
    (1.0, 1.08, 0.82)
)


# ============================================================
# 15. Title
#     Keep original two-line format.
# ============================================================

ax.set_title(
    "3D Spectral-Spatial Distortion & HQNR Performance\n"
    "ZSVD Model Distillation Analysis",
    fontsize=17.0,
    fontweight='bold',
    y=1,
    pad=4
)


# ============================================================
# 16. Clean 3D background
# ============================================================

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


# ============================================================
# 17. Grid
# ============================================================

for axis in [
    ax.xaxis,
    ax.yaxis,
    ax.zaxis
]:

    axis._axinfo["grid"]["linestyle"] = '-'
    axis._axinfo["grid"]["linewidth"] = 0.8
    axis._axinfo["grid"]["color"] = grid_color
    axis._axinfo["grid"]["alpha"] = 0.88


# ============================================================
# 18. Legend
#
#     Keep the original compact 2-row × 5-column structure.
# ============================================================

legend_elements = [

    # Teacher
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='FusionMamba',
        markerfacecolor=teacher_color,
        markeredgecolor='white',
        markersize=8
    ),

    Line2D(
        [0], [0],
        marker='^',
        color='w',
        label='ARConv',
        markerfacecolor=teacher_color,
        markeredgecolor='white',
        markersize=8
    ),

    Line2D(
        [0], [0],
        marker='D',
        color='w',
        label='RWKVFusion',
        markerfacecolor=teacher_color,
        markeredgecolor='white',
        markersize=8
    ),

    Line2D(
        [0], [0],
        marker='s',
        color='w',
        label='PanNet',
        markerfacecolor=teacher_color,
        markeredgecolor='white',
        markersize=8
    ),

    Line2D(
        [0], [0],
        color=path_color,
        linestyle='--',
        label='Distillation Path',
        linewidth=1.4
    ),

    # Student
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label='FusionMamba*',
        markerfacecolor=student_color,
        markeredgecolor='white',
        markersize=8
    ),

    Line2D(
        [0], [0],
        marker='^',
        color='w',
        label='ARConv*',
        markerfacecolor=student_color,
        markeredgecolor='white',
        markersize=8
    ),

    Line2D(
        [0], [0],
        marker='D',
        color='w',
        label='RWKVFusion*',
        markerfacecolor=student_color,
        markeredgecolor='white',
        markersize=8
    ),

    Line2D(
        [0], [0],
        marker='s',
        color='w',
        label='PanNet*',
        markerfacecolor=student_color,
        markeredgecolor='white',
        markersize=8
    ),

    Line2D(
        [0], [0],
        color=projection_color,
        linestyle='-',
        label='Projection Line',
        linewidth=1.4
    )
]


ax.legend(
    # Matplotlib fills a multi-column legend by columns. Interleave
    # teacher/student handles so the rendered rows match the reference.
    handles=[
        legend_elements[index]
        for index in [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    ],
    loc='lower center',
    bbox_to_anchor=(0.5, -0.14),
    ncol=5,
    frameon=True,
    fancybox=False,
    framealpha=0.92,
    fontsize=10.0,
    handletextpad=0.40,
    columnspacing=0.90,
    borderpad=0.50
)


# ============================================================
# 19. Compact layout
# ============================================================

plt.subplots_adjust(
    left=0.02,
    right=0.97,
    top=0.91,
    bottom=0.15
)


# ============================================================
# 20. Save
# ============================================================


# ============================================================
# 21. Print calculated labels for verification
# ============================================================

print('\n===== HQNR Improvement =====')

for teacher_name, student_name in pairs:

    teacher_hqnr = teachers[teacher_name]['HQNR']
    student_hqnr = students[student_name]['HQNR']

    improvement = (
        (student_hqnr - teacher_hqnr)
        / teacher_hqnr
        * 1000
    )

    print(
        f'{teacher_name:12s}: '
        f'{teacher_hqnr:.4f} → {student_hqnr:.4f} '
        f'(↑{improvement:.1f}‰)'
    )
fig.savefig(
    '/media/zouhe/Elements/HeZou/zup/zsvd_distillation_3d.png',
    format='png',
    dpi=300,
    facecolor=fig.get_facecolor()
)
plt.close(fig)
