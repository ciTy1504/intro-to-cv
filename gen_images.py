import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

IMG_DIR = r'c:\Users\admin\Desktop\intro-to-cv\Tex\images'

# CBAM diagram
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')
rboxes = [
    (0.2, 2.8, 1.6, 1.2, '#2c3e50', 'Frame 1\n(RGB)'),
    (0.2, 1.0, 1.6, 1.2, '#2c3e50', 'Frame 2\n(RGB)'),
    (2.2, 2.8, 1.8, 1.2, '#2980b9', 'Feature\nEncoder'),
    (2.2, 1.0, 1.8, 1.2, '#2980b9', 'Feature\nEncoder'),
    (4.4, 1.5, 2.2, 1.6, '#8e44ad', '4D Correlation\nVolume\n(H x W x H x W)'),
    (7.0, 1.5, 2.0, 1.6, '#e67e22', 'GRU\nIterative\nRefinement'),
    (9.4, 1.8, 2.2, 1.2, '#27ae60', 'Optical\nFlow\n(u,v)'),
]
for x, y, w, h, c, lbl in rboxes:
    rect = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=c,alpha=0.88,edgecolor='white',lw=1.8)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, lbl, ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
rarrows = [(1.8,3.4,2.2,3.4),(1.8,1.6,2.2,1.6),(4.0,3.4,4.4,2.7),(4.0,1.6,4.4,2.3),(6.6,2.3,7.0,2.3),(9.0,2.3,9.4,2.3)]
for (x1,y1,x2,y2) in rarrows:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#ddd', lw=1.8))
ax.text(6.0, 4.5, 'RAFT Architecture', ha='center', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'raft_architecture.png'), dpi=120, bbox_inches='tight')
plt.close()
print('raft_architecture.png OK')

# CBAM
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')
boxes_ch = [
    (0.2, 1.8, 1.6, 1.4, '#3498db', 'Input F\nH x W x C'),
    (2.2, 2.5, 1.4, 0.7, '#e67e22', 'Avg Pool'),
    (2.2, 1.5, 1.4, 0.7, '#e67e22', 'Max Pool'),
    (4.0, 1.8, 1.4, 1.4, '#27ae60', 'Shared\nMLP'),
    (5.8, 1.8, 1.4, 1.4, '#8e44ad', 'Sigmoid\nMc'),
    (7.6, 1.8, 1.6, 1.4, '#c0392b', 'Channel\nScaled F'),
    (9.4, 2.5, 1.4, 0.7, '#16a085', 'AvgPool\nchannel'),
    (9.4, 1.5, 1.4, 0.7, '#16a085', 'MaxPool\nchannel'),
]
for x, y, w, h, c, lbl in boxes_ch:
    rect = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=c,alpha=0.85,edgecolor='white',lw=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, lbl, ha='center', va='center', fontsize=8, color='white', fontweight='bold')
arrows = [(1.8,2.5,2.2,2.85),(1.8,2.2,2.2,1.85),(3.6,2.85,4.0,2.5),(3.6,1.85,4.0,2.1),(5.4,2.5,5.8,2.5),(7.2,2.5,7.6,2.5),(9.2,2.85,9.4,2.85),(9.2,1.85,9.4,1.85)]
for (x1,y1,x2,y2) in arrows:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
ax.text(5.5, 4.5, 'CBAM: Channel + Spatial Attention', ha='center', fontsize=12, fontweight='bold')
ax.text(4.5, 0.5, 'Channel Attention', ha='center', fontsize=9, color='#555')
ax.text(10.0, 0.5, 'Spatial', ha='center', fontsize=9, color='#555')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'cbam_diagram.png'), dpi=120, bbox_inches='tight')
plt.close()
print('cbam_diagram.png OK')

# Loop Closing
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, title, closed in zip(axes, ['Before Loop Closing (drift)', 'After Loop Closing (consistent)'], [False, True]):
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, fontsize=10, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    t = np.linspace(0, 2*np.pi*1.8, 200)
    if closed:
        x = 2.2 * np.cos(t)
        y = 2.0 * np.sin(t)
    else:
        x = 2.2 * np.cos(t) + np.linspace(0, 0.8, 200)
        y = 2.0 * np.sin(t) + np.linspace(0, 0.5, 200)
    colors_traj = plt.cm.coolwarm(np.linspace(0, 1, 200))
    for i in range(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], color=colors_traj[i], linewidth=2.5, alpha=0.9)
    ax.scatter([x[0]], [y[0]], c='lime', s=80, zorder=5, label='Start')
    ax.scatter([x[-1]], [y[-1]], c='red', s=80, zorder=5, label='End')
    if not closed:
        ax.annotate('', xy=(x[-1], y[-1]-0.4), xytext=(x[0], y[0]-0.4),
                    arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.5))
        ax.text((x[0]+x[-1])/2, min(y[0],y[-1])-0.7, 'drift!', color='yellow', ha='center', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'loop_closing.png'), dpi=120, bbox_inches='tight')
plt.close()
print('loop_closing.png OK')

# SLAM overview
fig, ax = plt.subplots(figsize=(11, 4))
ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis('off')
slam_boxes = [
    (0.3, 1.8, 1.6, 1.4, '#2c3e50', 'Camera\n(Video)'),
    (2.3, 3.0, 2.0, 1.2, '#3498db', 'Tracking\n(Pose/frame)'),
    (2.3, 1.5, 2.0, 1.2, '#e67e22', 'Local\nMapping'),
    (5.0, 1.5, 2.0, 1.2, '#27ae60', 'Loop\nClosing'),
    (7.6, 1.5, 2.0, 1.2, '#9b59b6', 'Global\nOptimization'),
    (9.8, 1.8, 1.4, 1.4, '#c0392b', 'Consistent\n3D Map'),
]
for x,y,w,h,c,lbl in slam_boxes:
    rect = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=c,alpha=0.88,edgecolor='white',lw=2)
    ax.add_patch(rect)
    ax.text(x+w/2,y+h/2,lbl,ha='center',va='center',fontsize=8.5,color='white',fontweight='bold')
sarrows = [(1.9,2.5,2.3,3.6),(1.9,2.5,2.3,2.1),(4.3,2.1,5.0,2.1),(7.0,2.1,7.6,2.1),(9.6,2.1,9.8,2.1)]
for (x1,y1,x2,y2) in sarrows:
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.8))
ax.text(5.5,4.5,'Visual SLAM System Overview',ha='center',fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'slam_overview.png'), dpi=120, bbox_inches='tight')
plt.close()
print('slam_overview.png OK')

# depth applications
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
titles = ['Autonomous Driving', 'AR / VR', 'Robotics Grasping', 'Portrait Mode (Bokeh)']
np.random.seed(5)
for ax, title in zip(axes.flat, titles):
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, fontsize=10, color='white', fontweight='bold')
    ax.axis('off')
    img = np.random.rand(60, 90, 3) * 0.3
    depth = np.zeros((60, 90))
    for _ in range(5):
        r, c = np.random.randint(5,55), np.random.randint(5,85)
        h2, w2 = np.random.randint(8,20), np.random.randint(10,25)
        d = np.random.rand()
        depth[r:r+h2, c:c+w2] = d
        col = plt.cm.Set1(np.random.rand())[:3]
        img[r:r+h2, c:c+w2] = col
    depth_colored = plt.cm.plasma(depth)[:,:,:3]
    ax.imshow(np.hstack([img, depth_colored]))
plt.suptitle('Applications of Depth Estimation', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'depth_applications.png'), dpi=120, bbox_inches='tight')
plt.close()
print('depth_applications.png OK')

print('All images generated!')
