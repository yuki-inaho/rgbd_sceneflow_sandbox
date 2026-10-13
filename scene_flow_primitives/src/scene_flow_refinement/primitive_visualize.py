"""Standalone scientific plots and grayscale contact sheets from numeric results.

Every chart is its own matplotlib figure. Contact sheets are composed from saved
raster images with Pillow, not from a multiple-axes plotting layout.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .primitives import render_frame,make_scene

_FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def font(size: int):
    try:return ImageFont.truetype(_FONT,size)
    except OSError:return ImageFont.load_default(size=size)


def gray_image(a: np.ndarray, lower: float = 0., upper: float = 1.) -> Image.Image:
    if upper<=lower:raise ValueError('invalid display range')
    finite=np.isfinite(a)
    scaled=np.clip(np.nan_to_num((a-lower)/(upper-lower),nan=0.),0,1)
    pixels=np.rint(scaled*255).astype(np.uint8)
    yy,xx=np.indices(pixels.shape)
    pixels[~finite]=np.where((xx[~finite]+yy[~finite])%6<3,128,224)
    return Image.fromarray(pixels).convert('RGB')


def sheet(panels: list[tuple[str,Image.Image]], path: Path, columns: int = 2,
          size: tuple[int,int] = (480,360), footer: str = '') -> None:
    w,h=size;pad=20;header=46;bottom=60 if footer else 10
    rows=(len(panels)+columns-1)//columns
    canvas=Image.new('RGB',(columns*(w+pad)+pad,rows*(h+header+pad)+pad+bottom),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(label,img) in enumerate(panels):
        x=pad+(i%columns)*(w+pad);y=pad+(i//columns)*(h+header+pad)
        draw.text((x,y+6),label,font=font(22),fill='black')
        resample=Image.Resampling.LANCZOS if img.width>=w else Image.Resampling.NEAREST
        canvas.paste(img.resize(size,resample),(x,y+header))
    if footer:draw.text((pad,canvas.height-bottom+8),footer,font=font(18),fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def render_gallery(out: Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    panels=[]
    names={'gallery':'基本配置','occlusion':'遮蔽の多い配置','curved':'曲面中心の配置'}
    for layout,name in names.items():
        for time in (0,1):
            f=render_frame(make_scene(layout),480,640,float(time),supersample=2)
            im=gray_image(f['gray']);im.save(out/f'{layout}_gray{time}_hires.png')
            panels.append((f'{name}・時刻 {time}',im))
    sheet(panels,out/'render_gallery.png',size=(640,480),
          footer='C++解析的描画：640×480。精度評価の入力解像度は160×120です。')
    # A dedicated textured / texture-free rotation scene makes the ambiguity visible.
    rotation=[]
    for contrast,label in ((1.,'テクスチャあり'),(0.,'テクスチャなし')):
        for t in (0,1):
            f=render_frame(make_scene('rotation'),360,480,float(t),contrast=contrast,supersample=2)
            rotation.append((f'{label}・時刻 {t}',gray_image(f['gray'])))
    sheet(rotation,out/'rotation_observability.png',footer='球は同じ中心の周りに回転します。深度と輪郭は同じでも、表面点は動いています。')


def case_figures(directory: Path) -> None:
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    if Path(_FONT).exists():
        font_manager.fontManager.addfont(_FONT)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=_FONT).get_name()
    data=dict(np.load(directory/'data.npz'))
    methods={name:dict(np.load(directory/f'{name}.npz')) for name in ('raw','refined') if (directory/f'{name}.npz').exists()}
    if 'refined' not in methods:return
    r=methods['refined'];figdir=directory/'figures';figdir.mkdir(exist_ok=True)
    for t in (0,1):
        gray_image(data[f'gray{t}']).convert('L').save(figdir/f'gray{t}_input.png')
        observed=data[f'depth{t}'];encoded=np.where(np.isfinite(observed),np.clip(np.rint(observed*10000),1,65535),0).astype(np.uint16)
        Image.fromarray(encoded).save(figdir/f'depth{t}_0p1mm.png')
        for tag,z in [('gt',data[f'gt_depth{t}']),('observed',observed),('refined',r[f'depth{t}'])]:
            gray_image(z,.75,2.15).save(figdir/f'depth{t}_{tag}_display.png')
    sheet([(f'灰度入力・時刻 {t}',gray_image(data[f'gray{t}'])) for t in (0,1)]+
          [(f'観測深度・時刻 {t}',gray_image(data[f'depth{t}'],.75,2.15)) for t in (0,1)],figdir/'inputs.png',
          footer='実入力160×120を拡大表示。深度：黒0.75 m → 白2.15 m。縞模様は欠測です。')
    gt=data['gt_scene_flow'];mask=data['gt_visible']
    mag=lambda v:np.linalg.norm(v,axis=-1)*1000
    labels={'raw':'直接逆投影','refined':'補正版 v0.3'}
    panels=[('正解変位の大きさ',gray_image(mag(gt),0,50)),
            ('v0.2 変位の大きさ',gray_image(mag(r['scene_flow']),0,50))]
    for name in ('raw','refined'):
        if name in methods:
            error=mag(methods[name]['scene_flow']-gt)
            gray_image(error,0,20).save(figdir/f'error_{name}.png')
    panels.extend([('直接逆投影の誤差',gray_image(mag(methods['raw']['scene_flow']-gt),0,20)),
                   ('v0.2 の誤差',gray_image(mag(r['scene_flow']-gt),0,20)),
                   ('正解：可視=白／遮蔽等=黒',gray_image(mask.astype(float))),
                   ('推定：対応支持あり=白',gray_image(r['support']))])
    sheet(panels,figdir/'comparison.png',footer='変位：0–50 mm、誤差：0–20 mm（白は上限以上）。可視性と推定支持は別の量です。')
    sheet([(labels[name]+'：変位誤差',gray_image(mag(methods[name]['scene_flow']-gt),0,20)) for name in ('raw','refined') if name in methods],
          figdir/'error_comparison.png',columns=3,size=(400,300),footer='同一表示尺度：黒0 mm → 白20 mm以上。縞模様は数値未定義です。')
    # One profile chart, with shared units and no object-specific fitting.
    objmask=data['gt_labels0']==2
    row=int(np.argmax(objmask.sum(axis=1)))
    xs=np.flatnonzero(objmask[row]);lo=max(0,int(xs.min())-4);hi=min(data['depth0'].shape[1],int(xs.max())+5)
    fig=plt.figure(figsize=(8.5,4.4));ax=fig.add_subplot(111)
    ax.plot(np.arange(lo,hi),data['gt_depth0'][row,lo:hi]*1000,label='正解')
    ax.plot(np.arange(lo,hi),data['depth0'][row,lo:hi]*1000,label='観測深度',linewidth=.8)
    for name in ('refined',):
        if name in methods:ax.plot(np.arange(lo,hi),methods[name]['depth0'][row,lo:hi]*1000,label=labels[name])
    ax.set_xlabel('画素 x');ax.set_ylabel('光軸方向深度 [mm]');ax.set_title(f'曲面の深度断面（y={row}）')
    ax.legend();ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(figdir/'depth_profile.png',dpi=160);plt.close(fig)
    # Error distribution on one fixed evaluation mask.
    finite=np.isfinite(methods['raw']['scene_flow']).all(-1);common=mask&finite
    fig=plt.figure(figsize=(8.5,4.4));ax=fig.add_subplot(111)
    for name,result in methods.items():
        values=np.sort(mag(result['scene_flow']-gt)[common]);values=values[np.isfinite(values)]
        ax.plot(values,np.arange(1,len(values)+1)/len(values),label=labels[name])
    ax.set_xlim(0,30);ax.set_ylim(0,1);ax.set_xlabel('3次元端点誤差 [mm]');ax.set_ylabel('累積割合')
    ax.set_title('同じ可視画素集合での誤差分布');ax.legend();ax.grid(True,alpha=.25)
    fig.tight_layout();fig.savefig(figdir/'error_cdf.png',dpi=160);plt.close(fig)
    # Each 3-D vector chart is a standalone figure, with fixed axes and scale.
    h,w=data['depth0'].shape;yy,xx=np.mgrid[:h,:w];fx,fy,cx,cy=data['intrinsics']
    rays=np.stack(((xx-cx)/fx,(yy-cy)/fy,np.ones_like(xx)),axis=-1)
    take=(xx%4==0)&(yy%4==0)&data['gt_moving']
    cloud=(xx%2==0)&(yy%2==0)&(data['gt_labels0']>=2)
    for name,scene,z in [('gt',gt,data['gt_depth0']),('refined',r['scene_flow'],r['depth0'])]:
        xyz=z[...,None]*rays;points=xyz[take];vectors=scene[take]*5
        fig=plt.figure(figsize=(8.2,6.2));ax=fig.add_subplot(111,projection='3d')
        pc=xyz[cloud];ax.scatter(pc[:,0],pc[:,2],-pc[:,1],c=data['gray0'][cloud],cmap='gray',vmin=0,vmax=1,s=4,alpha=.55)
        ax.quiver(points[:,0],points[:,2],-points[:,1],vectors[:,0],vectors[:,2],-vectors[:,1],arrow_length_ratio=.25,linewidth=.65)
        ax.set_xlim(-.55,.55);ax.set_ylim(.75,1.95);ax.set_zlim(-.30,.40)
        ax.set_xlabel('X [m]');ax.set_ylabel('Z [m]');ax.set_zlabel('−Y [m]');ax.view_init(elev=24,azim=-65)
        ax.set_title(('正解' if name=='gt' else labels[name])+'：3次元変位（矢印5倍）')
        fig.tight_layout();fig.savefig(figdir/f'vectors3d_{name}.png',dpi=160);plt.close(fig)
    # Machine-readable PNG conventions avoid confusing visualization with depth data.
    (figdir/'image_formats.json').write_text(json.dumps({'gray_png':'uint8; float input is preserved in data.npz',
          'depth_0p1mm_png':{'dtype':'uint16','metres_per_unit':.0001,'invalid':0},
          'depth_display_range_m':[.75,2.15],'scene_magnitude_range_mm':[0,50],'error_display_range_mm':[0,20],
          'vector_scale':5,'quiver_includes_occluded_source_points':True},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path('results/benchmark'))
    p.add_argument('--gallery',type=Path,default=Path('results/renderings'));p.add_argument('--all',action='store_true')
    a=p.parse_args();render_gallery(a.gallery)
    if a.all:
        for path in sorted(a.results.glob('*/data.npz')):case_figures(path.parent)
    else:
        for name in ('gallery_dis_seed11','occlusion_dis_seed11','curved_dis_seed11'):
            if (a.results/name/'refined.npz').exists():case_figures(a.results/name)

if __name__=='__main__':main()
