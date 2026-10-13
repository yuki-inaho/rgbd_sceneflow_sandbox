"""Individual Matplotlib figures and a Pillow-composed comparison sheet."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image,ImageDraw,ImageFont
from .metrics import rays,project_flow


def configure_fonts() -> None:
    candidates=[Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update({"font.size":10,"axes.unicode_minus":False,"savefig.dpi":160})


def map_figure(array: np.ndarray, title: str, path: Path, *, minimum: float|None=None,
               maximum: float|None=None, unit: str|None=None) -> None:
    fig,ax=plt.subplots(figsize=(6.2,4.6),layout="constrained")
    image=ax.imshow(array,vmin=minimum,vmax=maximum,interpolation="nearest")
    ax.set_title(title);ax.set_xlabel("横画素座標");ax.set_ylabel("縦画素座標")
    if array.ndim==2:
        fig.colorbar(image,ax=ax,label=unit or "")
    fig.savefig(path);plt.close(fig)


def vector_figure(data:dict,scene:np.ndarray,title:str,path:Path) -> None:
    xyz=data["gt_depth0"][...,None]*rays(data["gt_depth0"].shape,data["intrinsics"])
    sampled=np.zeros(data["gt_depth0"].shape,dtype=bool);sampled[::6,::6]=True
    sampled &= data["gt_labels0"]>0
    sampled &= np.isfinite(scene).all(axis=-1)
    x=xyz[sampled];s=scene[sampled]
    fig=plt.figure(figsize=(6.4,5.0),layout="constrained");ax=fig.add_subplot(111,projection="3d")
    ax.scatter(x[:,0],x[:,1],x[:,2],s=3)
    ax.quiver(x[:,0],x[:,1],x[:,2],5*s[:,0],5*s[:,1],5*s[:,2],normalize=False,arrow_length_ratio=.25)
    ax.set(xlabel="X [m]",ylabel="Y [m]",zlabel="Z [m]",title=title+"\n矢印は変位の5倍、動物体のみ表示")
    ax.set_xlim(-.38,.44);ax.set_ylim(-.25,.25);ax.set_zlim(.75,1.2)
    ax.view_init(elev=24,azim=-63)
    fig.savefig(path);plt.close(fig)


def make_figures(data: dict, outputs: dict, evaluation_masks: dict, directory: Path, motion_bound: float = 0.05) -> None:
    configure_fonts();directory.mkdir(parents=True,exist_ok=True)
    joint=outputs["joint_tgv"];raw=outputs["raw"]["scene_flow"]
    for t in (0,1):
        rgb=data[f"rgb{t}"]
        Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(directory/f"rgb{t}.png")
        map_figure(rgb,f"入力RGB画像：時刻 {t}",directory/f"rgb{t}_labeled.png")
        for name,z in (("observed",data[f"depth{t}"]),("gt",data[f"gt_depth{t}"]),("refined",joint[f"depth{t}"])):
            label={"observed":"観測","gt":"正解","refined":"推定"}[name]
            map_figure(z,f"{label}深度：時刻 {t}",directory/f"depth{t}_{name}.png",minimum=.8,maximum=1.5,unit="深度 [m]")
    scenes={"gt":data["gt_scene_flow"],"raw":raw,"clipped":outputs["clipped"]["scene_flow"],
            "rematched":joint["initial_scene_flow"],"refined":joint["scene_flow"]}
    bound_label=f"{motion_bound*100:g} cm"
    labels={"gt":"正解","raw":"観測値の直接逆投影","clipped":f"直接逆投影＋{bound_label}制限",
            "rematched":"対応候補の初期選択","refined":"深度・変位の同時推定"}
    common=evaluation_masks["common_visible"]
    for name,scene in scenes.items():
        map_figure(np.linalg.norm(scene,axis=-1)*1000,f"{labels[name]}：3次元変位の大きさ",
                   directory/f"motion_{name}.png",minimum=0,maximum=50,unit="変位 [mm]（表示上限50）")
        if name!="gt":
            error=np.linalg.norm(scene-data["gt_scene_flow"],axis=-1)*1000
            map_figure(np.where(common,error,np.nan),f"{labels[name]}：3次元端点誤差",
                       directory/f"error_{name}.png",minimum=0,maximum=20,unit="誤差 [mm]（表示上限20）")
    for name,flow in (("gt",data["gt_flow"]),("observed",data["flow_observed"]),("refined",joint["flow"])):
        label={"gt":"正解","observed":"雑音・誤対応を含む入力","refined":"推定変位の再投影"}[name]
        map_figure(np.linalg.norm(flow,axis=-1),f"{label}：画像上の移動量",directory/f"flow_{name}.png",
                   minimum=0,maximum=8,unit="移動量 [画素]（表示上限8）")
        fig,ax=plt.subplots(figsize=(6.2,4.6),layout="constrained")
        y,x=np.mgrid[:flow.shape[0]:6,:flow.shape[1]:6];f=flow[::6,::6]
        ax.imshow(data["rgb0"]);ax.quiver(x,y,f[...,0],f[...,1],angles="xy",scale_units="xy",scale=1)
        ax.set_title(label+"：オプティカルフロー");ax.set_xlabel("横画素座標");ax.set_ylabel("縦画素座標")
        fig.savefig(directory/f"flow_vectors_{name}.png");plt.close(fig)
    for name,scene in (("gt",data["gt_scene_flow"]),("refined",joint["scene_flow"])):
        vector_figure(data,scene,"正解のシーンフロー" if name=="gt" else "推定したシーンフロー",directory/f"vectors3d_{name}.png")
    map_figure(joint["support"],"観測から対応候補が得られた画素（1）",directory/"support.png",minimum=0,maximum=1)
    map_figure(data["injected_outlier"].astype(float),"人工的に混入した誤対応（評価専用）",directory/"injected_outliers.png",minimum=0,maximum=1)
    map_figure(data["gt_visible"].astype(float),"正解の相互可視領域（評価専用）",directory/"gt_visibility.png",minimum=0,maximum=1)
    map_figure(joint["geometry_residual"]*1000,"候補対応と推定変位の幾何残差",directory/"geometry_residual.png",minimum=0,maximum=2,unit="残差 [mm]（表示上限2）")
    history=joint["history"]
    fig,ax=plt.subplots(figsize=(6.4,4.0),layout="constrained")
    step=history[:,0]*(history[:,1].max()+1)+history[:,1]
    ax.plot(step,history[:,2],label="記録時点の目的関数")
    ax.plot(step,history[:,3],label="採用する最良値")
    ax.set(xlabel="累積反復番号",ylabel="無次元の目的関数値",title="目的関数と採用値の推移")
    ax.legend();fig.savefig(directory/"objective.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,4.2),layout="constrained")
    for method,label in (("raw","直接逆投影"),("clipped",bound_label+"へ制限"),("joint_tv","同時推定＋TV"),("joint_tgv","同時推定＋二階正則化")):
        if method not in outputs:continue
        error=np.linalg.norm(outputs[method]["scene_flow"]-data["gt_scene_flow"],axis=-1)[common]*1000
        error=np.sort(error[np.isfinite(error)])
        ax.plot(error,np.arange(1,len(error)+1)/len(error),label=label)
    ax.set(xlabel="3次元端点誤差 [mm]",ylabel="累積比率",xlim=(0,25),ylim=(0,1),title="同じ可視画素集合での誤差分布")
    ax.legend();fig.savefig(directory/"error_cdf.png");plt.close(fig)
    row=data["depth0"].shape[0]//2
    fig,ax=plt.subplots(figsize=(6.4,4),layout="constrained")
    for label,z in (("正解",data["gt_scene_flow"]),(bound_label+"へ制限",outputs["clipped"]["scene_flow"]),("同時推定",joint["scene_flow"])):
        ax.plot(z[row,:,0]*1000,label=label)
    ax.set(xlabel="横画素座標",ylabel="X方向変位 [mm]",title=f"物体境界を横断する変位断面：行 {row}")
    ax.legend();fig.savefig(directory/"motion_profile.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,4),layout="constrained")
    for label,z in (("観測深度",data["depth0"]),("補正後深度",joint["depth0"])):
        ax.plot((z[row]-data["gt_depth0"][row])*1000,label=label)
    ax.set(xlabel="横画素座標",ylabel="正解深度からの差 [mm]",title=f"両時刻を補正：時刻0の深度誤差断面（行 {row}）")
    ax.legend();fig.savefig(directory/"depth_error_profile.png");plt.close(fig)
    # Compose existing standalone figures; every analytical plot above has its own axes.
    tiles=["rgb0_labeled.png","rgb1_labeled.png","motion_gt.png","motion_refined.png","error_raw.png","error_refined.png"]
    compose(tiles,directory,directory/"comparison.png",columns=2)


def compose(names:list[str],directory:Path,out:Path,columns:int=2)->None:
    images=[Image.open(directory/name).convert("RGB") for name in names]
    width=max(im.width for im in images);height=max(im.height for im in images)
    canvas=Image.new("RGB",(columns*width,((len(images)+columns-1)//columns)*height),(255,255,255))
    for index,image in enumerate(images):canvas.paste(image,((index%columns)*width,(index//columns)*height))
    canvas.save(out)
