"""Generate the Japanese report's numbers and figures from saved experiment arrays.

No inference or parameter selection is performed here. JSON and CSV values are
regenerated from arrays where a unit/constraint audit is necessary. Run after the
benchmark has completed, not concurrently with its output writes.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scene_flow_refinement import Config
from scene_flow_refinement.metrics import project_flow

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'data'
DOC = ROOT.parent / 'docs'
FIG = DOC / 'figures'
ORDER = ['v02','standard','no_rigid','quadratic_geometry','no_rigid_quadratic',
         'no_identity','full_search','cold_start','unfused','long_solver','quick','tv',
         'affine_photo','no_bound','no_edges','no_reciprocal','weak_rigid','balanced4','exact_kernel_only']
LABEL = dict(zip(ORDER, ['第0.2版','標準版','剛体補助項なし','幾何を二乗に戻す','剛体なし・幾何二乗',
    '恒等候補の予約なし','全探索','双対を毎回初期化','計算融合なし','長反復','短反復','一次TV',
    '輝度利得補正あり','移動上限なし','正則化の境界停止なし','往復照合なし','弱い剛体補助項','ステップ比4','同値な計算核変更のみ']))
LABEL['correct_bound'] = '正しい5 cm上限'
LAYOUT = {'gallery':'基本','occlusion':'遮蔽','curved':'曲面'}
STRESS = {'coplanar_split':'同深度・独立運動','illumination':'照明差','weak_texture':'弱い模様',
          'missing_depth':r'深度欠測8\%','rotation_uniform':'均一球の回転','rotation_textured':'模様付き球の回転',
          'wrong_bound':'誤った2 cm上限','resolution320':'320×240'}

def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def number(value: object, digits: int=3) -> str:
    if value is None or not np.isfinite(float(value)):
        return r'\NA{}'
    return f'{float(value):.{digits}f}'

def average(rows: list[dict], key: str) -> tuple[float,float]:
    vals = [r['metrics'][key] for r in rows if r['metrics'].get(key) is not None]
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals)>1 else 0.

def mean_sd(rows: list[dict], key: str, digits: int=3) -> str:
    mean, sd = average(rows, key)
    return f'{mean:.{digits}f} $\\pm$ {sd:.{digits}f}'

def table(name: str, headers: list[str], rows: Iterable[list[str]], caption: str,
          spec: str|None=None, small: str=r'\small') -> None:
    spec = spec or ('l' + 'r'*(len(headers)-1))
    rows = list(rows)
    if len(rows) <= 12:
        lines = [r'\begin{table}[H]\centering',small,r'\setlength{\tabcolsep}{4pt}',
                 r'\caption{'+caption+'}',r'\begin{tabular}{'+spec+'}',r'\toprule',
                 ' & '.join(headers)+r'\\\midrule']
        lines += [' & '.join(row)+r'\\' for row in rows]
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        (DOC/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return
    lines = [r'\begingroup'+small, r'\setlength{\tabcolsep}{4pt}',
        r'\begin{longtable}{'+spec+'}',r'\caption{'+caption+r'}\\',r'\toprule',
        ' & '.join(headers)+r'\\\midrule\endfirsthead',r'\toprule',
        ' & '.join(headers)+r'\\\midrule\endhead',r'\bottomrule\endfoot']
    lines += [' & '.join(row)+r'\\' for row in rows]
    lines += [r'\end{longtable}',r'\endgroup']
    (DOC/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')

def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,keys)
        writer.writeheader(); writer.writerows(rows)

def read_records() -> list[dict]:
    records=[]
    for split in ['retest','heldout','ablation','stress']:
        for path in sorted((OUT/split).glob('*/*.json')):
            record=json.loads(path.read_text())
            if 'method' not in record:
                continue
            record['_path']=path
            records.append(record)
    return records

def audit(records: list[dict]) -> dict:
    """Check arrays, metric units, constraints, and exact implementation controls."""
    rows=[]; unit_updates=[]
    for record in records:
        path=record['_path']; cfg=record['config']; metrics=record['metrics']
        with np.load(path.with_suffix('.npz')) as result, np.load(path.parent/'data.npz') as data:
            magnitude=np.linalg.norm(result['scene_flow'],axis=-1)
            true_max=float(np.nanmax(magnitude)*1000)
            if not math.isclose(metrics.get('max_estimated_motion_mm',-1),true_max, rel_tol=1e-12, abs_tol=1e-12):
                unit_updates.append({'split':record['split'],'case':record['case'],'method':record['method'],
                                     'old_value':metrics.get('max_estimated_motion_mm'),'correct_mm':true_max})
                metrics['max_estimated_motion_mm']=true_max
                dump(path,{k:v for k,v in record.items() if k!='_path'})
            depth_excess=0.
            for t in (0,1):
                original=data[f'depth{t}']; inferred=result[f'depth{t}']
                valid=np.isfinite(original)&(original>0)
                excess=np.abs(inferred[valid]-original[valid])-cfg['trust_sigma']*cfg['sigma_depth']
                depth_excess=max(depth_excess,float(np.max(excess,initial=0.)))
            predicted=project_flow(result['depth0'],result['scene_flow'],data['intrinsics'])
            finite=np.isfinite(predicted)&np.isfinite(result['flow'])
            reprojection=float(np.max(np.abs(predicted[finite]-result['flow'][finite]),initial=0.))
            motion_excess=float(max(0,np.nanmax(magnitude)-cfg['max_motion'])) if cfg['enforce_bound'] else None
            history=result['history']
            monotonic=float(np.max(np.diff(history[:,3]),initial=0.))
            rows.append({'split':record['split'],'case':record['case'],'method':record['method'],
                'bound_excess_m':motion_excess,'trust_excess_m':depth_excess,'reprojection_max_px':reprojection,
                'outer_best_energy_increase':monotonic,'adjoint_relative_error':metrics['adjoint_error']})
    control={}
    directory=OUT/'ablation/gallery_seed37'
    for first,second in [('standard','unfused'),('v02','exact_kernel_only')]:
        if (directory/f'{second}.npz').exists():
            with np.load(directory/f'{first}.npz') as a, np.load(directory/f'{second}.npz') as b:
                control[first+'_vs_'+second]={}
                for field in ['depth0','depth1','scene_flow','flow','match_flow','geometry_residual']:
                    valid=np.isfinite(a[field])&np.isfinite(b[field])
                    control[first+'_vs_'+second][field]=float(np.max(np.abs(a[field][valid]-b[field][valid]),initial=0.))
    report={'records':len(rows),'array_checks':rows,'implementation_controls':control,
            'unit_correction':'Legacy max_estimated_motion_mm was stored in metres. Recomputed from scene_flow in millimetres; EPE/timings unchanged.',
            'unit_correction_records':unit_updates}
    # Preserve the first correction audit on repeat generation.
    previous=OUT/'unit_corrections.json'
    if unit_updates:
        old=json.loads(previous.read_text()) if previous.exists() else []
        dump(previous,old+unit_updates)
    dump(OUT/'numerical_audit.json',report)
    assert max((r['bound_excess_m'] or 0) for r in rows)<1e-10
    assert max(r['trust_excess_m'] for r in rows)<1e-10
    assert max(r['reprojection_max_px'] for r in rows)<1e-9
    return report

def generate_tables(records: list[dict], checks: dict) -> dict:
    summary={}
    rows=[]
    for split in ['retest','heldout']:
        for method in ['v02','standard']:
            selected=[r for r in records if r['split']==split and r['method']==method]
            if len(selected)!=6:
                raise RuntimeError(f'Incomplete {split}/{method}: {len(selected)} of 6')
            summary[split+'_'+method]={'n':len(selected),**{key:{'mean':average(selected,key)[0],'sd':average(selected,key)[1]}
                for key in ['common_visible_epe_mm','moving_epe_mm','elapsed_seconds','depth0_rmse_mm','depth1_rmse_mm',
                            'static_gray_change_epe_mm','occluded_epe_mm','geometry_mean_mm']}}
            rows.append([{'retest':'再試験','heldout':'新規条件'}[split],LABEL[method],str(len(selected)),
                mean_sd(selected,'common_visible_epe_mm'),mean_sd(selected,'moving_epe_mm'),mean_sd(selected,'elapsed_seconds',2)])
    table('table_main.tex',['区分','手法','$n$','共通可視 [mm]','動体 [mm]','C++時間 [s]'],rows,
          '主比較。6入力を等重みで集計した平均と標本標準偏差。各入力は160×120。',spec='llrrrr')
    with (DOC/'table_main.tex').open('a',encoding='utf-8') as stream:
        stream.write(r'新規条件の配置別の内訳は次のとおりです。'+'\n')
    layout_rows=[]
    for layout in ['gallery','occlusion','curved']:
        for method in ['v02','standard']:
            rs=[r for r in records if r['split']=='heldout' and r['layout']==layout and r['method']==method]
            layout_rows.append([LAYOUT[layout],LABEL[method],number(average(rs,'common_visible_epe_mm')[0]),
                                number(average(rs,'moving_epe_mm')[0]),number(average(rs,'elapsed_seconds')[0],2)])
    table('table_layouts.tex',['配置','手法','共通可視 [mm]','動体 [mm]','時間 [s]'],layout_rows,
          '新規条件の配置別平均。各配置は寸法・姿勢・模様・運動を変えた2入力。',spec='llrrr')
    with (DOC/'table_main.tex').open('a',encoding='utf-8') as stream:
        stream.write(r'\input{table_layouts.tex}'+'\n')
    depth_rows=[]
    for split in ['retest','heldout']:
        for method in ['v02','standard']:
            selected=[r for r in records if r['split']==split and r['method']==method]
            depth_rows.append([{'retest':'再試験','heldout':'新規条件'}[split],LABEL[method]]+
                [number(average(selected,key)[0]) for key in ['depth0_rmse_mm','depth1_rmse_mm','geometry_mean_mm']])
    table('table_depth_tradeoff.tex',['区分','手法','時刻0深度RMSE','時刻1深度RMSE','幾何残差平均'],depth_rows,
          '深度と幾何整合性の変化。全てmm、6入力の平均。変位誤差とは別の指標です。',spec='llrrr')
    macros={'HeldOld':summary['heldout_v02']['common_visible_epe_mm']['mean'],
            'HeldNew':summary['heldout_standard']['common_visible_epe_mm']['mean'],
            'HeldTimeOld':summary['heldout_v02']['elapsed_seconds']['mean'],
            'HeldTimeNew':summary['heldout_standard']['elapsed_seconds']['mean']}
    (DOC/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+name+'}{'+f'{value:.3f}'+'}' for name,value in macros.items())+'\n',encoding='utf-8')
    ab={r['method']:r for r in records if r['split']=='ablation'}
    if len(ab)!=len(ORDER):
        raise RuntimeError(f'Incomplete ablation: {len(ab)}/{len(ORDER)}')
    table('table_ablation.tex',['条件','共通可視','動体','静止灰度変化','遮蔽','時間 [s]'],
          [[LABEL[name]]+[number(ab[name]['metrics'].get(key)) for key in ['common_visible_epe_mm','moving_epe_mm','static_gray_change_epe_mm','occluded_epe_mm','elapsed_seconds']] for name in ORDER],
          '19構成の同一入力比較。時間以外の単位はmm。基準は「標準版」です。',spec='lrrrrr',small=r'\small')
    fact=[]
    for h,r,name in [(0,0,'no_rigid_quadratic'),(1,0,'no_rigid'),(0,1,'quadratic_geometry'),(1,1,'standard')]:
        fact.append([str(h),str(r),number(ab[name]['metrics']['common_visible_epe_mm']),number(ab[name]['metrics']['moving_epe_mm']),number(ab[name]['metrics']['static_gray_change_epe_mm'])])
    for key in ['common_visible_epe_mm','moving_epe_mm','static_gray_change_epe_mm']:
        summary['interaction_'+key]=ab['standard']['metrics'][key]-ab['quadratic_geometry']['metrics'][key]-ab['no_rigid']['metrics'][key]+ab['no_rigid_quadratic']['metrics'][key]
    fact.append(['差の差','',number(summary['interaction_common_visible_epe_mm']),number(summary['interaction_moving_epe_mm']),number(summary['interaction_static_gray_change_epe_mm'])])
    table('table_factorial.tex',['Huber','剛体補助項','共通可視 [mm]','動体 [mm]','静止灰度変化 [mm]'],fact,
          '幾何Huber化と剛体補助項の2×2比較。差の差は相互作用の記述値です。',spec='llrrr')
    time_rows=[]
    for method in ['v02','exact_kernel_only','standard','full_search','unfused','cold_start','quick','long_solver']:
        m=ab[method]['metrics']; total=m['elapsed_seconds']; match=m['matching_seconds']; prior=m.get('prior_seconds',0.)
        time_rows.append([LABEL[method],number(match,2),number(prior,3),number(total-match-prior,2),number(total,2),str(int(m.get('inner_iterations',1800)))])
    table('table_runtime.tex',['条件','対応探索','補助項','残り','合計 [s]','内側反復'],time_rows,
          'C++処理時間の内訳。全て秒。「残り」は作用素の構築・最適化・出力を含みます。',spec='lrrrrr')
    residual_rows=[]
    for method in ['standard','quick','long_solver','cold_start','balanced4']:
        with np.load(ab[method]['_path'].with_suffix('.npz')) as a:
            energy=a['history'][-1,3]
        m=ab[method]['metrics']
        residual_rows.append([LABEL[method],number(energy,2),number(m['fixed_point_primal'],5),number(m['fixed_point_dual'],5),number(m['common_visible_epe_mm'])])
    table('table_residual.tex',['条件','最終保持目的値','$r_P$','$r_D$','誤差 [mm]'],residual_rows,
          '同じ形式の目的関数の到達値と固定点残差。候補の選択は変わり得ます。',spec='lrrrr')
    stress_rows=[]
    for name in STRESS:
        rs=[r for r in records if r['split']=='stress' and r['case']==name]
        for r in sorted(rs,key=lambda x:['standard','no_rigid','affine_photo','correct_bound'].index(x['method'])):
            m=r['metrics']; label=LABEL[r['method']]
            if name=='wrong_bound' and r['method']=='standard':label='2 cm上限'
            stress_rows.append([STRESS[name],label,number(m['common_visible_epe_mm']),number(m['moving_epe_mm']),number(m['occluded_epe_mm']),number(m['elapsed_seconds'],2)])
    table('table_stress.tex',['条件','手法','共通可視','動体','遮蔽','時間 [s]'],stress_rows,
          'ストレス条件と反例。誤差単位はmm。各条件は1入力で、信頼区間ではありません。',spec='llrrrr',small=r'\small')
    package=OUT/'package_verification.json'
    package_check=json.loads(package.read_text()) if package.exists() else {}
    cr=[['保存配列の制約検査',f"{checks['records']}組"],
        ['移動上限の最大超過 [m]',f"{max((r['bound_excess_m'] or 0.) for r in checks['array_checks']):.2e}"],
        ['深度許容域の最大超過 [m]',f"{max(r['trust_excess_m'] for r in checks['array_checks']):.2e}"],
        ['再投影フローの最大差 [画素]',f"{max(r['reprojection_max_px'] for r in checks['array_checks']):.2e}"],
        ['融合あり／なしの変位最大差 [m]',f"{checks['implementation_controls']['standard_vs_unfused']['scene_flow']:.2e}"],
        ['同値核変更／前版の変位最大差 [m]',f"{checks['implementation_controls']['v02_vs_exact_kernel_only']['scene_flow']:.2e}"]]
    if package_check:
        cr+=[['展開・再ビルド後の自動試験',package_check['pytest_summary']],
             ['再計算した代表入力の最大配列差',number(package_check['reproduction_max_abs'],12)]]
    table('table_checks.tex',['検査項目','結果'],cr,'実行した数値検査。小さい差はこの環境の丸め誤差の範囲です。',spec=r'p{.68\linewidth}r')
    descriptions={
        'scale':'長さの無次元化尺度 [m]','sigma_depth':'深度の公称誤差 [m]','sigma_flow':'フロー誤差 [画素]',
        'sigma_image':'灰度残差尺度','sigma_geometry':'幾何残差尺度 [m]','max_motion':'移動距離の上限 [m]',
        'trust_sigma':'深度補正幅の倍率','min_depth':'最小深度 [m]','max_depth':'最大深度 [m]',
        'edge_threshold':'深度段差の閾値 [m]','depth_alpha1':'深度の一次残差係数','depth_alpha0':'深度の二階係数',
        'flow_alpha1':'変位の一次残差係数','flow_alpha0':'変位の二階係数','image_weight':'画像照合の係数',
        'flow_weight':'初期フロー整合の係数','flow_truncation':'初期フロー損失の切断値','photo_reject':'画像照合の採用上限',
        'huber_k':'深度Huberの折れ点','ridge':'補助場の二次項係数','iterations':'最初の内側反復数',
        'outer_iterations':'外側の反復数','candidate_count':'保持候補数','max_search':'探索半径の計算上限 [画素]',
        'patch_radius':'画像窓の半径 [画素]','edge_aware':'正則化の境界停止','use_tgv':'二階正則化（偽なら一次TV）',
        'enforce_bound':'候補・変位の物理制約','centered_photo':'平均輝度差の除去','reciprocal_check':'往復照合',
        'diverse_candidates':'候補間隔による多様化','fast_match':'階層的な有限探索','affine_photo':'輝度利得の代入型補正',
        'rigid_prior':'観測由来の剛体補助項','fused_solver':'双対計算の融合','warm_start':'双対初期値の再利用',
        'keep_identity':'恒等候補の保持','later_iterations':'2回目以降の内側反復数','coarse_stride':'整数探索の間隔',
        'seed_count':'小数画素探索の種の数','rigid_min_points':'剛体推定の最小対応数','rigid_sigma':'剛体補助項の尺度 [m]',
        'rigid_huber':'剛体IRLSの折れ点 [m]','step_balance':'主双対ステップの相反倍率',
        'geometry_huber':'ベクトル幾何Huber','geometry_k':'幾何Huberの折れ点'}
    table('table_config.tex',['設定','値','意味'],
          [[r'\code{'+k+'}',str(v),descriptions[k]] for k,v in asdict(Config()).items()],
          '標準Configの全46項目。数値配列の並びもこの順序です。',spec='lll',small=r'\small')
    dump(OUT/'summary.json',summary)
    return summary

def compose(paths: list[Path], target: Path, columns: int=2, margin: int=12) -> None:
    images=[Image.open(path).convert('RGB') for path in paths]
    width=max(im.width for im in images);height=max(im.height for im in images)
    canvas=Image.new('RGB',(columns*width+(columns+1)*margin,math.ceil(len(images)/columns)*height+(math.ceil(len(images)/columns)+1)*margin),'white')
    for i,im in enumerate(images):canvas.paste(im,((i%columns)*(width+margin)+margin,(i//columns)*(height+margin)+margin))
    canvas.save(target)

def figures(records: list[dict]) -> None:
    FIG.mkdir(parents=True,exist_ok=True)
    parts=[]; error_parts=[]
    for case,title in [('gallery_scene101','New scene: primitives'),('curved_scene202','New scene: curved surfaces')]:
        directory=OUT/'heldout'/case
        with np.load(directory/'data.npz') as d:
            for t in (0,1):
                # Display radiance values, rather than choosing chart colors.
                image=Image.fromarray(np.rint(np.clip(d[f'gray{t}'],0,1)*255).astype('uint8'),'L').convert('RGB')
                image=image.resize((640,480),resample=Image.Resampling.NEAREST)
                canvas=Image.new('RGB',(640,520),'white');canvas.paste(image,(0,40))
                draw=ImageDraw.Draw(canvas)
                font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
                draw.text((8,8),f'{title} | frame {t}',fill='black',font=font)
                path=FIG/f'{case}_gray{t}.png';canvas.save(path);parts.append(path)
            for method in ('v02','standard'):
                with np.load(directory/f'{method}.npz') as r:
                    error=np.linalg.norm(r['scene_flow']-d['gt_scene_flow'],axis=-1)*1000
                # One chart per figure, default matplotlib color map, identical limits.
                fig=plt.figure(figsize=(6.4,5.2));ax=fig.add_subplot(111)
                im=ax.imshow(error,vmin=0,vmax=20,interpolation='nearest')
                ax.set_title(('New gallery: ' if 'gallery' in case else 'New curved scene: ')+('v0.2' if method=='v02' else 'v0.3'),fontsize=17)
                ax.set_xlabel('x [pixels]',fontsize=16);ax.set_ylabel('y [pixels]',fontsize=16);ax.tick_params(labelsize=15)
                bar=fig.colorbar(im,ax=ax);bar.set_label('3-D endpoint error [mm]',fontsize=15);bar.ax.tick_params(labelsize=14)
                fig.tight_layout();slug={'v02':'previous','standard':'refined'}.get(method,method)
                path=FIG/f'{case}_{slug}_error.png';fig.savefig(path,dpi=160);plt.close(fig);error_parts.append(path)
    compose(parts,FIG/'heldout_inputs.png');compose(error_parts,FIG/'paired_errors.png')
    rs={r['method']:r for r in records if r['split']=='ablation'}
    labels={'v02':'v0.2','standard':'Standard','no_rigid':'No rigid','quick':'Short iterations','tv':'TV',
            'long_solver':'Long iterations','full_search':'Full search','cold_start':'Cold dual','affine_photo':'Affine gray'}
    selected=list(labels)
    fig=plt.figure(figsize=(8.5,5.4));ax=fig.add_subplot(111)
    xs=[rs[k]['metrics']['elapsed_seconds'] for k in selected];ys=[rs[k]['metrics']['common_visible_epe_mm'] for k in selected]
    ax.scatter(xs,ys)
    offsets={'standard':(10,0),'quick':(-23,23),'tv':(-10,-18),'cold_start':(12,-13),'affine_photo':(12,20),
             'full_search':(12,5),'long_solver':(12,-15),'v02':(-45,10),'no_rigid':(10,4)}
    for k,x,y in zip(selected,xs,ys,strict=True):ax.annotate(labels[k],(x,y),xytext=offsets[k],textcoords='offset points',fontsize=9)
    ax.set_xlabel('C++ inference time [s]');ax.set_ylabel('Mean 3-D endpoint error [mm]')
    ax.set_title('Matched-input ablation: gallery, noise seed 37')
    ax.set_xlim(4,38);ax.set_ylim(.4,3.65);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(FIG/'speed_accuracy.png',dpi=170);plt.close(fig)
    # Negative-control graph: dynamic region only, so static pixels cannot mask failure.
    names=['coplanar_split','rotation_uniform','rotation_textured']
    fig=plt.figure(figsize=(8,5));ax=fig.add_subplot(111)
    for method,label in [('standard','Standard'),('no_rigid','No rigid')]:
        vals=[next(r for r in records if r['split']=='stress' and r['case']==name and r['method']==method)['metrics']['moving_epe_mm'] for name in names]
        ax.plot(range(len(names)),vals,marker='o',label=label)
    ax.set_xticks(range(len(names)),['Same-depth independent motion','Uniform rotating sphere','Textured rotating sphere'])
    ax.set_ylabel('Moving-region endpoint error [mm]');ax.set_title('Negative controls (single inputs)');ax.legend();fig.tight_layout()
    fig.savefig(FIG/'negative_controls.png',dpi=170);plt.close(fig)

def main() -> None:
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--skip-figures',action='store_true');parser.add_argument('--results',type=Path,default=OUT);args=parser.parse_args()
    OUT=args.results.resolve()
    DOC.mkdir(parents=True,exist_ok=True)
    records=read_records(); checks=audit(records)
    flat=[];objects=[]
    for r in records:
        base={k:r[k] for k in ['split','case','method','layout','seed','scene_seed']}
        flat.append({**base,**r['metrics']})
        objects += [{**base,**o} for o in r['objects']]
    csv_write(OUT/'metrics.csv',flat);csv_write(OUT/'object_metrics.csv',objects)
    csv_write(OUT/'ablation.csv',[r for r in flat if r['split']=='ablation'])
    summary=generate_tables(records,checks)
    if not args.skip_figures:figures(records)
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
