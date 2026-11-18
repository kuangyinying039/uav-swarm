"""唯一人工策略配置入口；不读取旧采样 JSON 的策略概率。"""
import copy,json,math
from pathlib import Path
from . import REGISTRY

DEFAULT=Path(__file__).resolve().parents[1]/'config/pursuer.json'

def load_config(source=None):
    cfg=copy.deepcopy(source) if isinstance(source,dict) else json.loads(Path(source or DEFAULT).read_text())
    if set(cfg)!={'schema_version','sampling_seed','common','strategies','spawn'} or cfg['schema_version']!=1:
        raise ValueError('Invalid pursuer config schema')
    if type(cfg['sampling_seed']) is not int or cfg['sampling_seed']<0:raise ValueError('sampling_seed must be a nonnegative integer')
    from parallel.env.scene import validate_pursuer_spawn
    validate_pursuer_spawn(cfg['spawn'])
    c=cfg['common']
    numeric={'detection_range','responsibility_radius','search_duration','patrol_speed','goal_switch_threshold'}
    if set(c)!=numeric|{'use_detection_range','idle_mode'}:raise ValueError('Invalid common parameter keys')
    if type(c['use_detection_range']) is not bool or c['idle_mode'] not in ('hold','patrol'):raise ValueError('Invalid detection or idle mode')
    for k in numeric:
        if isinstance(c[k],bool) or not isinstance(c[k],(int,float)) or not math.isfinite(c[k]) or c[k]<0:raise ValueError(f'Invalid {k}')
    entries=cfg['strategies']
    if not isinstance(entries,list) or not entries:raise ValueError('No strategies enabled')
    names=[];instances=[];weights=[]
    for e in entries:
        if set(e)!={'name','weight','params'} or e['name'] not in REGISTRY or e['name'] in names:raise ValueError('Unknown/duplicate strategy or invalid keys')
        w=e['weight']
        if isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<0:raise ValueError('Invalid strategy weight')
        if not isinstance(e['params'],dict):raise ValueError('Strategy params must be an object')
        obj=REGISTRY[e['name']](e['params']);e['params']=obj.params.copy()
        names.append(e['name']);instances.append(obj);weights.append(w)
    total=sum(weights)
    if not math.isfinite(total) or total<=0:raise ValueError('Strategy weights must have a positive finite sum')
    return cfg,tuple(names),instances,[w/total for w in weights]
