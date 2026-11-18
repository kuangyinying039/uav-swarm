"""项目级任务/奖励配置。显式完整配置，拒绝未知字段；不修改共享默认文件。"""
import copy,json,math
from pathlib import Path

DEFAULT=Path(__file__).resolve().parents[1]/'config/environment.json'
TERM_NAMES=('captured','static_collision','dynamic_collision','out_of_bounds','physical_failure',
            'success','timeout','progress','goal_velocity','step','stall')
EVENT_NAMES=TERM_NAMES[:7]

def keys(value,expected,label):
    if not isinstance(value,dict) or set(value)!=set(expected):raise ValueError(f'Invalid keys in {label}')

def number(value,label,*,minimum=None,strict=False):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):raise ValueError(f'{label} must be finite numeric')
    if minimum is not None and (value<=minimum if strict else value<minimum):raise ValueError(f'Invalid {label}')

def positive_integer(value,label):
    if type(value) is not int or value<=0:raise ValueError(f'{label} must be positive int')

def load_settings(source=None):
    cfg=copy.deepcopy(source) if isinstance(source,dict) else json.loads(Path(source or DEFAULT).read_text())
    if 'observation' not in cfg:cfg['observation']={}
    from parallel.env.observation import validate_observation_config
    cfg['observation']=validate_observation_config(cfg['observation'])
    keys(cfg,('schema_version','task','reward','observation','scene','simulation'),'environment')
    if cfg['schema_version']!=1:raise ValueError('Unsupported environment schema')
    validate_simulation(cfg['simulation'])
    from parallel.env.scene import validate_scene
    validate_scene(cfg['scene'])
    resolved_env_spacing(cfg['simulation'],cfg['scene']['bounds'])
    t=cfg['task'];keys(t,('type','goal_radius','max_episode_steps','terminate_on','physical_failure'),'task')
    if t['type']!='peg_evader':raise ValueError('Only peg_evader task is implemented')
    number(t['goal_radius'],'goal_radius',minimum=0,strict=True)
    positive_integer(t['max_episode_steps'],'max_episode_steps')
    keys(t['terminate_on'],EVENT_NAMES[:-1],'terminate_on')
    if any(type(v) is not bool for v in t['terminate_on'].values()):raise ValueError('terminate_on values must be bool')
    q=t['physical_failure'];keys(q,('min_height','max_height','max_horizontal_speed'),'physical_failure')
    for k,v in q.items():number(v,k,minimum=0)
    if q['min_height']>=q['max_height'] or q['max_horizontal_speed']<=0:raise ValueError('Invalid physical thresholds')
    r=cfg['reward'];keys(r,('terms','progress','stall'),'reward');keys(r['terms'],TERM_NAMES,'reward.terms')
    for k,v in r['terms'].items():
        keys(v,('enabled','scale'),k)
        if type(v['enabled']) is not bool:raise ValueError('enabled must be bool')
        number(v['scale'],k+'.scale')
    q=r['progress'];keys(q,('mode','gamma','potential_distance_scale','zero_potential_on_termination','zero_potential_on_truncation'),'progress')
    if q['mode'] not in ('distance','potential'):raise ValueError('Unknown progress mode')
    number(q['gamma'],'gamma',minimum=0)
    if q['gamma']>1:raise ValueError('gamma must be <=1')
    number(q['potential_distance_scale'],'potential_distance_scale',minimum=0,strict=True)
    for k in ('zero_potential_on_termination','zero_potential_on_truncation'):
        if type(q[k]) is not bool:raise ValueError(k+' must be bool')
    q=r['stall'];keys(q,('window_steps','near_goal_distance','min_static_clearance','min_pursuer_clearance','min_progress','max_speed'),'stall')
    positive_integer(q['window_steps'],'window_steps')
    for k,v in q.items():number(v,k,minimum=0)
    if q['min_progress']<=0:raise ValueError('min_progress must be >0')
    return cfg


def validate_simulation(c):
    keys(c,('drone_model','controller','dt','flight_mode','target_height','headless','aircraft_collisions',
            'env_spacing','minimum_env_spacing','env_spacing_margin','evader_speed_limit'),'simulation')
    if (c['drone_model'],c['controller'])!=('Hummingbird','LeePositionController'):
        raise ValueError('Supported model/controller pair: Hummingbird + LeePositionController')
    if c['flight_mode'] not in ('planar','free_3d'): raise ValueError('Invalid flight_mode')
    if type(c['headless']) is not bool: raise ValueError('headless must be bool')
    if type(c['aircraft_collisions']) is not bool: raise ValueError('aircraft_collisions must be bool')
    for k in ('dt','target_height','minimum_env_spacing','env_spacing_margin','evader_speed_limit'):
        number(c[k],'simulation.'+k,minimum=0,strict=True)
    # Isaac 4.1 stores int(1 / dt) Hz, so fractional frequencies silently change dt.
    dt=c['dt']
    if dt>1 or not math.isfinite(1/dt):
        raise ValueError('simulation.dt must correspond to an integer physics frequency >= 1 Hz')
    hz=int(1/dt)
    if hz<1 or not math.isclose(dt,1/hz,rel_tol=1e-12,abs_tol=0):
        raise ValueError(f'simulation.dt={dt} is not representable by Isaac 4.1 integer Hz; use 1/N seconds, e.g. {1/62!r} (62 Hz) or 0.02 (50 Hz)')
    if c['env_spacing'] is not None: number(c['env_spacing'],'env_spacing',minimum=0,strict=True)


def resolved_env_spacing(simulation,bounds):
    needed=2*max(bounds.values())+simulation['env_spacing_margin']
    explicit=simulation['env_spacing']
    if explicit is not None:
        if explicit<needed: raise ValueError('env_spacing too small for map bounds and env_spacing_margin')
        return explicit
    return max(simulation['minimum_env_spacing'],needed)


def configure_isaac_app(cfg,simulation):
    """运行入口必须在启动app之前调用；避免demo.yaml覆盖项目dt/headless。"""
    validate_simulation(simulation)
    cfg.headless=simulation['headless']
    cfg.sim.dt=simulation['dt']
    cfg.sim.device='cuda:0'
    cfg.sim.use_gpu_pipeline=True
    cfg.sim.use_fabric=True
    return cfg


def load_visualization(source=None):
    path=DEFAULT.with_name('visualization.json')
    c=copy.deepcopy(source) if isinstance(source,dict) else json.loads(Path(source or path).read_text())
    c.setdefault('markers',dict(evader_color=[.1,.7,1.],pursuer_color=[1.,.15,.15],goal_color=[.25,1.,.15],boundary_color=[1.,.85,.25],line_width=.06))
    c['camera'].setdefault('follow_evader',False)
    c['camera'].setdefault('zoom',1.)
    c['camera'].setdefault('view','top')
    c['camera'].setdefault('position',[20.,-28.,32.])
    c['camera'].setdefault('target',[0.,0.,0.])
    keys(c,('schema_version','enabled','static_obstacles','dynamic_obstacles','camera','recording','markers'),'visualization')
    if c['schema_version']!=1: raise ValueError('Unsupported visualization schema')
    if type(c['enabled']) is not bool: raise ValueError('visualization.enabled must be bool')
    for group in ('static_obstacles','dynamic_obstacles'):
        q=c[group];keys(q,('visible','color','height'),'visualization.'+group)
        if type(q['visible']) is not bool: raise ValueError('visible must be bool')
        number(q['height'],'display height',minimum=0,strict=True)
        if not isinstance(q['color'],list) or len(q['color'])!=3: raise ValueError('color must be RGB list')
        for channel in q['color']:
            number(channel,'color',minimum=0)
            if channel>1: raise ValueError('color must be within [0,1]')
    camera=c['camera'];keys(camera,('resolution','height','margin','view','position','target','follow_evader','zoom'),'camera')
    if type(camera['follow_evader']) is not bool:raise ValueError('camera.follow_evader must be bool')
    number(camera['zoom'],'camera.zoom',minimum=0,strict=True)
    if camera['view'] not in ('top','oblique'):raise ValueError('camera.view must be top or oblique')
    for key in ('position','target'):
        if not isinstance(camera[key],list) or len(camera[key])!=3:raise ValueError('camera position/target must be XYZ lists')
        for v in camera[key]:number(v,'camera.'+key)
    if camera['position'][2]<=camera['target'][2]:raise ValueError('camera position must be above target')
    m=c['markers'];keys(m,('evader_color','pursuer_color','goal_color','boundary_color','line_width'),'markers')
    number(m['line_width'],'markers.line_width',minimum=0,strict=True)
    for key in ('evader_color','pursuer_color','goal_color','boundary_color'):
        if not isinstance(m[key],list) or len(m[key])!=3:raise ValueError('marker color must be RGB list')
        for v in m[key]:
            number(v,key,minimum=0)
            if v>1:raise ValueError('marker color must be within [0,1]')
    size=camera['resolution']
    if not isinstance(size,list) or len(size)!=2: raise ValueError('camera resolution must be [width,height]')
    for v in size:
        positive_integer(v,'resolution')
        if v%2: raise ValueError('resolution must be even for H264 video')
    number(camera['height'],'camera.height',minimum=0,strict=True)
    number(camera['margin'],'camera.margin',minimum=0)
    record=c['recording'];keys(record,('enabled','output_dir','fps'),'recording')
    if type(record['enabled']) is not bool: raise ValueError('recording.enabled must be bool')
    if not isinstance(record['output_dir'],str) or not record['output_dir'].strip(): raise ValueError('output_dir must be nonempty')
    positive_integer(record['fps'],'recording.fps')
    return c
