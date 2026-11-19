"""原生随机场景与逐局随机流；不导入旧二维环境，不推进物理或障碍运动。"""
import copy
import math
import numpy as np
import torch
from parallel.env.config_loader import keys, number, positive_integer
from parallel.env.layout import EntityLayout

SIDES=('north','south','east','west')
OPPOSITE=dict(north='south',south='north',east='west',west='east')


def validate_scene(c):
    keys(c, ('bounds','evader_radius','start_goal','static_obstacles','dynamic_obstacles','max_sampling_attempts'), 'scene')
    keys(c['bounds'], ('half_extent_x','half_extent_y'), 'scene.bounds')
    for k,v in c['bounds'].items(): number(v,k,minimum=0,strict=True)
    number(c['evader_radius'],'evader_radius',minimum=0,strict=True)
    positive_integer(c['max_sampling_attempts'],'max_sampling_attempts')
    q=c['start_goal'];keys(q,('start_side','goal_side','side_mode','half_extent','min_distance','boundary_clearance'),'scene.start_goal')
    for k in ('start_side','goal_side'):
        if q[k] not in (None,'north','south','east','west'): raise ValueError('Invalid '+k)
    if q['side_mode'] not in ('opposite','different'): raise ValueError('Invalid side_mode')
    for k in ('half_extent','min_distance','boundary_clearance'): number(q[k],k,minimum=0,strict=True)
    if q['start_side'] is not None and q['start_side']==q['goal_side']: raise ValueError('Start and goal sides must differ')
    if q['side_mode']=='opposite' and q['start_side'] is not None and q['goal_side'] is not None:
        if OPPOSITE[q['start_side']]!=q['goal_side']: raise ValueError('Fixed sides contradict opposite mode')
    for group,expected in (
        ('static_obstacles',('count','radius_min','radius_max','region_half_extent','spacing_margin','start_goal_clearance')),
        ('dynamic_obstacles',('count','radii','speed_min','speed_max','region_half_extent','spacing_margin','start_goal_clearance','local_goal_half_range','goal_switch_threshold'))):
        q=c[group];keys(q,expected,'scene.'+group)
        if group=='static_obstacles': positive_integer(q['count'],group+'.count')
        elif type(q['count']) is not int or q['count']<0: raise ValueError(group+'.count must be nonnegative int')
        for k in expected:
            if k not in ('count','radii'): number(q[k],group+'.'+k,minimum=0,strict=k in ('radius_min','radius_max','region_half_extent','local_goal_half_range','goal_switch_threshold'))
    s=c['static_obstacles'];d=c['dynamic_obstacles']
    if s['radius_min']>s['radius_max'] or d['speed_min']>d['speed_max']: raise ValueError('Invalid scene sampling range')
    if not isinstance(d['radii'],list) or not d['radii']: raise ValueError('dynamic radii must be a nonempty list')
    for r in d['radii']: number(r,'dynamic radius',minimum=0,strict=True)
    bound=min(c['bounds'].values())
    if c['evader_radius']+c['start_goal']['boundary_clearance']>=bound: raise ValueError('Evader does not fit inside bounds')
    # region_half_extent is a center bound. Static placement is clipped per radius.
    if d['region_half_extent']+max(d['radii'])>bound: raise ValueError('Dynamic activity region plus radius exceeds scene bounds')
    if s['radius_max']>=bound: raise ValueError('Static radius exceeds scene bounds')


def validate_pursuer_spawn(q):
    keys(q,('count','radius','speed_min','speed_max','region_half_extent','patrol_radius','stratified','spacing_margin','start_goal_clearance','anchor_clearance'), 'pursuer.spawn')
    if type(q['count']) is not int or q['count']<0: raise ValueError('pursuer.spawn.count must be nonnegative int')
    if type(q['stratified']) is not bool: raise ValueError('stratified must be bool')
    for k,v in q.items():
        if k not in ('count','stratified'): number(v,'pursuer.spawn.'+k,minimum=0,strict=k in ('radius','region_half_extent'))
    if q['speed_min']>q['speed_max'] or q['speed_max']>2: raise ValueError('P speed range must satisfy 0 <= min <= max <= 2 m/s')


class Scene:
    def __init__(self,seeds,config,pursuer_spawn,*,device,dtype,reset_motion,target_height=None,formation=None):
        if formation not in (None,'one_side_triangle','one_side'):
            raise ValueError('formation must be one_side, one_side_triangle or omitted')
        self.formation=formation
        self.config=copy.deepcopy(config);self.pursuer_spawn=copy.deepcopy(pursuer_spawn)
        validate_scene(self.config);validate_pursuer_spawn(self.pursuer_spawn)
        self.layout=EntityLayout(self.pursuer_spawn['count'],self.config['dynamic_obstacles']['count'])
        if pursuer_spawn['region_half_extent']+pursuer_spawn['radius']>min(config['bounds'].values()):
            raise ValueError('P activity region plus radius exceeds scene bounds')
        self.device,self.dtype,self.reset_motion=device,dtype,reset_motion
        if target_height is None:
            from parallel.env.config_loader import load_settings
            target_height=load_settings()['simulation']['target_height']
        self.target_height=target_height
        self.generators=[np.random.default_rng(seed) for seed in seeds]
        if not self.generators: raise ValueError('At least one environment is required')
        self.initial=[self._generate(rng) for rng in self.generators]
        self.load_initial()

    def rng_state(self,i):
        return copy.deepcopy(self.generators[i].bit_generator.state)

    def _point_on_side(self,rng,side):
        c=self.config;r=c['evader_radius'];q=c['start_goal']
        x=min(q['half_extent'],c['bounds']['half_extent_x']-r-q['boundary_clearance'])
        y=min(q['half_extent'],c['bounds']['half_extent_y']-r-q['boundary_clearance'])
        if side in ('north','south'): return np.array([rng.uniform(-x,x),y if side=='north' else -y])
        return np.array([x if side=='east' else -x,rng.uniform(-y,y)])

    def _anchors(self,rng):
        q=self.config['start_goal']
        for _ in range(self.config['max_sampling_attempts']):
            start,goal=q['start_side'],q['goal_side']
            if start is None:
                choices=SIDES if goal is None else tuple(s for s in SIDES if s!=goal)
                start=OPPOSITE[goal] if goal is not None and q['side_mode']=='opposite' else str(rng.choice(choices))
            if goal is None:
                goal=OPPOSITE[start] if q['side_mode']=='opposite' else str(rng.choice([s for s in SIDES if s!=start]))
            a,b=self._point_on_side(rng,start),self._point_on_side(rng,goal)
            if np.linalg.norm(b-a)>=q['min_distance']: return a,b
        raise ValueError('Cannot sample start/goal: check distance, bounds and side settings')

    def _place(self,rng,radius,lo,hi,anchors,anchor_gap,occupied,spacing,label):
        for _ in range(self.config['max_sampling_attempts']):
            p=rng.uniform(lo,hi,size=2)
            if any(np.linalg.norm(p-a)<=anchor_gap for a in anchors): continue
            if any(np.linalg.norm(p-np.asarray(o[:2]))<=radius+o[2]+spacing for o in occupied): continue
            return p
        raise ValueError(f'Cannot place {label}: sampling budget exhausted; reduce density/clearances or enlarge scene')

    def _target(self,rng,origin,pursuer):
        q=self.pursuer_spawn if pursuer else self.config['dynamic_obstacles']
        if pursuer:
            angle=rng.uniform(-math.pi,math.pi);radius=q['patrol_radius']*math.sqrt(rng.random())
            offset=radius*np.array([math.cos(angle),math.sin(angle)])
        else: offset=rng.uniform(-q['local_goal_half_range'],q['local_goal_half_range'],2)
        return np.clip(origin+offset,-q['region_half_extent'],q['region_half_extent'])

    def sample_dynamic_goal(self,i,j):
        if not 0<=j<self.layout.num_dynamic: raise IndexError('Dynamic obstacle index out of range')
        o=self.initial[i]['dynamic_obstacles'][j]
        return self._target(self.generators[i],np.array([o['origin_x'],o['origin_y']]),False)

    def _one_side_triangle(self,rng,start,static,objects,q):
        """三架追捕机改到目标同一侧的三角形上，开局速度为零，机头朝向目标。"""
        if q['count']!=3:
            raise ValueError('one_side_triangle requires exactly 3 pursuers')
        from parallel.env.mid360 import one_side_triangle
        buildings=[(b['x0'],b['y0'],b['x1'],b['y1'],b['height']) for b in static]
        bounds=(self.config['bounds']['half_extent_x'],self.config['bounds']['half_extent_y'])
        target=np.array([start[0],start[1],self.target_height],dtype=float)
        positions,yaws=one_side_triangle(rng,target,buildings,bounds)
        placed=[]
        for point,yaw in zip(positions,yaws):
            placed.append(dict(x=float(point[0]),y=float(point[1]),z=float(point[2]),
                origin_x=float(point[0]),origin_y=float(point[1]),radius=float(q['radius']),
                speed=float(q['speed_max']),heading=float(yaw),vx=0.,vy=0.,angular_velocity=0.,
                goal_x=float(point[0]),goal_y=float(point[1]),is_pursuer=True,pursuer_mode=0,
                pursuer_ever_detected=False,pursuer_detected=False,pursuer_search_steps_remaining=0,
                last_seen_x=0.,last_seen_y=0.))
        return placed

    def sample_pursuer_goal(self,i,j):
        if not 0<=j<self.layout.num_pursuers: raise IndexError('Pursuer index out of range')
        o=self.initial[i]['dynamic_obstacles'][self.layout.num_dynamic+j]
        return self._target(self.generators[i],np.array([o['origin_x'],o['origin_y']]),True)

    def _handoff_triangle(self,rng,start,goal,columns,bounds):
        """三架放在逃逸机朝向目标的一侧。这一侧放不下时只改逃逸机位置，圆柱不动。"""
        from parallel.env.mid360 import one_side_visible
        height=self.target_height
        half_x,half_y=bounds
        body=float(self.config['evader_radius'])
        goal_gap=float(self.config['start_goal']['min_distance'])
        goal=np.asarray(goal,dtype=float)

        def free(point):
            if abs(point[0])>half_x-0.5 or abs(point[1])>half_y-0.5:
                return False
            if np.linalg.norm(point-goal)<goal_gap:
                return False
            return all(math.hypot(point[0]-x,point[1]-y)>radius+body+0.45 for x,y,radius in columns)

        last_error=None
        origin=np.asarray(start,dtype=float)
        for attempt in range(128):
            if attempt:
                origin=np.array([rng.uniform(-half_x+0.5,half_x-0.5),rng.uniform(-half_y+0.5,half_y-0.5)])
                if not free(origin):
                    continue
            heading=math.atan2(float(goal[1]-origin[1]), float(goal[0]-origin[0]))
            try:
                placed=one_side_visible(rng,np.array([origin[0],origin[1],height]),columns,(float(half_x),float(half_y)),heading)
            except RuntimeError as exc:
                last_error=exc
                continue
            return placed,origin
        raise RuntimeError('Unable to place a visible one-side formation among cylinders') from last_error

    def _generate(self,rng):
        c=self.config;start,goal=self._anchors(rng)
        forward=(goal-start)/np.linalg.norm(goal-start);left=np.array([-forward[1],forward[0]])
        anchors=(start,goal);occupied=[];static=[];objects=[]
        s=c['static_obstacles'];bounds=np.array([c['bounds']['half_extent_x'],c['bounds']['half_extent_y']])
        buildings=[]
        triangle=None
        if self.formation=='one_side_triangle':
            from parallel.env.buildings import sample_buildings
            last_error=None
            for _attempt in range(16):
                buildings=[]
                for x0,y0,x1,y1,height in sample_buildings(rng,start,goal,(float(bounds[0]),float(bounds[1]))):
                    buildings.append(dict(x0=x0,y0=y0,x1=x1,y1=y1,height=height))
                try:
                    triangle=self._one_side_triangle(rng,start,buildings,[],self.pursuer_spawn)
                    break
                except RuntimeError as exc:
                    last_error=exc
                    triangle=None
            if triangle is None:
                raise RuntimeError('Unable to place five buildings and a visible one-side triangle') from last_error
            static.extend(buildings)
        else:
            for j in range(s['count']):
                radius=rng.uniform(s['radius_min'],s['radius_max']);extent=np.minimum(s['region_half_extent'],bounds-radius)
                xy=self._place(rng,radius,-extent,extent,anchors,radius+c['evader_radius']+s['start_goal_clearance'],occupied,s['spacing_margin'],f'static obstacle {j}')
                occupied.append((*xy,radius));static.append(dict(x=float(xy[0]),y=float(xy[1]),radius=float(radius)))
        for pursuer,q in ((False,c['dynamic_obstacles']),(True,self.pursuer_spawn)):
            if pursuer and self.formation=='one_side_triangle':
                objects.extend(triangle)
                continue
            if pursuer and self.formation=='one_side':
                columns=[(o['x'],o['y'],o['radius']) for o in static if 'radius' in o]
                columns.extend((o['x'],o['y'],o['radius']) for o in objects)
                (positions,yaws),start=self._handoff_triangle(rng,start,goal,columns,(float(bounds[0]),float(bounds[1])))
                for point,yaw in zip(positions,yaws):
                    objects.append(dict(x=float(point[0]),y=float(point[1]),z=float(point[2]),
                        origin_x=float(point[0]),origin_y=float(point[1]),radius=float(q['radius']),
                        speed=0.,heading=float(yaw),vx=0.,vy=0.,angular_velocity=0.,
                        goal_x=float(point[0]),goal_y=float(point[1]),is_pursuer=True,pursuer_mode=0,
                        pursuer_ever_detected=False,pursuer_detected=False,pursuer_search_steps_remaining=0,
                        last_seen_x=0.,last_seen_y=0.))
                continue
            if pursuer and q['stratified'] and q['count']:
                cols=math.ceil(math.sqrt(q['count']))
                rows=math.ceil(q['count']/cols)
                # 每局无放回抽取出生格；排列顺序同时随机分配 P 身份。
                # 三个 P 使用四格中的三个，空格不再固定在右上。
                spawn_cells=rng.permutation(cols*rows)[:q['count']]
            for j in range(q['count']):
                radius=q['radius'] if pursuer else float(rng.choice(q['radii']))
                extent=q['region_half_extent'];lo=np.full(2,-extent);hi=np.full(2,extent)
                if pursuer and q['stratified'] and q['count']:
                    cell_id=int(spawn_cells[j])
                    cell=np.array([2*extent/cols,2*extent/rows])
                    lo+=cell*np.array([cell_id%cols,cell_id//cols]);hi=lo+cell
                    lo+=.15*cell;hi-=.15*cell
                gap=radius+c['evader_radius']+q['start_goal_clearance']
                if pursuer: gap=max(gap,q['anchor_clearance'])
                xy=self._place(rng,radius,lo,hi,anchors,gap,occupied,q['spacing_margin'],f'{"pursuer" if pursuer else "dynamic obstacle"} {j}')
                occupied.append((*xy,radius));speed=float(rng.uniform(q['speed_min'],q['speed_max']))
                heading=float(rng.uniform(-math.pi,math.pi));target=self._target(rng,xy,pursuer)
                delta=target-xy;velocity=delta/max(float(np.linalg.norm(delta)),1e-12)*speed
                if pursuer: velocity=np.zeros(2) if self.reset_motion=='hover' else speed*np.array([math.cos(heading),math.sin(heading)])
                objects.append(dict(x=float(xy[0]),y=float(xy[1]),origin_x=float(xy[0]),origin_y=float(xy[1]),radius=float(radius),
                    speed=speed,heading=heading,vx=float(velocity[0]),vy=float(velocity[1]),angular_velocity=0.,
                    goal_x=float(target[0]),goal_y=float(target[1]),is_pursuer=pursuer,pursuer_mode=0,
                    pursuer_ever_detected=False,pursuer_detected=False,pursuer_search_steps_remaining=0,last_seen_x=0.,last_seen_y=0.))
        forward=(goal-start)/np.linalg.norm(goal-start);left=np.array([-forward[1],forward[0]])
        return dict(schema_version=1,start=start.tolist(),goal=goal.tolist(),frame_x=forward.tolist(),frame_y=left.tolist(),
            static_obstacles=[o for o in static if 'radius' in o],buildings=buildings,dynamic_obstacles=objects,
            robot_state=dict(x=float(start[0]),y=float(start[1]),heading=math.atan2(forward[1],forward[0]),vx=0.,vy=0.,angular_velocity=0.),
            rng_state=copy.deepcopy(rng.bit_generator.state))

    def reset(self,selected,resample=True,seeds=None):
        # 先暂存整组选中局；后面的采样失败不能导致前面的局已经被重置。
        pending=[]
        for k,i in enumerate(selected):
            rng=copy.deepcopy(self.generators[i]) if seeds is None else np.random.default_rng(seeds[k])
            if resample: snapshot=self._generate(rng)
            else:
                snapshot=self.initial[i];rng.bit_generator.state=copy.deepcopy(snapshot['rng_state'])
            pending.append((i,rng,snapshot))
        for i,rng,snapshot in pending:
            self.generators[i]=rng;self.initial[i]=snapshot
        if resample: self.load_initial(torch.tensor(selected,device=self.device,dtype=torch.long))

    def tensor(self,data):
        return torch.as_tensor(np.asarray(data),device=self.device,dtype=self.dtype)

    def load_initial(self, ids=None):
        states = self.initial if ids is None else [self.initial[i] for i in ids.cpu().tolist()]
        circles=[s['static_obstacles'] for s in states]
        if circles[0]:
            value = self.tensor([[[o['x'], o['y'], o['radius']] for o in row] for row in circles])
        else:
            value = torch.zeros(len(states), 0, 3, device=self.device, dtype=self.dtype)
        if ids is None: self.static = value
        else: self.static[ids] = value
        blocks=[s.get('buildings',[]) for s in states]
        if blocks[0]:
            value = self.tensor([[[b['x0'], b['y0'], b['x1'], b['y1'], b['height']] for b in row] for row in blocks])
        else:
            value = torch.zeros(len(states), 0, 5, device=self.device, dtype=self.dtype)
        if ids is None: self.buildings = value
        else: self.buildings[ids] = value
        value = self.tensor([s['goal'] for s in states])
        if ids is None: self.goal = value
        else: self.goal[ids] = value
        value = self.tensor([[s['frame_x'], s['frame_y']] for s in states])
        if ids is None: self.frame = value
        else: self.frame[ids] = value
        value = self.tensor([[[o['origin_x'], o['origin_y']] for o in s['dynamic_obstacles']] for s in states]).reshape(len(states),self.layout.num_objects,2)
        if ids is None: self.origin = value
        else: self.origin[ids] = value
        value = self.tensor([[o['radius'] for o in s['dynamic_obstacles']] for s in states])
        if ids is None: self.radii = value
        else: self.radii[ids] = value
        value = self.tensor([[o['speed'] for o in s['dynamic_obstacles']] for s in states])
        if ids is None: self.speed = value
        else: self.speed[ids] = value
        value = self.tensor([[[s['robot_state']['x'], s['robot_state']['y'], self.target_height]] + [[o['x'], o['y'], o.get('z', self.target_height)] for o in s['dynamic_obstacles'][self.layout.pursuer_objects]] for s in states])
        if ids is None: self.initial_pos = value
        else: self.initial_pos[ids] = value
        value = self.tensor([[s['robot_state']['heading']] + [o['heading'] for o in s['dynamic_obstacles'][self.layout.pursuer_objects]] for s in states])
        if ids is None: self.initial_yaw = value
        else: self.initial_yaw[ids] = value
        value = self.tensor([[[0., 0., 0., 0., 0., 0.]] + [[o['vx'],o['vy'],0.,0.,0.,o['angular_velocity']] for o in s['dynamic_obstacles'][self.layout.pursuer_objects]] for s in states])
        if ids is None: self.initial_velocity = value
        else: self.initial_velocity[ids] = value
