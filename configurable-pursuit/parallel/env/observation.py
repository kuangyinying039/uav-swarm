"""配置驱动的 E 观测；只读取显式输入，不修改场景或推进时间。"""
import math
import copy
import torch


def norm(x):
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def unit(x):
    return torch.where(norm(x)>1e-9,x/norm(x).clamp_min(1e-9),torch.zeros_like(x))


def build_observation(state, *, goal, frame, static, background, bg_velocity, radii, dtype=torch.float32, config=None, motor_state=None, boundary_half_extent=(10.5,13.)):
    sensor=validate_observation_config(config or {})
    lidar_range=sensor['lidar_range'];dynamic_range=sensor['dynamic_range']
    e=state[:,0,:2]; vel=state[:,0,7:9]
    to_frame=lambda x: torch.einsum('n...d,nkd->n...k',x,frame)
    rel=goal-e
    state_obs=torch.cat((unit(to_frame(rel)),norm(rel),to_frame(vel)),dim=-1)
    # Near-tangent ray intersections amplify float32 cancellation in b*b-c.
    # Compute only analytic lidar geometry in float64, then return the normal
    # observation dtype. Unit rays preserve metric distance on either device.
    geometry_dtype=torch.float64
    angles=torch.arange(36,device=state.device,dtype=geometry_dtype)*(2*math.pi/36)
    rays=torch.einsum('rk,nkd->nrd',torch.stack((angles.cos(),angles.sin()),-1),frame.to(geometry_dtype))
    rays=rays/torch.linalg.vector_norm(rays,dim=-1,keepdim=True)
    if static.shape[1]:
        static=static.to(geometry_dtype)
        offset=e.to(geometry_dtype)[:,None,None,:]-static[:,None,:,:2]
        b=(offset*rays[:,:,None]).sum(-1)
        c=offset.square().sum(-1)-static[:,None,:,2].square()
        disc=b.square()-c
        near=-b-disc.clamp_min(0).sqrt(); far=-b+disc.clamp_min(0).sqrt()
        hits=torch.where((disc>=0)&(far>=0),near.clamp_min(0),torch.full_like(near,lidar_range))
        distances=hits.amin(-1).clamp_max(lidar_range)
    else:
        distances=torch.full((state.shape[0], rays.shape[1]), lidar_range, device=state.device, dtype=geometry_dtype)
    bounds=torch.tensor(boundary_half_extent,device=state.device,dtype=geometry_dtype)
    t=(torch.where(rays>0,bounds,-bounds)-e.to(geometry_dtype)[:,None,:])/torch.where(rays.abs()>1e-9,rays,torch.ones_like(rays))
    t=torch.where((rays.abs()>1e-9)&(t>=0),t,torch.full_like(t,lidar_range))
    distances=torch.minimum(distances,t.amin(-1))
    lidar=(lidar_range-distances).to(dtype)
    positions=background
    velocities=bg_velocity
    delta=positions-e[:,None]; d=norm(delta).squeeze(-1)
    order=d.argsort(dim=-1,stable=True)[:,:3]
    feats=torch.cat((unit(to_frame(delta)),d[...,None],to_frame(velocities),radii[:,:background.shape[1],None]),-1)
    feats=torch.where((d<=dynamic_range)[...,None],feats,torch.zeros_like(feats))
    feats=feats.gather(1,order[...,None].expand(-1,-1,6))
    feats=torch.cat((feats,feats.new_zeros(len(state),3-feats.shape[1],6)),1)
    pursuers,mask=build_pursuer_observation(state,radii[:,background.shape[1]:],config=config,motor_state=motor_state)
    return torch.cat((state_obs,lidar,feats.flatten(1),frame[:,0],pursuers.flatten(1),mask.to(dtype)),-1)


# 固定字段顺序来自配置列表；坐标均为当前局世界系（扣除并行场景偏移）。
FIELD_SPECS={
    'position':(3,'m','world'),
    'relative_position':(3,'m','world_relative_to_E'),
    'velocity':(3,'m/s','world'),
    'attitude':(4,'unit quaternion wxyz','world'),
    'angular_velocity':(3,'rad/s','world'),
    'motor_throttle':(4,'normalized rotor speed [0,1]','rotor order'),
    'radius':(1,'m','scalar'),
}


def validate_observation_config(config):
    if not isinstance(config,dict) or set(config)-{'pursuers','lidar_range','dynamic_range'}:raise ValueError('Invalid observation keys')
    c=copy.deepcopy(config.get('pursuers',{}))
    if not isinstance(c,dict) or set(c)-{'fields','range','max_count'}:raise ValueError('Invalid pursuer observation keys')
    c.setdefault('fields',[]);c.setdefault('range',4.);c.setdefault('max_count',3)
    fields=c['fields']
    if not isinstance(fields,list) or any(not isinstance(k,str) or k not in FIELD_SPECS for k in fields) or len(set(fields))!=len(fields):
        raise ValueError('Unknown or duplicate pursuer observation field')
    if type(c['max_count']) is not int or c['max_count']<0:raise ValueError('max_count must be nonnegative int')
    r=c['range']
    if r is not None and (isinstance(r,bool) or not isinstance(r,(int,float)) or not math.isfinite(r) or r<0):raise ValueError('range must be nonnegative or null')
    sensor={k:config.get(k,4.) for k in ('lidar_range','dynamic_range')}
    for k,v in sensor.items():
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
            raise ValueError(k+' must be finite and positive')
    return {'pursuers':c,**sensor}


def build_pursuer_observation(state,radii,*,config=None,motor_state=None):
    c=validate_observation_config(config or {})['pursuers'];count=c['max_count']
    present=min(count,state.shape[1]-1)
    ps=state.new_zeros(state.shape[0],count,13)
    ps[:,:present]=state[:,1:1+present]
    mask=torch.arange(count,device=state.device)[None,:].expand(state.shape[0],-1)<present
    if c['range'] is not None:
        mask &= torch.linalg.vector_norm(ps[:,:,:2]-state[:,None,0,:2],dim=-1)<=c['range']
    if not c['fields']:mask.zero_()
    data=[]
    for field in c['fields']:
        if field=='position':value=ps[:,:,:3]
        elif field=='relative_position':value=ps[:,:,:3]-state[:,None,0,:3]
        elif field=='velocity':value=ps[:,:,7:10]
        elif field=='attitude':value=ps[:,:,3:7]
        elif field=='angular_velocity':value=ps[:,:,10:13]
        elif field=='radius':
            value=state.new_zeros(state.shape[0],count,1);value[:,:present,0]=radii[:,:present]
        elif field=='motor_throttle':
            if motor_state is None or motor_state.shape!=(state.shape[0],state.shape[1],4):
                raise ValueError('motor_throttle requires actual [N,aircraft,4] rotor state')
            value=state.new_zeros(state.shape[0],count,4);value[:,:present]=motor_state[:,1:1+present]
        if not torch.isfinite(value).all():raise ValueError(f'Nonfinite {field}')
        data.append(torch.where(mask[...,None],value,torch.zeros_like(value)))
    features=torch.cat(data,-1) if data else state.new_empty(state.shape[0],count,0)
    return features,mask


def observation_layout(config):
    c=validate_observation_config(config)['pursuers']
    result=[dict(name='self_goal',start=0,stop=5),dict(name='lidar',start=5,stop=41),
            dict(name='background',start=41,stop=59),dict(name='frame',start=59,stop=61)]
    offset=61
    for i in range(c['max_count']):
        for field in c['fields']:
            width,units,frame=FIELD_SPECS[field]
            result.append(dict(name=f'pursuer_{i}.{field}',start=offset,stop=offset+width,units=units,frame=frame))
            offset+=width
    result.append(dict(name='pursuer_mask',start=offset,stop=offset+c['max_count'],units='0/1',frame='fixed P index'))
    return result


def observation_scale(config, *, environment=None, pursuer=None):
    # 仅训练示例归一化；原始观测不隐式缩放。
    layout=observation_layout(config);scale=torch.ones(layout[-1]['stop'])
    if environment is None:
        from parallel.env.config_loader import load_settings
        environment=load_settings()
    if pursuer is None:
        from parallel.strategies.config import load_config
        pursuer=load_config()[0]
    sensor=validate_observation_config(config)
    scene=environment['scene'];simulation=environment['simulation']
    distance_scale=2*max(*scene['bounds'].values(),simulation['target_height'])
    speed_scale=simulation['evader_speed_limit']
    bg_speed=max(scene['dynamic_obstacles']['speed_max'],speed_scale)
    radius_scale=max(*scene['dynamic_obstacles']['radii'],pursuer['spawn']['radius'])
    scale[2]=distance_scale;scale[3:5]=speed_scale;scale[5:41]=sensor['lidar_range']
    scale[41:59]=torch.tensor([1.,1.,sensor['dynamic_range'],bg_speed,bg_speed,radius_scale]).repeat(3)
    for item in layout[4:-1]:
        field=item['name'].split('.')[-1]
        value={'position':distance_scale,'relative_position':distance_scale,
               'velocity':max(environment['task']['physical_failure']['max_horizontal_speed'],pursuer['spawn']['speed_max']),
               'angular_velocity':10.,'radius':radius_scale}.get(field,1.)
        scale[item['start']:item['stop']]=value
    return scale
