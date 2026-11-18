"""演示相机和MP4编码；只接收仿真图像，不捕获桌面，不读取策略模型。"""
import math
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np


class FrameNotReady(RuntimeError):
    pass


class VideoRecorder:
    def __init__(self,path,*,fps,resolution):
        self.path=Path(path);self.partial=self.path.with_suffix('.partial.mp4')
        if self.path.exists() or self.partial.exists():raise FileExistsError(self.path)
        self.fps=fps;self.width,self.height=resolution;self.frames=0;self.last_time=-1.
        ffmpeg=shutil.which('ffmpeg')
        if ffmpeg is None:raise RuntimeError('Recording requires ffmpeg with libx264 on PATH')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.log=self.path.with_suffix('.ffmpeg.log').open('wb')
        try:
            self.process=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-n',
                '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{self.width}x{self.height}',
                '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p',
                '-crf','18','-preset','slow','-movflags','+faststart',str(self.partial)],stdin=subprocess.PIPE,stderr=self.log)
        except Exception:
            self.log.close();raise
        self.closed=False

    def append(self,rgba,simulation_time):
        if self.closed:raise RuntimeError('Recorder closed')
        if not math.isfinite(simulation_time) or simulation_time<self.last_time or simulation_time<0:
            raise ValueError('Recording time must be finite, nonnegative and monotonic')
        pixels=np.asarray(rgba)
        if pixels.dtype!=np.uint8 or pixels.shape not in ((self.height,self.width,3),(self.height,self.width,4)):
            raise ValueError('Expected uint8 HxWx3/4 camera frame')
        self.last_time=simulation_time
        # Video follows simulated time, independent of wall-clock rendering speed.
        count=math.floor(simulation_time*self.fps+1e-7)+1-self.frames
        if count>0:
            data=np.ascontiguousarray(pixels[:,:,:3]).tobytes()
            for _ in range(count):
                try:self.process.stdin.write(data)
                except BrokenPipeError as exc:raise RuntimeError(f'Video encoder failed; inspect {self.path.with_suffix(".ffmpeg.log")}') from exc
                self.frames+=1

    def close(self,success=True):
        if self.closed:return
        self.closed=True
        try:
            try:self.process.stdin.close()
            except BrokenPipeError:success=False
            try:code=self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait();raise RuntimeError('Video encoder did not finalize in time')
            if code!=0:raise RuntimeError(f'Video encoder exit {code}; inspect {self.path.with_suffix(".ffmpeg.log")}')
            if success and self.frames and self.partial.exists():
                self.partial.rename(self.path)
            elif success:raise RuntimeError('No video frames recorded')
        finally:self.log.close()


def demo_app_options(cfg,visuals):
    w,h=visuals['camera']['resolution']
    # FXAA avoids temporal TAA history blur; all textures keep native resolution.
    # PathTracing-only sample counts do not improve RayTracedLighting.
    return dict(headless=bool(cfg.headless),anti_aliasing=2,renderer='RayTracedLighting',
                width=w,height=h,window_width=w,window_height=h)


def create_demo_app(cfg,visuals):
    from isaacsim import SimulationApp
    return SimulationApp(demo_app_options(cfg,visuals),
        experience=str(Path(os.environ['EXP_PATH'])/'omni.isaac.sim.python.kit'))


def camera_layout(config,bounds,highest):
    """USD camera rotation columns: right, up, backward; fit full scene box."""
    from scipy.spatial.transform import Rotation
    eye=np.array([0.,0.,config['height']] if config['view']=='top' else config['position'],dtype=float)
    target=np.array([0.,0.,0.] if config['view']=='top' else config['target'],dtype=float)
    if eye[2]<=highest:raise ValueError('Camera must be above aircraft and display obstacles')
    back=eye-target;back/=np.linalg.norm(back)
    up_hint=np.array([0.,1.,0.]) if abs(back[2])>.999 else np.array([0.,0.,1.])
    right=np.cross(up_hint,back);right/=np.linalg.norm(right)
    up=np.cross(back,right);rotation=np.column_stack((right,up,back))
    q=Rotation.from_matrix(rotation).as_quat();orientation=np.array([q[3],*q[:3]])
    corners=np.array([[x,y,z] for x in [-bounds['half_extent_x'],bounds['half_extent_x']]
                      for y in [-bounds['half_extent_y'],bounds['half_extent_y']] for z in [0.,highest]])
    projected=(corners-target)@rotation
    if np.any((corners-eye)@back>=0):raise ValueError('Camera position places scene behind camera')
    w,h=config['resolution'];margin=config['margin']
    width=2*max(abs(projected[:,0]).max()+margin,(abs(projected[:,1]).max()+margin)*w/h)
    return eye,orientation,width/config.get('zoom',1.),float(np.linalg.norm(corners-eye,axis=1).max()+10)


class CameraFollower:
    """Translate by target displacement; preserve interactive pose edits and zoom."""
    def __init__(self,anchor):
        self.anchor=np.asarray(anchor,dtype=float).copy()

    def update(self,camera,target):
        target=np.asarray(target,dtype=float)
        delta=target-self.anchor
        if np.any(delta):
            position,orientation=camera.get_world_pose(camera_axes='usd')
            if hasattr(position,'detach'):position=position.detach().cpu().numpy()
            if hasattr(orientation,'detach'):orientation=orientation.detach().cpu().numpy()
            camera.set_world_pose(position=np.asarray(position)+delta,orientation=orientation,camera_axes='usd')
        self.anchor=target.copy()


class DemoCamera:
    def __init__(self,env,config):
        from omni.isaac.sensor import Camera
        from pxr import UsdGeom,Gf,UsdShade,Sdf
        self.env=env;self.camera_config=config['camera'];self.style=config['markers']
        base=env.unwrapped;scene=base.rules.settings['scene'];w,h=self.camera_config['resolution']
        highest=max(base.rules.settings['simulation']['target_height']+1.,config['static_obstacles']['height'],config['dynamic_obstacles']['height'])
        eye,orientation,width,far=camera_layout(self.camera_config,scene['bounds'],highest)
        offset=base.offsets[0].cpu().numpy()
        self.camera=Camera('/World/DemoCamera',frequency=-1,resolution=(w,h))
        self.camera.set_world_pose(position=offset+eye,orientation=orientation,camera_axes='usd')
        self.camera.prim.GetAttribute('projection').Set('orthographic')
        self.camera.initialize()
        self.camera.set_horizontal_aperture(width)
        self.camera.set_clipping_range(.1,far)
        target=np.zeros(3) if self.camera_config['view']=='top' else np.asarray(self.camera_config['target'])
        self.follower=CameraFollower(offset+target) if self.camera_config['follow_evader'] else None
        # Thin emissive lines identify real drone models without covering their bodies.
        def line(name,points,color):
            path='/World/DemoMarkers/'+name
            curve=UsdGeom.BasisCurves.Define(base.stage,path)
            curve.CreateTypeAttr('linear');curve.CreateWrapAttr('nonperiodic')
            curve.CreateCurveVertexCountsAttr([len(points)])
            curve.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
            curve.CreateWidthsAttr([self.style['line_width']]);curve.SetWidthsInterpolation('constant')
            curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            material=UsdShade.Material.Define(base.stage,path+'Material')
            shader=UsdShade.Shader.Define(base.stage,path+'Material/Shader');shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.,0.,0.))
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.)
            shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.,0.,0.))
            shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)
            curve.AddTranslateOp();return curve
        # Color the existing drone visuals; do not scale rigid bodies or replace physics.
        for j in range(base.rules.layout.num_aircraft):
            color=self.style['evader_color' if j==0 else 'pursuer_color']
            path=f'/World/DemoMaterials/Drone{j}'
            material=UsdShade.Material.Define(base.stage,path)
            shader=UsdShade.Shader.Define(base.stage,path+'/Shader');shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.,0.,0.))
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.)
            shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.,0.,0.))
            shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            prim=base.stage.GetPrimAtPath('/World/envs/env_0/'+base.rules.layout.aircraft_prim_name(j))
            if not prim.IsValid():raise RuntimeError('Demo could not find the existing drone model')
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        # Flat display colors stay legible under the upstream ground asset's bright lights.
        for group in ('static_obstacles','dynamic_obstacles'):
            color=config[group]['color'];path='/World/DemoMaterials/'+group
            material=UsdShade.Material.Define(base.stage,path)
            shader=UsdShade.Shader.Define(base.stage,path+'/Shader');shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.,0.,0.))
            shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.)
            shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.,0.,0.))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            count=base.rules.static.shape[1]
            indices=range(count) if group=='static_obstacles' else range(count,count+base.rules.bg.shape[1])
            for j in indices:
                prim=base.stage.GetPrimAtPath(f'/World/envs/env_0/Geometry/C{j}')
                if prim.IsValid():UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        def circle(radius,z=0.):
            return [(radius*math.cos(a),radius*math.sin(a),z) for a in np.linspace(0,2*math.pi,65)]
        self.glyphs=[line(f'Agent{j}',circle(.45),self.style['evader_color' if j==0 else 'pursuer_color']) for j in range(base.rules.layout.num_aircraft)]
        radius=base.rules.settings['task']['goal_radius'];height=base.rules.settings['simulation']['target_height']
        self.goal=line('Goal',circle(radius),self.style['goal_color'])
        self.goal_ground=line('GoalGround',circle(radius),self.style['goal_color'])
        self.goal_pole=line('GoalPole',[(0,0,.1),(0,0,height+.8),(.7,0,height+.5),(0,0,height+.3)],self.style['goal_color'])
        bx=scene['bounds']['half_extent_x'];by=scene['bounds']['half_extent_y']
        boundary=line('Boundary',[(-bx,-by,.1),(bx,-by,.1),(bx,by,.1),(-bx,by,.1),(-bx,-by,.1)],self.style['boundary_color'])
        boundary.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(*offset.tolist()))
        self._keyboard_subscription=None
        if not base.rules.settings['simulation']['headless']:
            from omni.kit.viewport.utility import get_active_viewport
            viewport=get_active_viewport()
            if viewport is None:raise RuntimeError('No active viewport for demo window')
            viewport.camera_path='/World/DemoCamera';viewport.set_texture_resolution((w,h))
            import carb.input,omni.appwindow
            self._input=carb.input.acquire_input_interface()
            self._keyboard=omni.appwindow.get_default_app_window().get_keyboard()
            self._keyboard_subscription=self._input.subscribe_to_keyboard_events(self._keyboard,self._on_key)
        self.update()

    def _on_key(self,event,*args):
        import carb.input
        if event.type!=carb.input.KeyboardEventType.KEY_PRESS:return True
        key=carb.input.KeyboardInput
        if event.input in (key.F6,key.F7):
            factor=.8 if event.input==key.F6 else 1.25
            width=min(500.,max(.5,self.camera.get_horizontal_aperture()*factor))
            self.camera.set_horizontal_aperture(width)
            print(f'[demo] Camera width: {width:.2f} m',flush=True)
        elif event.input==key.F8:
            target=(self.env.state()[0,0,:3]+self.env.unwrapped.offsets[0]).cpu().numpy()
            self.follower=CameraFollower(target) if self.follower is None else None
            print(f'[demo] Follow evader: {self.follower is not None}',flush=True)
        return True

    def close(self):
        if self._keyboard_subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard,self._keyboard_subscription)
            self._keyboard_subscription=None

    def update(self):
        from pxr import Gf
        base=self.env.unwrapped;offset=base.offsets[0]
        state=self.env.state()[0,:,:3]+offset
        if self.follower is not None:self.follower.update(self.camera,state[0].cpu().numpy())
        for marker,position in zip(self.glyphs,state.cpu().tolist()):
            marker.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(position[0],position[1],position[2]+.04))
        xy=(base.rules.goal[0]+offset[:2]).cpu().tolist();z=float(offset[2])
        for marker,height in [(self.goal,base.rules.settings['simulation']['target_height']), (self.goal_ground,.12),(self.goal_pole,0.)]:
            marker.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(*xy,z+height))

    def frame(self):
        self.update();self.env.unwrapped.sim.render()
        frame=np.asarray(self.camera.get_rgba());w,h=self.camera_config['resolution']
        if frame.size==0 or frame.ndim<3:raise FrameNotReady('Camera has not delivered an image yet')
        if frame.dtype!=np.uint8 or frame.shape!=(h,w,4):raise RuntimeError(f'Camera frame not ready or invalid: {frame.shape}, {frame.dtype}')
        return frame

    def warmup(self):
        # A correctly shaped first frame need not have converged yet. Render-only
        # updates settle assets/AA without advancing simulation dt or physics.
        latest=None
        for _ in range(20):
            try:latest=self.frame()
            except FrameNotReady:continue
        if latest is None:
            raise RuntimeError('Demo camera did not produce RGBA after 20 render-only updates')
        return latest
