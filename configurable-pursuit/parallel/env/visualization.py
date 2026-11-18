"""配置驱动的障碍显示；位置/半径只读取真实场景，不增加碰撞属性。"""
import torch
from parallel.env.config_loader import load_visualization


class ObstacleVisualization:
    def __init__(self, stage=None, paths=(), obstacle_count=0, *, static_count=None, config=None, enabled=None):
        self.config=load_visualization(config)
        if enabled is not None:
            if type(enabled) is not bool: raise ValueError('enabled must be bool or None')
            self.config['enabled']=enabled
        self.enabled=self.config['enabled']
        self.rows=[]
        if not self.enabled: return
        from pxr import UsdGeom, Gf
        static_count=obstacle_count if static_count is None else static_count
        for path in paths:
            row=[]
            for j in range(obstacle_count):
                style=self.config['static_obstacles' if j<static_count else 'dynamic_obstacles']
                if not style['visible']: continue
                cylinder=UsdGeom.Cylinder.Define(stage,f'{path}/Geometry/C{j}')
                cylinder.CreateHeightAttr(style['height'])
                cylinder.CreateAxisAttr('Z')
                cylinder.CreateDisplayColorAttr().Set([Gf.Vec3f(*style['color'])])
                cylinder.AddTranslateOp()
                row.append((j,cylinder,style['height']/2))
            self.rows.append(row)

    def update(self, static, dynamic_positions, dynamic_radii):
        if not self.enabled: return
        from pxr import Gf
        dynamic=torch.cat((dynamic_positions,dynamic_radii[...,None]),-1)
        geometry=torch.cat((static,dynamic),1).cpu().tolist()
        for row,data in zip(self.rows,geometry):
            for index,cylinder,z in row:
                x,y,radius=data[index]
                cylinder.GetRadiusAttr().Set(radius)
                cylinder.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(x,y,z))

    def close(self):
        """释放显示句柄；USD stage 的销毁仍由运行入口管理。"""
        self.rows.clear()
        self.enabled=False
