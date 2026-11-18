"""构造时固定的实体数量与分区；观测容量独立配置。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EntityLayout:
    """场景对象轴为 [D, P]，飞机轴为 [E, P]；数量在构造后不变。"""

    num_pursuers: int
    num_dynamic: int

    def __post_init__(self):
        if type(self.num_pursuers) is not int or self.num_pursuers < 0:
            raise ValueError('num_pursuers must be a nonnegative int')
        if type(self.num_dynamic) is not int or self.num_dynamic < 0:
            raise ValueError('num_dynamic must be a nonnegative int')

    @property
    def num_aircraft(self):
        return 1 + self.num_pursuers

    @property
    def num_objects(self):
        return self.num_dynamic + self.num_pursuers

    @property
    def dynamic_objects(self):
        return slice(0, self.num_dynamic)

    @property
    def pursuer_objects(self):
        return slice(self.num_dynamic, self.num_objects)

    @property
    def pursuer_aircraft(self):
        return slice(1, self.num_aircraft)

    def aircraft_prim_name(self,index):
        if type(index) is not int or not 0<=index<self.num_aircraft: raise IndexError('Aircraft index out of range')
        width=len(str(self.num_aircraft-1))
        return f'Hummingbird_{index:0{width}d}'
