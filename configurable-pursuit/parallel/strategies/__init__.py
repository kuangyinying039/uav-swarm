"""新增策略：实现 Strategy 子类，再在此注册名字；无需改物理层和 wrapper。"""
from .direct_chase import DirectChase
from .cut_front import CutFront
from .follow_pressure import FollowPressure
from .choke_block import ChokeBlock

REGISTRY={'direct_chase':DirectChase,'cut_front':CutFront,
          'follow_pressure':FollowPressure,'choke_block':ChokeBlock}
