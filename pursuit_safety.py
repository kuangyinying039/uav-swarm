"""Reason-labelled execution checks and swept geometry for pursuit planning."""
import numpy as np

REASONS = ('boundary', 'building', 'obstacle')


def rejection_causes(env, position):
    c = env.cfg
    reasons, objects = [], []
    lower, upper = [.2, .2, c.min_altitude], [c.grid_size-.2, c.grid_size-.2, c.max_altitude]
    for axis in range(3):
        if position[axis] < lower[axis] or position[axis] > upper[axis]:
            objects.append(f'boundary:{axis}:{"low" if position[axis] < lower[axis] else "high"}')
    if objects:
        reasons.append('boundary')
    for index, (rect, height) in enumerate(zip(env.buildings, env.building_heights)):
        x0, y0, x1, y1 = rect
        m = c.quadrotor_clearance
        if x0-m <= position[0] <= x1+m and y0-m <= position[1] <= y1+m and position[2] <= height+m:
            if 'building' not in reasons:
                reasons.append('building')
            objects.append(f'building:{index}')
    if len(env.obstacles):
        points = np.column_stack([env.obstacles, env.obstacle_altitudes])
        for index in np.flatnonzero(np.linalg.norm(points-position, axis=1) <= c.obstacle_radius):
            if 'obstacle' not in reasons:
                reasons.append('obstacle')
            objects.append(f'obstacle:{index}')
    return reasons, objects


def segment_box(start, end, lower, upper):
    low, high = 0., 1.
    for axis in range(3):
        delta = end[axis]-start[axis]
        if abs(delta) < 1e-10:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
        else:
            a, b = (lower[axis]-start[axis])/delta, (upper[axis]-start[axis])/delta
            low, high = max(low, min(a, b)), min(high, max(a, b))
            if low > high:
                return False
    return True


def clear_segment(env, start, end, extra=0.):
    c = env.cfg
    if np.any(end < [.2, .2, c.min_altitude]) or np.any(end > [c.grid_size-.2, c.grid_size-.2, c.max_altitude]):
        return False
    m = c.quadrotor_clearance+extra
    for rect, height in zip(env.buildings, env.building_heights):
        x0, y0, x1, y1 = rect
        if segment_box(start, end, [x0-m, y0-m, -1e6], [x1+m, y1+m, height+m]):
            return False
    if len(env.obstacles):
        points = np.column_stack([env.obstacles, env.obstacle_altitudes])
        delta = end-start
        fractions = np.clip((points-start)@delta/max(delta@delta, 1e-12), 0, 1)
        if np.any(np.linalg.norm(points-(start+fractions[:, None]*delta), axis=1) <= c.obstacle_radius+extra):
            return False
    return True


class RejectionLog:
    def __init__(self, agents):
        self.counts = dict.fromkeys(REASONS, 0)
        self.streaks = np.zeros((agents, len(REASONS)), dtype=int)
        self.maximum = dict.fromkeys(REASONS, 0)
        self.object_streaks = [{} for _ in range(agents)]
        self.object_maximum = {}

    def update(self, causes, objects):
        for agent, reasons in enumerate(causes):
            for index, reason in enumerate(REASONS):
                self.streaks[agent, index] = self.streaks[agent, index]+1 if reason in reasons else 0
                self.counts[reason] += int(reason in reasons)
                self.maximum[reason] = max(self.maximum[reason], int(self.streaks[agent, index]))
            prior = self.object_streaks[agent]
            self.object_streaks[agent] = {key: prior.get(key, 0)+1 for key in objects[agent]}
            for key, value in self.object_streaks[agent].items():
                name = f'uav{agent}:{key}'
                self.object_maximum[name] = max(self.object_maximum.get(name, 0), value)

    def snapshot(self):
        return {'reason_counts': dict(self.counts), 'max_consecutive_by_reason': dict(self.maximum),
                'max_consecutive_by_object': dict(self.object_maximum)}
