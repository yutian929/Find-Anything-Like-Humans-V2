"""Deterministic navigation/point-target cases, including the user's map snapshot."""
from pathlib import Path
import copy
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def scene_from_array(name, cells, resolution, objects):
    return {"name": name, "grid": {"rows": [''.join(row) for row in cells],
            "resolution": resolution, "origin": [0, 0]},
            "objects": [{"id": f'o{i+1}', "position": list(map(float, point))} for i, point in enumerate(objects)]}


def fixtures():
    robot = json.loads((ROOT/'config/robot.example.json').read_text())
    camera = json.loads((ROOT/'config/camera.d435i.json').read_text())
    user = json.loads((Path(__file__).with_name('fixtures')/'user_map_20261014.json').read_text())
    yield 'user_map', user, robot, camera
    room = np.full((16, 24), '.', dtype='<U1'); room[[0,-1],:]='#'; room[:,[0,-1]]='#'
    points = [[.7, .8, .8], [1.4, 3.3, .6], [2.8, 1.2, 1.1], [3.7, 3.1, 1.3],
              [4.7, 1.5, .5], [5.2, 3.2, .9], [3.1, 2., 1.], [1.7, 1.9, .7]]
    yield 'room_small', scene_from_array('room_small', room, .25, points), robot, camera
    separated = np.full((24, 44), '.', dtype='<U1'); separated[[0,-1],:]='#'; separated[:,[0,-1]]='#'; separated[:,21:23]='#'
    targets = [[1.,1.,.8],[2.5,4.7,1.],[4.4,2.6,.6],[6.6,1.1,.7],[8.5,4.8,1.2],[9.6,2.5,.9]]
    yield 'disconnected', scene_from_array('disconnected', separated, .25, targets), robot, camera
    doors = np.full((36, 48), '.', dtype='<U1'); doors[[0,-1],:]='#'; doors[:,[0,-1]]='#'
    doors[:,23:25]='#'; doors[13:23,23:25]='.'; doors[8:13,8:15]='#'; doors[24:28,32:39]='#'
    targets = [[1.,1.,.6],[2.,5.,1.],[3.8,3.2,.8],[5.7,2.,1.3],[7.5,5.6,.7],[8.8,3.4,1.1], [4.7,3.6,.8]]
    yield 'doorway_obstacles', scene_from_array('doorway_obstacles', doors, .2, targets), robot, camera
    # o3 is within range from some stations, but too high for the fixed pitch.
    gaps = [[.7,.8,.8],[2.8,1.2,.5],[3.,2.,2.5],[50.,50.,.8]]
    yield 'height_and_range_gaps', scene_from_array('height_and_range_gaps',room,.25,gaps),robot,camera
    for size in (300,512):
        cells = np.full((size,size),'.',dtype='<U1'); cells[[0,-1],:]='#'; cells[:,[0,-1]]='#'
        cells[size//4:size//2,size//4:size//3]='#'
        cells[size//2:3*size//4,2*size//3:3*size//4]='#'
        rng=np.random.default_rng(size)
        xy=rng.uniform(.7,size*.05-.7,size=(24,2)); z=rng.uniform(.3,1.4,size=24)
        targets=np.column_stack((xy,z)).tolist()
        yield f'large_{size}',scene_from_array(f'large_{size}',cells,.05,targets),robot,camera
    yield 'empty_objects',scene_from_array('empty_objects',room,.25,[]),robot,camera
