"""Defines SilverBadger quadruped constants."""

from etils import epath

from mujoco_playground._src import mjx_env


ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "silver_badger"
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"


def task_to_xml(task_name: str) -> epath.Path:
  return {"flat_terrain": FLAT_TERRAIN_XML}[task_name]


FEET_SITES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
FEET_GEOMS = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]
ROOT_BODY = "trunk"
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
