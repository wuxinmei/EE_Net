import enum
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import ImageGrab
import folium
from folium.plugins import HeatMap
from folium.plugins import MousePosition
import math
from folium.plugins import MarkerCluster
from gaze_estimation.utils import compute_angle_error

class FacePartsName(enum.Enum):
    FACE = enum.auto()
    REYE = enum.auto()
    LEYE = enum.auto()

class FaceParts:
    def __init__(self, name: FacePartsName):
        self.name = name
        self.center: Optional[np.ndarray] = None
        self.head_pose_rot: Optional[Rotation] = None
        self.normalizing_rot: Optional[Rotation] = None
        self.normalized_head_rot2d: Optional[np.ndarray] = None
        self.normalized_image: Optional[np.ndarray] = None
        # self.heatmap: Optional[np.ndarray] = None

        self.normalized_gaze_angles: Optional[np.ndarray] = None
        self.normalized_gaze_vector: Optional[np.ndarray] = None
        self.gaze_vector: Optional[np.ndarray] = None
        self.true_gaze_angle = np.array([-0.02749294, -0.22787614])
        
        
    @property
    def heatmap(self) -> None:
        """
        Heatmap of the face part.
        """
        if self.normalized_image is None:
            return None
        # screen = ImageGrab.grab()
        # screen_width, screen_height = screen.size    ## 1920 * 1080
        screen_width = 1920
        screen_height = 1080
        heatmap = folium.Map([40, 116],
                        # tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                        # attr='百度地图',
                        tiles='Stamen Terrain',
                        zoom_start=10,)
        MousePosition().add_to(heatmap)

        gaze_angle = self.vector_to_angle(self.normalized_gaze_vector)
        gaze_angles = []
        gaze_angles.append(gaze_angle)
        print(gaze_angles)

        pixels = [[400, 500], [200, 300], [100, 200], [960, 540]]
        weights = np.repeat(list(range(1, 25)), 1)
        data = []
        # Compare the angle error between adjacent pairs of gaze_angle array
        angle_errors = []
        for i in range(len(gaze_angles) - 1):
            angle_error = compute_angle_error(gaze_angles[i], gaze_angles[i + 1])
            angle_errors.append(angle_error)
        # Filter out gaze angles with angle error less than 2
        filtered_gaze_angles = []
        for i in range(len(angle_errors)):
            if angle_errors[i] >= 2:
                filtered_gaze_angles.append(gaze_angles[i])

        # Replace gaze_angles with filtered_gaze_angles
        gaze_angles = filtered_gaze_angles
        print(gaze_angles)
            

        # Calculate lat and lng for each pixel and store the results in data
        for i, (x, y) in enumerate(pixels):
            lat, lng = self.screen_to_geo(heatmap, screen_width, screen_height, x, y)
            data.append([lat, lng, weights[i]])
        # print(data)
        gaze_data = (data * np.array([[1, 1, 1]])).tolist()

        HeatMap(gaze_data).add_to(heatmap)
        marker_cluster = MarkerCluster().add_to(heatmap)

        for lat, lng, weight in gaze_data:
            # popups = f'lat:{lat:.2f}<br>lon:{lng:.2f}'
            folium.Marker(location=[lat, lng], icon=None, popup=weight,).add_to(marker_cluster)
        # add marker_cluster to map
        heatmap.add_child(marker_cluster)

        return heatmap

    @property
    def distance(self) -> float:
        return np.linalg.norm(self.center)

    def angle_to_vector(self) -> None:
        pitch, yaw = self.normalized_gaze_angles
        self.normalized_gaze_vector = -np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])

    def denormalize_gaze_vector(self) -> None:
        normalizing_rot = self.normalizing_rot.as_matrix()
        # Here gaze vector is a row vector, and rotation matrices are
        # orthogonal, so multiplying the rotation matrix from the right is
        # the same as multiplying the inverse of the rotation matrix to the
        # column gaze vector from the left.
        self.gaze_vector = self.normalized_gaze_vector @ normalizing_rot

    def _angle_to_vector(self, gaze_angle) -> None:
        pitch, yaw = gaze_angle[0], gaze_angle[1]
        normalized_gaze_vector = -np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])
        return normalized_gaze_vector

    def ground_gaze_vector(self) -> np.ndarray:
        normalizing_rot = self.normalizing_rot.as_matrix()
        # Here gaze vector is a row vector, and rotation matrices are
        # orthogonal, so multiplying the rotation matrix from the right is
        # the same as multiplying the inverse of the rotation matrix to the
        # column gaze vector from the left.
        ground_gaze_vector = self._angle_to_vector(self.true_gaze_angle)
        ground_gaze_vector = ground_gaze_vector @ normalizing_rot
        return ground_gaze_vector

    @staticmethod
    def vector_to_angle(vector: np.ndarray) -> np.ndarray:
        assert vector.shape == (3, )
        x, y, z = vector
        pitch = np.arcsin(-y)
        yaw = np.arctan2(-x, -z)
        return np.array([pitch, yaw])
                
    def screen_to_geo(self, map, screen_width, screen_height, point_x, point_y):
        # 创建地图对象
        # map = folium.Map(location=[center_latitude, center_longitude], zoom_start=zoom_start)
        # 通过调用 _repr_html_ 方法生成地图的 HTML 代码
        # map_html = map._repr_html_()qu
        center_latitude = map.location[0]
        center_longitude = map.location[1]
        # 解析 HTML 代码，获取地图的投影信息和像素坐标
        # pattern = r'EPSG(\d{4})'

        projection = map.crs
        zoom = 10

        # 计算屏幕中心点的像素坐标
        center_x = screen_width / 2
        center_y = screen_height / 2

        # 计算屏幕中心点的坐标偏移量
        offset_x = point_x - center_x
        offset_y = point_y - center_y

        # 计算点 B 的经纬度
        if projection == "EPSG3857":
            # 使用 Web Mercator 投影

            meter_x_per_pixel = (156543.03392 * math.cos(math.radians(center_latitude)))/math.pow(2, zoom)
            meter_y_per_pixel = 156543.03392 / math.pow(2, zoom)

            offset_longitude = offset_x * meter_x_per_pixel/111320
            offset_latitude = offset_y * meter_y_per_pixel/100000

            point_longitude = center_longitude + offset_longitude
            point_latitude = center_latitude - offset_latitude
        elif projection == "EPSG4326":
            # 使用 WGS84 投影        
            # delta_longitude = (offset_x / map_x) * 360
            # delta_latitude = (offset_y / map_y) * 180

            # point_longitude = center_longitude + delta_longitude
            # point_latitude = center_latitude - delta_latitude
            scale = 2 ** zoom
            x = point_x * 40075016.68 / scale - 20037508.34
            y = 20037508.34 - point_y * 40075016.68 / scale
            point_longitude, point_latitude = self.mercator_to_wgs84(x, y)
        else:
            raise ValueError("Unsupported projection: {}".format(projection))

        return point_latitude, point_longitude

    
        # def wgs84_to_mercator(self, lon, lat):
        #     r_major = 6378137.000
        #     x = r_major * math.radians(lon)
        #     scale = x/lon
        #     y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + lat * (math.pi/180.0)/2.0)) * scale
        #     return (x, y)

        def mercator_to_wgs84(self, x, y):
            r_major = 6378137.000
            lon = math.degrees(x / r_major)
            lat = math.degrees(2 * math.atan(math.exp(y / r_major)) - math.pi / 2)
            return (lon, lat)

        # def wgs84_to_pixel(self, lon, lat, zoom):
        #     x, y = self.wgs84_to_mercator(lon, lat)
        #     scale = 2 ** zoom
        #     pixel_x = (x + 20037508.34) * scale / 40075016.68
        #     pixel_y = (20037508.34 - y) * scale / 40075016.68
        #     return pixel_x, pixel_y

        # def pixel_to_wgs84(self, pixel_x, pixel_y, zoom):
        #     scale = 2 ** zoom
        #     x = pixel_x * 40075016.68 / scale - 20037508.34
        #     y = 20037508.34 - pixel_y * 40075016.68 / scale
        #     lon, lat = self.mercator_to_wgs84(x, y)
        #     return lon, lat

        # def wgs84_to_screen(self, map, lon, lat):
        #     pixel_x, pixel_y = self.wgs84_to_pixel(lon, lat, map.zoom_start)
        #     screen_width, screen_height = ImageGrab.grab().size
        #     center_x = screen_width / 2
        #     center_y = screen_height / 2
        #     offset_x = (pixel_x - center_x) / 2
        #     offset_y = (pixel_y - center_y) / 2
        #     return center_x + offset_x, center_y + offset_y

        # def screen_to_wgs84(self, map, point_x, point_y):
        #     screen_width, screen_height = ImageGrab.grab().size
        #     center_x = screen_width / 2
        #     center_y = screen_height / 2
        #     pixel_x = (point_x - center_x) * 2 + center_x
        #     pixel_y = (point_y - center_y) * 2 + center_y
        #     lon, lat = self.pixel_to_wgs84(pixel_x, pixel_y, map.zoom_start)
        #     return lon, lat

        # def wgs84_to_screen_offset(self, map, lon, lat):
        #     pixel_x, pixel_y = self.wgs84_to_pixel(lon, lat, map.zoom_start)
        #     screen_width, screen_height = ImageGrab.grab().size
        #     center_x = screen_width / 2
        #     center_y = screen_height / 2
        #     offset_x = (pixel_x - center_x) / 2
        #     offset_y = (pixel_y - center_y) / 2
        #     return offset_x, offset_y


