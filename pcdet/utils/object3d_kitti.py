import numpy as np

#返回标签信息的文件路径
def get_objects_from_label(label_file):
    #objects = get_objects_from_label('../data/kitti/training/label_2/000000.txt')
    #objects[0].__dict__示例
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

#返回目标检测类别，并将类别信息标为id形式
def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]#类别属性
        self.cls_id = cls_type_to_id(self.cls_type)#类别id
        self.truncation = float(label[1])#物体的截断程度，该属性的值通常为一个介于0和1之间的浮点数，表示截断的程度。值为0表示物体完全可见，值为1表示物体完全被图像边缘截断
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown 遮挡程度
        self.alpha = float(label[3])#物体的观察角度
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)#二维边界框，左上角和右下角的坐标
        self.h = float(label[8])#物体的高度
        self.w = float(label[9])#物体的宽度
        self.l = float(label[10])#物体的长度
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)#物体在世界坐标系中的位置
        self.dis_to_cam = np.linalg.norm(self.loc)#计算loc属性的欧几里得范数（即到相机的距离）
        self.ry = float(label[14])#物体的朝向角度
        self.score = float(label[15]) if label.__len__() == 16 else -1.0#物体的置信度分数，否则为-1
        self.level_str = None
        self.level = self.get_kitti_obj_level()
    
    #用于根据一些条件判断来确定对象的级别
    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    #用于生成对象的三维角点表示
    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    #将对象类型转换为字符串
    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    #用于将对象的属性转换为符合KITTI数据集格式的字符串表示
    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str
