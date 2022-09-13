from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import KittiDetectionDataset


def kitti_viewer():
    root = "data/kitti_detection/kitti_mini"
    label_path = "data/kitti_detection/kitti_mini/label_2"
    dataset = KittiDetectionDataset(root, label_path)

    vi = Viewer(box_type="Kitti")
    vi.set_ob_color_map('gnuplot')

    for i in range(3, len(dataset)):
        P2, V2C, points, image, labels, label_names = dataset[i]
        print('P2 shape:', P2.shape)
        print('v2c shape', V2C.shape)
        print('points shape', points.shape)
        print('image shape', image.shape)
        print('label2s', labels)
        print('labels name', label_names)

        mask = label_names == "Car"
        labels = labels[mask]
        label_names = label_names[mask]

        vi.add_points(points[:, :3], scatter_filed=points[:, 2], color_map_name='viridis')
        vi.add_3D_boxes(labels, box_info=label_names)
        vi.add_3D_cars(labels, box_info=label_names)
        vi.add_image(image)
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)
        vi.show_2D()
        vi.show_3D()


if __name__ == '__main__':
    kitti_viewer()
