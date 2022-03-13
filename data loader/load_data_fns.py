import scipy.io as scio
import torch
import numpy as np
from scipy import sparse
import mat73


def load_mat73(data_dir, x_name, y_name, transpose=True):
    mat = mat73.loadmat(data_dir)
    view_num = len(mat[x_name])
    if transpose:
        x = [torch.from_numpy(mat[x_name][i][0].T) for i in range(view_num)]
    else:
        x = [torch.from_numpy(mat[x_name][i][0]) for i in range(view_num)]
    y = torch.from_numpy(np.squeeze(mat[y_name]).astype('int'))
    return x, y


def load_mat(data_dir, x_name, y_name, transpose=False):
    mat = scio.loadmat(data_dir)
    view_num = len(mat[x_name][0])
    if transpose:
        x = [torch.tensor(mat[x_name][0][i].astype(np.float).T, dtype=torch.float) for i in range(view_num)]
    else:
        x = [torch.tensor(mat[x_name][0][i].astype(np.float), dtype=torch.float) for i in range(view_num)]
    y = torch.from_numpy(np.squeeze(mat[y_name]).astype('int'))
    return x, y


def load_scene15(data_dir):
    mat = scio.loadmat(data_dir)
    x = [
        torch.from_numpy(mat['X'][0][0].astype('float32')),
        torch.from_numpy(mat['X'][0][1].astype('float32'))
    ]
    y = torch.from_numpy(np.squeeze(mat['Y']).astype('int'))
    return x, y, x[0].shape[0]


def load_landuse21(data_dir):
    mat = scio.loadmat(data_dir)
    x = [
        torch.from_numpy(sparse.csr_matrix(mat['X'][0, 0]).A),
        torch.from_numpy(sparse.csr_matrix(mat['X'][0, 1]).A),
        torch.from_numpy(sparse.csr_matrix(mat['X'][0, 2]).A)
    ]

    y = torch.from_numpy(np.squeeze(mat['Y']).astype('int'))

    return x, y, x[0].shape[0]


def load_caltech101_20(data_dir):
    x, y = load_mat(data_dir, "X", "truth", True)
    return x, y, x[0].shape[0]


def load_mnist(data_dir):
    x, y = load_mat(data_dir, "X", "truth", True)
    return x, y, x[0].shape[0]


def load_awa(data_dir):
    x, y = load_mat73(data_dir, "X", "Y")
    return x, y, x[0].shape[0]


def load_bbcsport(data_dir):
    mat = scio.loadmat(data_dir)
    x = [
        torch.from_numpy(sparse.csr_matrix(mat['X'][0, 0]).A),
        torch.from_numpy(sparse.csr_matrix(mat['X'][0, 1]).A),
    ]

    y = torch.from_numpy(np.squeeze(mat['Y']).astype('int'))

    return x, y, x[0].shape[0]


def load_caltech101_7(data_dir):
    x, y = load_mat(data_dir, "X", "Y")
    return x, y, x[0].shape[0]


def load_caltech101_all(data_dir):
    x, y = load_mat73(data_dir, "X", "Y")
    return x, y, x[0].shape[0]


def load_handwritten(data_dir):
    x, y = load_mat(data_dir, "X", "Y")
    return x, y, x[0].shape[0]


def load_mirflickr(data_dir):
    pass


def load_nus_wide_obj(data_dir):
    x, y = load_mat73(data_dir, "X", "Y")
    return x, y, x[0].shape[0]


def load_orl_mtv(data_dir):
    x, y = load_mat(data_dir, "X", "gt", True)
    return x, y, x[0].shape[0]


def load_sun_rgbd(data_dir):
    x, y = load_mat73(data_dir, "X", "Y")
    return x, y, x[0].shape[0]


def load_web_kb(data_dir):
    x, y = load_mat(data_dir, "X", "gnd")
    return x, y, x[0].shape[0]


def load_youtube_video(data_dir):
    x, y = load_mat73(data_dir, "X", "Y")
    return x, y, x[0].shape[0]


def load_voc(data_dir):
    a = np.load(data_dir)
    x = [torch.from_numpy(a['view_0']), torch.from_numpy(a['view_1'])]
    return x, torch.from_numpy(a['labels']), a['labels'].shape[0]


def load_rgbd(data_dir):
    a = np.load(data_dir)
    x = [torch.from_numpy(a['view_0']), torch.from_numpy(a['view_1'])]
    return x, torch.from_numpy(a['labels']), a['labels'].shape[0]


def load_CiteSeer(data_dir):
    CiteSeer = scio.loadmat(data_dir)
    x = [torch.from_numpy(CiteSeer['fea'][0][i].todense()) for i in range(2)]
    y = torch.from_numpy(np.squeeze(CiteSeer['gt']).astype('int'))

    return x, y, x[0].shape[0]


def load_bbcsport_2(data_dir):
    bbcsport_2 = scio.loadmat(data_dir)
    x = [torch.from_numpy(bbcsport_2['X1']).T, torch.from_numpy(bbcsport_2['X2']).T]
    y = torch.from_numpy(np.squeeze(bbcsport_2['truth']).astype('int'))
    return x, y, x[0].shape[0]


def load_bbcsport_3(data_dir):
    bbcsport_3 = scio.loadmat(data_dir)
    x = [torch.from_numpy(bbcsport_3['X'][0][i]).T for i in range(3)]
    y = torch.from_numpy(np.squeeze(bbcsport_3['truth']).astype('int'))
    return x, y, x[0].shape[0]


def load_100Leaves(data_dir):
    Leaves = scio.loadmat(data_dir)
    x = [torch.from_numpy(Leaves['X'][0][i]) for i in range(3)]
    y = torch.from_numpy(np.squeeze(Leaves['Y']).astype('int'))
    return x, y, x[0].shape[0]

