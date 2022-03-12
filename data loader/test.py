from MVDataBaseClass import DataShell
from load_data_fns import *


def show_each_view_shape(data: DataShell):
    view_num = len(data.x)
    for i in range(view_num):
        print(f"View {i} shape is:", data.x[i].shape)
    print(f"Label shape is:  {data.y.shape}")


# # example 1
# scene_15 = DataShell("./data/Scene-15.mat", load_scene15, missing=0.1)
# print("---- scene_15")
# show_each_view_shape(scene_15)
#
# # example 2
# land_use_21 = DataShell("./data/LandUse-21.mat", load_landuse21, missing=0.5)
# print("---- land_use_21")
# show_each_view_shape(land_use_21)
#
# # example 3
# caltech_101_20 = DataShell("data/Caltech101-20.mat", load_caltech101_20, missing=0.1)
# print("---- caltech_101_20")
# show_each_view_shape(caltech_101_20)
#
# # example 4
# mnist = DataShell("./data/mnist.mat", load_mnist, missing=0.1)
# print("---- mnist")
# show_each_view_shape(mnist)
#
# # example 5
# awa = DataShell("./data/AwA_fea.mat", load_awa, missing=0.1)
# print("---- awa")
# show_each_view_shape(awa)
#
# # example 6
# bbcsport = DataShell("./data/bbcsport.mat", load_bbcsport, missing=0.1)
# print("---- bbcsport")
# show_each_view_shape(bbcsport)
#
# # example 7
# caltech101_7 = DataShell("./data/Caltech101-7.mat", load_caltech101_7, missing=0.1)
# print("---- caltech101_7")
# show_each_view_shape(caltech101_7)
#
# # example 8
# caltech101_all = DataShell("./data/Caltech101-all_fea.mat", load_caltech101_all, missing=0.1)
# print("---- caltech101-all")
# show_each_view_shape(caltech101_all)
#
# # example 9
# handwritten = DataShell("./data/handwritten.mat", load_handwritten, missing=0.1)
# print("---- handwritten")
# show_each_view_shape(handwritten)
#
# # example 10. This has not been implemented !!!
# mirflickr = DataShell("./data/mirflickr.mat", load_mirflickr, missing=0.1)
# print("---- mirflickr")
# show_each_view_shape(mirflickr)
#
# # example 11
# nus_wide_obj = DataShell("./data/NUS-WIDE-OBJECT_fea.mat", load_nus_wide_obj, missing=0.1)
# print("---- NUS-WIDE-OBJECT")
# show_each_view_shape(nus_wide_obj)
#
# # example 12
# orl_mtv = DataShell("./data/ORL_mtv.mat", load_orl_mtv, missing=0.1)
# print("---- rl_mtv")
# show_each_view_shape(orl_mtv)
#
# # example 13
# sun_rgbd = DataShell("./data/SUNRGBD_fea.mat", load_sun_rgbd, missing=0.1)
# print("---- sun_rgbd")
# show_each_view_shape(sun_rgbd)
#
# # example 14
# web_kb = DataShell("./data/WebKB.mat", load_web_kb, missing=0.1)
# print("---- web-kb")
# show_each_view_shape(web_kb)
#
# # example 15
# youtube_video = DataShell("./data/YoutubeVideo_sel_fea.mat", load_youtube_video, missing=0.1)
# print("---- youtube-video")
# show_each_view_shape(youtube_video)
#
# # example 16
# voc = DataShell("./data/voc.npz", load_voc, missing=0)
# print("---- voc")
# show_each_view_shape(voc)
#
# # example 17
# rgbd = DataShell("./data/rgbd.npz", load_rgbd, missing=0.1)
# print("---- rgbd")
# show_each_view_shape(rgbd)
#
#  # example 18
cite = DataShell('./data/CiteSeer.mat', load_CiteSeer, missing=0)
print("---- cite")
show_each_view_shape(cite)
