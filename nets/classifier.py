import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc    = nn.Linear(4096, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score      = nn.Linear(4096, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        
    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        pool = pool.view(pool.size(0), -1)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 4096]
        #--------------------------------------------------------------#
        fc7 = self.classifier(pool)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)

        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4) # 边框回归
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score = nn.Linear(2048, n_class) # 分类
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        # x是backbone提取的特征图，  rois是RPN给出的候选框，坐标是在原图尺寸上
        # roi_indices是每个候选框对应的图像索引，属于哪个batch，img_size是原图大小
        n, _, _, _ = x.shape  # batch, c, h, w
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1) # 扁平化RoI，按照第0维和第1维进行扁平化，也就是相乘
        roi_indices = torch.flatten(roi_indices, 0, 1) # 属于哪一个batch
        # 例如 原本 [1, 300, 4] → [300, 4]
        # 例如 原本 [1, 300] → [300]
        ##     映射到特征图上，如下
        rois_feature_map = torch.zeros_like(rois) # 创建和rois相同形状的全0张量
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]  #将原图坐标映射到特征图的位置
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)#拼接batch信息，得到[bs,x1,y1..]
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois) # 从特征图中裁剪出对应区域，并池化到固定大小，输出为[300, 1024, 7, 7]
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        fc7 = self.classifier(pool) # 通过全连接层进行特征提取，classifier是resnet50的后半部分,包括layer4和avgpool
                                    # [300, 1024, 7, 7] → [300, 2048, 1, 1] 
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        #--------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1) # [300, 2048, 1, 1] -> [300, 2048],size(0)=300是roi的数量
        # self.cls_loc = nn.Linear(2048, n_class * 4)
        roi_cls_locs    = self.cls_loc(fc7) # nn.Linear只看张量的最后一个维度，如fc7的维度是2048，被送入到全连接层中作为输入
                                            # [300, 2048] -> [300, n_class*4]
        # self.score = nn.Linear(2048, n_class)
        roi_scores      = self.score(fc7) # nn.Linear只看张量的最后一个维度，如fc7的维度是2048，被送入到全连接层中作为输入
                                            # [300, 2048] -> [300, n_class]
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1)) # [300, n_class*4] -> [1, 300, n_class*4]
                                            # 第一个参数是batch_size，第二个参数是自动计算数量/bs，第三个参数是每个roi的边界框回归结果 
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1)) # [300, n_class] -> [1, 300, n_class]
        return roi_cls_locs, roi_scores # 每个RoI边界框回归结果,是修正，本身并不是框；每个RoI的分类分数，每个RoI有全部的类别分数，送入到后面网络取最大的

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

import torch
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork

if __name__ == "__main__":

    # 1. 创建 backbone
    features, classifier = resnet50(pretrained=False)
    # print(features,classifier)
    # 2. 创建 RPN
    rpn = RegionProposalNetwork(in_channels=1024, mode="training")  # layer3 输出通道 1024

    # 3. 创建 RoIHead
    roi_head = Resnet50RoIHead(n_class=21, roi_size=14, spatial_scale=1/16, classifier=classifier)
    # print("ok")
    # 4. 模拟输入
    img = torch.randn(1, 3, 600, 600)
    img_size = (600, 600)

    # 5. backbone 前向传播（到 layer3）
    x = img
    for i, layer in enumerate(features):
        x = layer(x)  # <== 在这里打断点观察 x.shape

    # 6. RPN 前向传播
    rpn_locs, rpn_scores, rois, roi_indices, anchors = rpn(x, img_size, scale=1.0)
    # <== 打断点查看 rpn_locs, rpn_scores, rois, roi_indices, anchors 的形状
    # print("ok")
    print(rpn_locs.shape, rpn_scores.shape, rois.shape, roi_indices.shape, anchors.shape)
    # 7. classifier / RoIHead 前向传播
    roi_cls_locs, roi_scores = roi_head(x, rois, roi_indices, img_size)
    print(roi_cls_locs.shape, roi_scores.shape)
    # print(roi_cls_locs.shape(), roi_scores.shape())
    # <== 在这里打断点查看：
    # roi_cls_locs.shape -> [batch_size, num_rois, n_class*4]
    # roi_scores.shape    -> [batch_size, num_rois, n_class]