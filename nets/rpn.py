import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


class ProposalCreator():
    # 这是一个工具类，负责 从 RPN 的预测结果中生成最终候选框 proposals
    def __init__(
        self, 
        mode, 
        nms_iou             = 0.7,
        n_train_pre_nms     = 12000,
        n_train_post_nms    = 600,
        n_test_pre_nms      = 3000,
        n_test_post_nms     = 300,
        min_size            = 16
    
    ):
        #-----------------------------------#
        #   设置预测还是训练
        #-----------------------------------#
        self.mode               = mode
        #-----------------------------------#
        #   建议框非极大抑制的iou大小
        #-----------------------------------#
        self.nms_iou            = nms_iou
        #-----------------------------------#
        #   训练用到的建议框数量
        #-----------------------------------#
        self.n_train_pre_nms    = n_train_pre_nms
        self.n_train_post_nms   = n_train_post_nms
        #-----------------------------------#
        #   预测用到的建议框数量
        #-----------------------------------#
        self.n_test_pre_nms     = n_test_pre_nms
        self.n_test_post_nms    = n_test_post_nms
        self.min_size           = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms   = self.n_train_pre_nms
            n_post_nms  = self.n_train_post_nms
        else: # 预测
            n_pre_nms   = self.n_test_pre_nms
            n_post_nms  = self.n_test_post_nms

        #-----------------------------------#
        #   将先验框转换成tensor
        #-----------------------------------#
        anchor = torch.from_numpy(anchor).type_as(loc)
        #-----------------------------------#
        #   将RPN网络预测结果转化成建议框
        #-----------------------------------#
        roi = loc2bbox(anchor, loc) # 将特征图上的框转化外原始图上的框
        #-----------------------------------#
        #   防止建议框超出图像边缘
        #-----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])
        
        #-----------------------------------#
        #   建议框的宽高的最小值不可以小于16，过滤太小的框
        #-----------------------------------#
        min_size    = self.min_size * scale
        keep        = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        #-----------------------------------#
        #   将对应的建议框保留下来
        #-----------------------------------#
        roi         = roi[keep, :]
        score       = score[keep]

        #-----------------------------------#
        #   根据得分进行排序，取出建议框
        #-----------------------------------#
        order       = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            order   = order[:n_pre_nms]
        roi     = roi[order, :]
        score   = score[order]

        #-----------------------------------#
        #   对建议框进行非极大抑制
        #   使用官方的非极大抑制会快非常多
        #-----------------------------------#
        keep    = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep        = torch.cat([keep, keep[index_extra]])
        keep    = keep[:n_post_nms]
        roi     = roi[keep]
        return roi # 原图上的框


class RegionProposalNetwork(nn.Module):
    def __init__(
        self, 
        in_channels     = 512, 
        mid_channels    = 512, 
        ratios          = [0.5, 1, 2],
        anchor_scales   = [8, 16, 32], 
        feat_stride     = 16,
        mode            = "training",
    ):
        super(RegionProposalNetwork, self).__init__()
        #-----------------------------------------#
        #   生成9种基础先验框，shape为[9, 4]
        #-----------------------------------------#
        self.anchor_base    = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)
        n_anchor            = self.anchor_base.shape[0]

        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合，整合backbone提取的特征
        #-----------------------------------------#
        self.conv1  = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体，每个anchor输出两个数（包含物体的概率和不包含物体的概率）
        #-----------------------------------------#
        self.score  = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        #-----------------------------------------#
        #   回归预测对先验框进行调整，每个anchor输出4个数（框的坐标）
        #-----------------------------------------#
        self.loc    = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        #-----------------------------------------#
        #   特征点间距步长
        #-----------------------------------------#
        self.feat_stride    = feat_stride
        #-----------------------------------------#
        #   用于对建议框解码并进行非极大抑制，后处理模块，负责NMS，筛选proposals
        #-----------------------------------------#
        self.proposal_layer = ProposalCreator(mode)
        #--------------------------------------#
        #   对FPN的网络部分进行权值初始化
        #--------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合
        #-----------------------------------------#
        x = F.relu(self.conv1(x))
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        #-----------------------------------------#
        rpn_locs = self.loc(x)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体
        #-----------------------------------------#
        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        
        #--------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        #--------------------------------------------------------------------------------------#
        rpn_softmax_scores  = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores       = rpn_softmax_scores[:, :, 1].contiguous()  # anchor是前景的概率
        rpn_fg_scores       = rpn_fg_scores.view(n, -1)

        #------------------------------------------------------------------------------------------------#
        #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        #------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)#特征图每个点生成anchor
        rois        = list()
        roi_indices = list()
        for i in range(n): # 对anchor进行解码 + 筛选，从而得到anchor
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale) #获候选框，经过筛选后的
            batch_index = i * torch.ones((len(roi),)) # 为每个候选框添加一个batch索引，表示这些候选框属于第几张图片
            rois.append(roi.unsqueeze(0)) # 添加一个第一维度bs
            roi_indices.append(batch_index.unsqueeze(0))

        rois        = torch.cat(rois, dim=0).type_as(x) # 拼接bs，也就是将所有的候选框进行拼接  
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor      = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor
        # 第一个参数是RPN对每个anchor的预测调整量，第二个是RPN对每个anchor的分类分数，第三个是最终的建议框proposals；第四个是每个建议框属于哪个batch
        # 第五个是所有的anchor ， 前两个都是在特征图上的，后面三个都是在原图上的

def normal_init(m, mean, stddev, truncated=False):# 卷积层初始化
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

if __name__ == "__main__":
    import torch
    from rpn import RegionProposalNetwork  # 假设 rpn.py 中定义了 RPN
    from utils.anchors import generate_anchor_base, _enumerate_shifted_anchor

    # 1. 模拟 backbone 输出特征图
    batch_size = 1
    C, H, W = 1024, 38, 38
    x = torch.randn(batch_size, C, H, W)

    # 2. 原图大小
    img_size = (600, 600)
    scale = 1.0

    # 3. 创建 RPN
    rpn = RegionProposalNetwork(in_channels=C, mode="training")

    # 4. 逐步调试 RPN 内部
    # conv1
    x_conv1 = torch.relu(rpn.conv1(x))  # <== 在这里打断点

    # loc 分支
    rpn_locs = rpn.loc(x_conv1)
    rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # <== 打断点

    # score 分支
    rpn_scores = rpn.score(x_conv1)
    rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)  # <== 打断点

    # softmax
    rpn_softmax_scores = torch.softmax(rpn_scores, dim=-1)
    rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous().view(batch_size, -1)  # <== 打断点

    # 生成 anchors
    anchor_base = generate_anchor_base()
    anchors = _enumerate_shifted_anchor(anchor_base, rpn.feat_stride, H, W)  # <== 打断点

    # proposals
    rois_list, roi_indices_list = [], []
    for i in range(batch_size):
        rois = rpn.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchors, img_size, scale)
        rois_list.append(rois.unsqueeze(0))
        roi_indices_list.append(torch.ones((len(rois),), dtype=torch.float32).unsqueeze(0))

    rois = torch.cat(rois_list, dim=0)  # <== 打断点
    roi_indices = torch.cat(roi_indices_list, dim=0)  # <== 打断点

    _ = x_conv1, rpn_locs, rpn_scores, rpn_fg_scores, anchors, rois, roi_indices

"""
x_conv1	                                [1, 512, 38, 38]	RPN 第一个 3x3 conv 输出
rpn_locs	                            [1, 12996, 4]	每个 anchor 的坐标回归预测（38x38特征图，每个点9个anchor → 38389=12996）
rpn_scores	                            [1, 12996, 2]	每个 anchor 的前景/背景分数
rpn_softmax_scores	                    [1, 12996, 2]	softmax 后的概率
rpn_fg_scores	                        [1, 12996]	每个 anchor 前景概率
anchor_base	                            [9, 4]	单个特征点的 9 种 anchor 基准框
anchors	                                [12996, 4]	所有特征图点的 anchors（已平铺）
rois	                                [1, 600, 4]（训练模式后）	ProposalCreator 输出的最终建议框，训练模式默认后处理为 600→600，测试模式为 300
roi_indices	                            [1, 600]	每个 proposal 对应的 batch 索引
"""