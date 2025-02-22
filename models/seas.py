

import torch
from torch import Tensor
from torch.nn import (
    functional as F, Module, Sequential, Identity, BatchNorm1d, AdaptiveMaxPool2d, AdaptiveAvgPool2d, LayerNorm,
    Dropout
)
from torchvision.models.detection._utils import BoxCoder, _box_loss
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from typing import Tuple, List, Union, Dict, Optional, Any
from collections import OrderedDict

from .backbones.resnet import ResNetBackbone, ResNetHead
from .backbones.convnext import ConvNeXtBackbone, ConvNeXtHead
from .backbones.solider import SOLIDERBackbone, SOLIDERHead
from .losses.boim import BidirectionalOnlineInstanceMatchingLoss
from .losses.bnr import BackgroundNoiseReductionLoss
from .modules.box_predictor import BoxPredictor
from .modules.momentum_batch_norm import MomentumBatchNorm
from .modules.heads import TripleConvHead
from .modules.position_embedder import LearnablePositionEmbedder1D, LearnablePositionEmbedder2D
from .modules.parallel_decoder import ParallelDecoder


#from models.embedder import MultiGranularityEmbedding, GlobalFeatureEmbedding, Embedder
#####from models.embedder import FusionEmbedding, GlobalFeatureEmbedding, Embedder   #AdaptiveHighPassEmbedding

from models.embedder import ResBlock, GlobalFeatureEmbedding, Embedder      #MultiHeadAttentionEmbedder   #Embedder#  MultiHeadAttentionEmbedder  #Embedder
####修改代码地方***
from models.denoiser import CrossAttention, LinearProjection

from utils.detection import Sampler, compute_iou, compute_centerness
from utils.general import normalize_weight_zero_bias, Pack

import pdb

class Initializer:
    @staticmethod
    def initialize_v1(module: Module) -> None:
        if hasattr(module, 'load_pretrained_weights'):
            module.load_pretrained_weights()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            raise ValueError

    @staticmethod
    def initialize_v2(module: Module) -> None:
        if hasattr(module, 'load_pretrained_weights'):
            module.load_pretrained_weights()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            module.apply(normalize_weight_zero_bias)
        else:
            raise ValueError

    def __init__(self, mode: str):
        self._initialize = getattr(Initializer, f'initialize_{mode}')

    def __call__(self, module: Module) -> None:
        self._initialize(module)


class SEAS(Module):
    def __init__(self, cfg):
        super().__init__()
        initialize = Initializer(cfg.MODEL.PARAM_INIT)
        ''' build transform ---------------------------------------------------------------------------------------- '''
        self.transform = GeneralizedRCNNTransform(
            min_size=cfg.MODEL.TRANSFORM.MIN_SIZE,
            max_size=cfg.MODEL.TRANSFORM.MAX_SIZE,
            image_mean=cfg.MODEL.TRANSFORM.IMG_MEAN,
            image_std=cfg.MODEL.TRANSFORM.IMG_STD,
        )

        ''' build backbone ----------------------------------------------------------------------------------------- '''
        backbone_type, backbone_head_type = {
            'ResNet': (ResNetBackbone, ResNetHead),
            'ConvNeXt': (ConvNeXtBackbone, ConvNeXtHead),
            'SOLIDER': (SOLIDERBackbone, SOLIDERHead),
        }[cfg.MODEL.BACKBONE]
        self.backbone = backbone_type(
            return_layers_idx=[3],
        )
        initialize(self.backbone)

        ''' build rpn ---------------------------------------------------------------------------------------------- '''
        anchor_generator = AnchorGenerator(
            sizes=cfg.MODEL.RPN.ANCHOR_SIZE,
            aspect_ratios=cfg.MODEL.RPN.ANCHOR_RATIO,
        )
        rpn_head = RPNHead(
            in_channels=self.backbone.out_channels_list[-1],
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=dict(
                training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN,
                testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
            ),
            post_nms_top_n=dict(
                training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN,
                testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
            ),
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )

        ''' build Faster-RCNN RoI heads ---------------------------------------------------------------------------- '''
        sampler = Sampler(
            fg_thresh=cfg.MODEL.DETECTION.SAMPLE.POS_THRESH,
            bg_thresh=cfg.MODEL.DETECTION.SAMPLE.NEG_THRESH,
            batch_size_per_image=cfg.MODEL.DETECTION.SAMPLE.BATCH_SIZE,
            positive_fraction=cfg.MODEL.DETECTION.SAMPLE.POS_FRAC,
            keep_positive_fraction=True,
            bg_label=0,
            append_gt_boxes=True,
        )
        box_coder = BoxCoder(
            weights=(10.0, 10.0, 5.0, 5.0),
        )
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=self.backbone.out_feat_names,
            output_size=cfg.MODEL.DETECTION.FEAT_MAP_SIZE,
            sampling_ratio=2,
        )
        box_head = backbone_head_type(
            down_sampling=True,
        )
        initialize(box_head)
        box_predictor = BoxPredictor(
            in_channels=box_head.out_channels,
            num_classes=2,
            quality=cfg.MODEL.DETECTION.QUALITY,
            batch_norm=True,
        )
        initialize(box_predictor)
        self.detection_roi_heads = FastRCNNRoIHeads(
            # modules
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            # utils
            sampler=sampler,
            box_coder=box_coder,
            # testing parameters
            score_thresh=cfg.MODEL.DETECTION.POST_PROCESS.SCORE_THRESH,
            nms_thresh=cfg.MODEL.DETECTION.POST_PROCESS.NMS_THRESH,
            detections_per_img=cfg.MODEL.DETECTION.POST_PROCESS.DETECTIONS_PER_IMAGE,
            # loss type
            box_reg_loss_type=cfg.MODEL.DETECTION.BOX_REG_LOSS_TYPE,
            quality_loss_type=cfg.MODEL.DETECTION.QUALITY_LOSS_TYPE,
        )

        ''' build background modulation network -------------------------------------------------------------------- '''
        bmn_enhancer_head = backbone_head_type(down_sampling=True)
        initialize(bmn_enhancer_head)
        bmn_enhancer = Pack(
            body=bmn_enhancer_head,
            in_feat_name=None,
            output_both_ends=True,
        )

        num_feat_map_used = {'Downsample': 1, 'OriginalAndDownsample': 2}[cfg.MODEL.REID.FEAT_MAP_USED]
        switch = cfg.MODEL.REID.EMBEDDING
        if switch == 'MGE':  # 这里是测试时候 是选择MGE还是选择 GFE来进行


            ##通道注意力的方式效果不好
            # bmn_embedder = Embedder(  # 调用的是embedder这个函数
            #     in_feat_names=bmn_enhancer.out_feat_names[-num_feat_map_used:],
            #     in_channels_list=bmn_enhancer.out_channels_list[-num_feat_map_used:],
            #     dim_out=cfg.MODEL.REID.DIM_IDENTITY,  # 输出1024维度
            #     emb_type=ResBlock,  # 暂时不添加编码器
            #     reduction=16,  #
            #     extra_cfg=dict(
            #         mode=['train'],
            #     )
            # )


            #多头注意力融合特征的 8头效果一般
            # bmn_embedder = MultiHeadAttentionEmbedder(  # 调用的是embedder这个函数
            #     in_feat_names=bmn_enhancer.out_feat_names[-num_feat_map_used:],
            #     in_channels_list=bmn_enhancer.out_channels_list[-num_feat_map_used:],
            #     dim_out=cfg.MODEL.REID.DIM_IDENTITY,  # 输出1024维度
            #     emb_type=ResBlock,  # 暂时不添加编码器
            #     num_heads=8,
            #     extra_cfg=dict(
            #        mode=['train'],
            #     )
            # )

            bmn_embedder = Embedder(  # 调用的是embedder这个函数
                in_feat_names=bmn_enhancer.out_feat_names[-num_feat_map_used:],
                in_channels_list=bmn_enhancer.out_channels_list[-num_feat_map_used:],
                dim_out=cfg.MODEL.REID.DIM_IDENTITY,  # 输出1024维度
                emb_type=ResBlock,
                extra_cfg=dict(
                    mode=['train'],
                ),
            )

                # emb_type=FusionEmbedding, #AdaptiveHighPassEmbedding,  # Use Adaptive High-Pass Embedding
                # extra_cfg=dict(
                #     base_radius=30,  # 直接传入数值
                #     scale_factor=0.1,  # 直接传入数值
                # ),


                ###修改代码*** 注释了
                # emb_type=MultiGranularityEmbedding,  # 多粒度特征提取
                # extra_cfg=dict(
                #     num_branches=cfg.MODEL.REID.EMBEDDING_MGE.NUM_BRANCHES,
                #     drop_path=cfg.MODEL.REID.EMBEDDING_MGE.DROP_PATH,
                # ),

        elif switch == 'GFE':
            bmn_embedder = Embedder(
                in_feat_names=bmn_enhancer.out_feat_names[-num_feat_map_used:],
                in_channels_list=bmn_enhancer.out_channels_list[-num_feat_map_used:],
                dim_out=cfg.MODEL.REID.DIM_IDENTITY,
                emb_type=GlobalFeatureEmbedding,
            )
        else:
            raise ValueError
        initialize(bmn_embedder)

        switch = cfg.MODEL.REID.BNR.MAPPING
        if switch == 'MBN':
            mapping = MomentumBatchNorm(1, momentum=cfg.MODEL.REID.BNR.MAPPING_MBN_MOMENTUM)
        elif switch == 'BN':
            mapping = BatchNorm1d(1, affine=True, track_running_stats=False)
        elif switch == 'Identity':
            mapping = Identity()
        else:
            raise ValueError
        bnr_loss = BackgroundNoiseReductionLoss(mapping=mapping)

        ''' build foreground modulation network -------------------------------------------------------------------- '''
        if cfg.MODEL.REID.EXTRACTOR and cfg.MODEL.REID.DENOISER is not None:
            ######  这里选用  _C.MODEL.REID.EXTRACTOR = 'TripleConvHead'
            switch = cfg.MODEL.REID.EXTRACTOR
            if switch == 'TripleConvHead':  ###三个卷积 BN Relu 噪声提取器
                fmn_extractor_head = TripleConvHead(in_channels=self.backbone.out_channels_list[-1])
            elif switch == 'BackboneHead':
                fmn_extractor_head = backbone_head_type(down_sampling=True)
            elif switch == 'Identity':
                fmn_extractor_head = Identity()
                fmn_extractor_head.in_channels = self.backbone.out_channels_list[-1]
                fmn_extractor_head.out_channels = self.backbone.out_channels_list[-1]
                fmn_extractor_head.reset_parameters = lambda: None
            else:
                raise ValueError
            initialize(fmn_extractor_head)
            fmn_extractor = Pack(
                body=fmn_extractor_head,
                in_feat_name=self.backbone.out_feat_names[-1],
                output_both_ends=False,
            )

            ##########这里选用 _C.MODEL.REID.DENOISER = 'CrossAttention'
            switch = cfg.MODEL.REID.DENOISER
            if switch == 'CrossAttention':
                pool = {   #这里选用最大池化
                    'Max': AdaptiveMaxPool2d, 'Avg': AdaptiveAvgPool2d
                }[cfg.MODEL.REID.DENOISER_CA.POOL_TYPE](output_size=cfg.MODEL.REID.DENOISER_CA.POOL_SIZE)

                sub_switch = cfg.MODEL.REID.DENOISER_CA.POS_EMB
                if sub_switch == 'Learnable1D':
                    embedder = LearnablePositionEmbedder1D(size=pool.output_size, dim=fmn_extractor.out_channels)
                elif sub_switch == 'Learnable2D': #选用2d的 位置编码
                    embedder = LearnablePositionEmbedder2D(size=pool.output_size, dim=fmn_extractor.out_channels)
                elif sub_switch is None:
                    embedder = None
                else:
                    raise ValueError
                ###添加layer Norm
                norm = LayerNorm(fmn_extractor.out_channels) if cfg.MODEL.REID.DENOISER_CA.LAYER_NORM else None

                decoder = ParallelDecoder(  ###modules包下的 进行编码
                    dim=bmn_embedder.dim_out,  #这个是输入的q的维度 1024  也就是把输入特征为bmn_embedder
                    num_heads=cfg.MODEL.REID.DENOISER_CA.PD_NUM_HEADS,
                    dim_ffn=cfg.MODEL.REID.DENOISER_CA.PD_DIM_FFN,
                    dropout=cfg.MODEL.REID.DENOISER_CA.PD_DROPOUT,
                    dim_memory=fmn_extractor.out_channels  #这里是输入fmn_extractor 也就是场景提取的特征 然后是其维度 2048
                )

                fmn_denoiser = CrossAttention(
                    pool=pool,
                    embedder=embedder,
                    norm=norm,
                    decoder=decoder,  ###这个的decoder就是 对噪声图进行 ParallelDecoder 输出的特征
                )

               ###这里上下 是两种如何进行融合的操作
            elif switch == 'LinearProjection':  #或者选用 线性映射
                fmn_denoiser = LinearProjection(
                    dim=bmn_embedder.dim_out,
                    memery_channels=fmn_extractor.out_channels
                )
            else:
                raise ValueError
            initialize(fmn_denoiser)
        else:
            fmn_extractor, fmn_denoiser = None, None
        ''' build reid roi head ------------------------------------------------------------------------------------ '''
        sampler = Sampler(
            fg_thresh=cfg.MODEL.REID.SAMPLE.POS_THRESH,
            bg_thresh=cfg.MODEL.REID.SAMPLE.NEG_THRESH,
            batch_size_per_image=cfg.MODEL.REID.SAMPLE.BATCH_SIZE,
            positive_fraction=cfg.MODEL.REID.SAMPLE.POS_FRAC,
            keep_positive_fraction=True,
            bg_label=0,
            append_gt_boxes=True,
        )
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=self.backbone.out_feat_names,
            output_size=cfg.MODEL.REID.FEAT_MAP_SIZE,
            sampling_ratio=2,
        )
        reid_loss = BidirectionalOnlineInstanceMatchingLoss(
            dim=cfg.MODEL.REID.DIM_IDENTITY,
            lut_size=cfg.MODEL.REID.LOSS.LUT_SIZE,
            cq_size=cfg.MODEL.REID.LOSS.CQ_SIZE,
            momentum=cfg.MODEL.REID.LOSS.MOMENTUM,
            scalar=cfg.MODEL.REID.LOSS.SCALAR,
            margin=cfg.MODEL.REID.LOSS.MARGIN,
            weight_softmax=cfg.MODEL.REID.LOSS.WEIGHT_SOFTMAX,
            weight_triplet=cfg.MODEL.REID.LOSS.WEIGHT_TRIPLET,
        )
        self.reid_roi_heads = ReIdRoIHeads(
            box_roi_pool=box_roi_pool,
            enhancer=bmn_enhancer,
            embedder=bmn_embedder,
            extractor=fmn_extractor,
            denoiser=fmn_denoiser,
            bnr_loss=bnr_loss,
            reid_loss=reid_loss,
            sampler=sampler,
        )
        ''' set loss weights --------------------------------------------------------------------------------------- '''
        self.loss_weights = {
            'rpn_cls':  cfg.MODEL.LOSS_WEIGHT.RPN_CLS,
            'rpn_reg':  cfg.MODEL.LOSS_WEIGHT.RPN_REG,
            'prop_cls': cfg.MODEL.LOSS_WEIGHT.PROPOSAL_CLS,
            'prop_reg': cfg.MODEL.LOSS_WEIGHT.PROPOSAL_REG,
            'prop_qlt': cfg.MODEL.LOSS_WEIGHT.PROPOSAL_QLT,
            'box_bnr':  cfg.MODEL.LOSS_WEIGHT.BOX_BNR,
            'box_reid': cfg.MODEL.LOSS_WEIGHT.BOX_REID,
        }

    def forward(
            self,
            images: List[Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None,
            use_gt_as_det: bool = False
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        # get original image size for post process, and preprocess
        original_img_sizes = [img.shape[-2:] for img in images] if not self.training else None
        images, targets = self.transform(images, targets)
        # extract feature map
        feats = self.backbone(images.tensors)
        #pdb.set_trace()
        # detection
        if not use_gt_as_det:
            det_targets = [
                {'boxes': tgt['boxes'], 'labels': torch.ones_like(tgt['labels'])} for tgt in targets
            ] if targets else None
            props, rpn_losses = self.rpn(images, feats, det_targets)
            boxes, scores, _, detector_losses = self.detection_roi_heads(feats, props, images.image_sizes, det_targets)
            if self.training:
                boxes = [torch.cat([boxes_in_img, props_in_img]) for boxes_in_img, props_in_img in zip(boxes, props)]
        else:  # using ground truth as the detection result
            assert targets is not None
            rpn_losses = {}
            boxes = [target['boxes'] for target in targets]
            scores = [boxes_in_img.new_ones(len(boxes_in_img)) for boxes_in_img in boxes]
            detector_losses = {}
        # re-identity
        identities, reid_losses = self.reid_roi_heads(feats, boxes, images.image_sizes, targets)
        # post process
        if self.training:  # wrap losses
            losses = rpn_losses
            losses.update(detector_losses)
            losses.update(reid_losses)
            for old_nm, new_nm in [  # change name
                ('loss_objectness', 'rpn_cls'),
                ('loss_rpn_box_reg', 'rpn_reg'),
            ]:
                if old_nm in losses:
                    losses[new_nm] = losses.pop(old_nm)
            for loss_nm in losses:  # multiply weight
                losses[loss_nm] *= self.loss_weights.get(loss_nm, 1.0)
            return losses
        else:  # wrap result
            results = [
                {'boxes': boxes_in_img, 'scores': scores_in_img, 'identities': identities_in_img}
                for boxes_in_img, scores_in_img, identities_in_img in zip(boxes, scores, identities)
            ]
            results = self.transform.postprocess(results, images.image_sizes, original_img_sizes)
            return results


class FastRCNNRoIHeads(Module):
    def __init__(
            self,
            box_roi_pool: MultiScaleRoIAlign,
            box_head: Module,
            box_predictor: Module,
            # utils
            box_coder: BoxCoder,
            sampler: Sampler,
            # evaluating parameters
            score_thresh: float,
            nms_thresh: float,
            detections_per_img: int,
            # loss type
            box_reg_loss_type: str = 'smooth_l1',
            quality_loss_type: str = 'iou'
    ):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.box_coder = box_coder
        self.sampler = sampler
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.box_reg_loss_type = box_reg_loss_type
        if quality_loss_type == 'centerness':
            self.quality_encoder = compute_centerness
        elif quality_loss_type == 'iou':
            self.quality_encoder = compute_iou
        else:
            assert False

    def compute_losses(
            self, prop_clss: Tensor, prop_regs: Tensor, prop_qlts: Optional[Tensor],
            props: List[Tensor], truths: List[Tensor], labels: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        props, truths, labels = torch.cat(props), torch.cat(truths), torch.cat(labels)
        loss_cls = F.cross_entropy(
            input=prop_clss,
            target=labels,
        )
        fg_idxs = torch.nonzero(labels > 0).view(-1)
        loss_reg = _box_loss(
            type=self.box_reg_loss_type,
            box_coder=self.box_coder,
            anchors_per_image=props[fg_idxs],
            matched_gt_boxes_per_image=truths[fg_idxs],
            bbox_regression_per_image=prop_regs[fg_idxs],
            cnf=dict(beta=1),
        ) / labels.numel()
        loss_qlt = F.binary_cross_entropy_with_logits(
            input=prop_qlts[fg_idxs].view(-1),
            target=self.quality_encoder(truths[fg_idxs], prop_regs[fg_idxs]),
            reduction='sum',
        ) / labels.numel() if prop_qlts is not None else None
        return loss_cls, loss_reg, loss_qlt

    @torch.no_grad()
    def postprocess(
            self, prop_clss: Tensor, prop_regs: Tensor, prop_qlts: Optional[Tensor],
            props: List[Tensor], image_size: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        num_imgs, num_classes = len(props), prop_clss.shape[-1]
        num_props_per_img = list(map(len, props))
        boxes, scores, classes = [], [], []
        for prop_clss_in_img, prop_regs_in_img, prop_qlts_in_img, props_in_img, img_sz in zip(
            prop_clss.split(num_props_per_img),
            prop_regs.split(num_props_per_img),
            prop_qlts.split(num_props_per_img) if prop_qlts is not None else [None] * num_imgs,
            props, image_size
        ):
            boxes_in_img = self.box_coder.decode_single(prop_regs_in_img, props_in_img)
            boxes_in_img = box_ops.clip_boxes_to_image(boxes_in_img, img_sz)  # clip the box so that it is all inside
            keep = box_ops.remove_small_boxes(boxes_in_img, min_size=1e-2)   # remove the small boxes
            boxes_in_img = boxes_in_img[keep]
            if not self.training:  # inference
                prop_clss_in_img = prop_clss_in_img[keep]
                prop_qlts_in_img = prop_qlts_in_img[keep] if prop_qlts_in_img is not None else None

                raw_scores_in_img = F.softmax(prop_clss_in_img, dim=-1)[:, 1:]  # remove background
                retained_score_idxs = torch.nonzero(raw_scores_in_img.view(-1) > self.score_thresh).view(-1)

                scores_in_img = (F.sigmoid(prop_qlts_in_img) if prop_qlts_in_img is not None else 1) * raw_scores_in_img
                scores_in_img = scores_in_img.view(-1)[retained_score_idxs]

                boxes_in_img = boxes_in_img[retained_score_idxs // (num_classes - 1)]  # do not have background
                classes_in_img = (retained_score_idxs % (num_classes - 1)) + 1  # 0 is background

                keep = box_ops.batched_nms(boxes_in_img, scores_in_img, classes_in_img, self.nms_thresh)
                keep = keep[:self.detections_per_img]

                boxes_in_img = boxes_in_img[keep]
                scores.append(scores_in_img[keep])
                classes.append(classes_in_img[keep])
            boxes.append(boxes_in_img)
        return boxes, scores, classes

    def forward(
            self,
            feats: Dict[str, Tensor],
            props: List[Tensor],
            image_sizes: List[Tuple[int, int]],
            targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], Optional[Dict[str, Tensor]]]:
        if self.training:
            props, truths, labels, _ = self.sampler(props, targets)
        else:
            truths, labels = None, None

        prop_feats = self.box_roi_pool(feats, props, image_sizes)
        prop_feats = self.box_head(prop_feats)
        prop_clss, prop_regs, prop_qlts = self.box_predictor(prop_feats)

        if self.training:
            loss_cls, loss_reg, loss_qlt = self.compute_losses(prop_clss, prop_regs, prop_qlts, props, truths, labels)
            losses = {'prop_cls': loss_cls, 'prop_reg': loss_reg}
            if loss_qlt is not None:
                losses['prop_qlt'] = loss_qlt
        else:
            losses = None

        boxes, scores, classes = self.postprocess(
            prop_clss.detach(), prop_regs.detach(), prop_qlts.detach() if prop_qlts is not None else None,
            props, image_sizes
        )

        return boxes, scores, classes, losses


class ReIdRoIHeads(Module):
    def __init__(
            self,
            box_roi_pool: MultiScaleRoIAlign,
            enhancer: Module,
            embedder: Module,
            extractor: Module,
            denoiser: Module,
            bnr_loss: Module,
            reid_loss: Module,
            sampler: Sampler,
    ):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.enhancer = enhancer
        self.embedder = embedder
        self.extractor = extractor
        self.denoiser = denoiser
        self.bnr_loss = bnr_loss
        self.reid_loss = reid_loss
        self.sampler = sampler

    def forward(
            self,
            feats: Dict[str, Tensor],
            boxes: List[Tensor],
            image_sizes: List[Tuple[int, int]],
            targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Optional[List[Tensor]], Optional[Dict[str, Tensor]]]:
        if self.training:
            boxes, _, labels, _ = self.sampler(boxes, targets)
        else:
            labels = None

        num_boxes_per_img = list(map(len, boxes))
        box_feats = self.box_roi_pool(feats, boxes, image_sizes)
        box_feats = self.enhancer(box_feats)
        box_embeddings = self.embedder(box_feats)
        if self.extractor and self.denoiser is not None:
            box_identities = self.denoiser(box_embeddings, self.extractor(feats), num_boxes_per_img)
        else:
            box_identities = box_embeddings
        box_identities = F.normalize(box_identities, dim=-1)

        if self.training:
            labels = torch.cat(labels)  # Tensor(B * N,)

            loss_box_bnr = self.bnr_loss(box_embeddings, labels)
            loss_box_reid = self.reid_loss(box_identities, labels, normalized=True)

            losses = {'box_bnr': loss_box_bnr}
            if loss_box_reid is not None:
                losses['box_reid'] = loss_box_reid
            box_identities = None
        else:
            box_identities = box_identities.split(num_boxes_per_img)
            losses = None

        return box_identities, losses
