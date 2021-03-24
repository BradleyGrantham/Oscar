# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
"""Extract the features needed to use Oscar."""
from __future__ import absolute_import, division, print_function
import os
from io import BytesIO
from itertools import zip_longest
from argparse import Namespace
import base64
import os.path as op
import random
from pprint import pprint

import numpy as np
import torch
from PIL import Image, ImageOps
from loguru import logger

from oscar.utils.logger import setup_logger
from oscar.utils.misc import set_seed
from oscar.modeling.modeling_bert import BertForImageCaptioning
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.data import MetadataCatalog
from torchvision.ops import nms
from detectron2.structures import Boxes, Instances


IMAGE_PATH = "/home/ubuntu/my-images/68b51467-19a3-4b41-ad51-c7054ebf84ae.JPG"
PACKAGE_LOCATION = "/home/ubuntu/py-bottom-up-attention"
D2_ROOT = op.join(
    PACKAGE_LOCATION, "detectron2/model_zoo/"
)  # Root of detectron2
MIN_BOXES = 36
MAX_BOXES = 36


def convert_image_to_b64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    return encoded_string


def convert_b64_to_image(s):
    r = base64.b64decode(s)
    file_like = BytesIO(r)
    im = Image.open(file_like)
    im = ImageOps.exif_transpose(im)
    # im.save("/home/ubuntu/image.jpg", "JPEG")
    im = np.array(im)[:, :, ::-1]
    return im


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def fast_rcnn_inference_single_image(
        boxes,
        scores,
        image_shape,
        score_thresh,
        nms_thresh,
        topk_per_image,
        cuda=True,
):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Select max scores
    max_scores, max_classes = scores.max(1)  # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    if cuda:
        torcharange = torch.arange(num_objs).cuda()
    else:
        torcharange = torch.arange(num_objs)

    idxs = torcharange * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]  # Select max boxes according to the max scores.

    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = max_classes[keep]

    return result, keep


def doit(detector, raw_images, cuda=True):
    with torch.no_grad():
        # Preprocessing
        inputs = []
        for raw_image in raw_images:
            image = detector.transform_gen.get_transform(
                raw_image
            ).apply_image(raw_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append(
                {
                    "image": image,
                    "height": raw_image.shape[0],
                    "width": raw_image.shape[1],
                }
            )
        images = detector.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = detector.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = detector.model.proposal_generator(
            images, features, None
        )

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in detector.model.roi_heads.in_features]
        box_features = detector.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(
            dim=[2, 3]
        )  # (sum_proposals, 2048), pooled to 1x1

        # Predict classes and boxes for each proposal.
        (
            pred_class_logits,
            pred_proposal_deltas,
        ) = detector.model.roi_heads.box_predictor(feature_pooled)
        rcnn_outputs = FastRCNNOutputs(
            detector.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            detector.model.roi_heads.smooth_l1_beta,
        )

        # Fixed-number NMS
        instances_list, ids_list = [], []
        probs_list = rcnn_outputs.predict_probs()
        boxes_list = rcnn_outputs.predict_boxes()
        for probs, boxes, image_size in zip(
                probs_list, boxes_list, images.image_sizes
        ):
            for nms_thresh in np.arange(0.3, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes,
                    probs,
                    image_size,
                    score_thresh=0.2,
                    nms_thresh=nms_thresh,
                    topk_per_image=MAX_BOXES,
                    cuda=cuda,
                )
                if len(ids) >= MIN_BOXES:
                    break
            instances_list.append(instances)
            ids_list.append(ids)

        # Post processing for features
        features_list = feature_pooled.split(
            rcnn_outputs.num_preds_per_image
        )  # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        roi_features_list = []
        for ids, features in zip(ids_list, features_list):
            roi_features_list.append(features[ids].detach())

        # Post processing for bounding boxes (rescale to raw_image)
        raw_instances_list = []
        for instances, input_per_image, image_size in zip(
                instances_list, inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            # we pass the height and width in as 1 so we get proportional heights and widths rather than raw numbers
            raw_instances = detector_postprocess(instances, 1, 1)
            raw_instances_list.append(raw_instances)

        return raw_instances_list, roi_features_list


def load_vg_classes():
    # Load VG Classes
    data_path = os.path.join(PACKAGE_LOCATION, "demo/data/genome/1600-400-20")

    vg_classes = []
    with open(os.path.join(data_path, "objects_vocab.txt")) as f:
        for object in f.readlines():
            vg_classes.append(object.split(",")[0].lower().strip())

        MetadataCatalog.get("vg").thing_classes = vg_classes
    class_names = MetadataCatalog.get("vg").as_dict()["thing_classes"]

    return class_names


def create_embeddings(detector, pathXid, cuda=True):
    imgs, img_ids = zip(*pathXid)

    instances_list, features_list = doit(detector, imgs, cuda)

    class_names = load_vg_classes()

    for img, image_id, instances, features in zip(imgs, img_ids, instances_list, features_list):

        boxes = instances.pred_boxes.tensor.to("cpu").numpy()
        preds = instances.scores.to("cpu").numpy()
        clses = instances.pred_classes.to("cpu").numpy()
        num_bboxes = boxes.shape[0]
        boxes_with_height_and_width = np.concatenate(
            (
                boxes,
                (boxes[:, 2] - boxes[:, 0])[:, np.newaxis],
                (boxes[:, 3] - boxes[:, 1])[:, np.newaxis],
            ),
            axis=1,
        )

        resnet_embeddings = features.to("cpu").numpy()

        embeddings_for_oscar = np.concatenate(
            (resnet_embeddings, boxes_with_height_and_width), axis=1
        )

    embeddings_for_oscar = torch.from_numpy(embeddings_for_oscar).to("cpu")
    if cuda:
        embeddings_for_oscar = embeddings_for_oscar.cuda()

    return embeddings_for_oscar, " ".join([class_names[id_] for id_ in clses])


def load_image_ids(img_root):
    """images in the same directory are in the same split"""
    paths_and_ids = []
    for name in os.listdir(img_root):
        idx = name.split(".")[0]
        paths_and_ids.append((os.path.join(img_root, name), idx))
    return paths_and_ids


def build_d2_model(cuda=False):
    """Build model and load weights for vg only."""
    cfg = get_cfg()  # Renew the cfg file
    cfg.merge_from_file(
        os.path.join(
            D2_ROOT,
            "configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml",
        )
    )
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
    cfg.MODEL.WEIGHTS = (
        "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
    )

    if not cuda:
        cfg.DEVICE = "cpu"
        cfg.MODEL.DEVICE = "cpu"
    detector = DefaultPredictor(cfg)
    return detector


def extract_oscar_features(image_string, cuda, d2_model):
    image = convert_b64_to_image(image_string)
    a, b = create_embeddings(
        d2_model, [(image, 0)], cuda
    )
    return a, b


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
            max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
            is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, text_a, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len)) # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1 
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                        (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention 
        # for caption as caption will have full attention on image. 
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start : c_end, c_start : c_end].copy_(self._triangle_mask[0 : seq_a_len, 0 : seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start : l_end, l_start : l_end] = 1
        attention_mask[r_start : r_end, r_start : r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start : c_end, l_start : l_end] = 1
        attention_mask[c_start : c_end, r_start : r_end] = 1
        # full attention for L-R:
        attention_mask[l_start : l_end, r_start : r_end] = 1
        attention_mask[r_start : r_end, l_start : l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        return input_ids.cuda(), attention_mask.cuda(), segment_ids.cuda(), img_feat.cuda(), masked_pos.cuda()


def test(args, image_features, object_tags, model, tokenizer):
    args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids( [tokenizer.cls_token, 
            tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token, '.']
        )
    model.eval()

    tensorizer = CaptionTensorizer(tokenizer, 50, 50, 20, 0.15, 3, is_train=False)
    input_ids, attention_mask, segment_ids, img_feat, masked_pos = tensorizer.tensorize_example(
        text_a="", img_feat=image_features, text_b=object_tags
    )

    with torch.no_grad():
        inputs = {'is_decode': True,
                  'input_ids': input_ids[None, :], 'attention_mask': attention_mask[None, :, :],
                  'token_type_ids': segment_ids[None, :], 'img_feats': img_feat[None, :, :],
                  'masked_pos': masked_pos[None, :],
                  'do_sample': False,
                  'bos_token_id': cls_token_id,
                  'pad_token_id': pad_token_id,
                  'eos_token_ids': [sep_token_id, pad_token_id],
                  'mask_token_id': mask_token_id,
                  # for adding od labels
                  'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

                  # hyperparameters of beam search
                  'max_length': args.max_gen_length,
                  'num_beams': args.num_beams,
                  "temperature": args.temperature,
                  "top_k": args.top_k,
                  "top_p": args.top_p,
                  "repetition_penalty": args.repetition_penalty,
                  "length_penalty": args.length_penalty,
                  "num_return_sequences": args.num_return_sequences,
                  "num_keep_best": 8,
                  }
        pprint(inputs)
        # captions, logprobs
        outputs = model(**inputs)
        all_caps = outputs[0]  # batch_size * num_keep_best * max_len
        all_confs = torch.exp(outputs[1])

        captions = []
        for cap, conf in zip(all_caps[0], all_confs[0]):
            captions.append(
                {"caption": tokenizer.decode(cap.tolist(), skip_special_tokens=True),
                 "conf": conf.item()
                 }
            )
        return captions


def restore_training_settings(args):
    assert not args.do_train
    assert args.do_test or args.do_eval
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
                max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
            'max_img_seq_length', 'img_feature_dim',
            'img_feature_type']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def load_models(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    args.seed = 88

    set_seed(args.seed, args.n_gpu)

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    checkpoint = args.eval_model_dir
    assert op.isdir(checkpoint)
    config = config_class.from_pretrained(checkpoint)
    config.output_hidden_states = args.output_hidden_states
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    logger.info("Evaluate the following checkpoint: %s", checkpoint)
    model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)

    args = restore_training_settings(args)

    return args, model, tokenizer


def get_captions(args, image_string, model_, tokenizer_, d2_model):

    img_features, object_ids = extract_oscar_features(image_string, cuda=False if args.no_cuda else True, d2_model=d2_model)

    captions = test(args, img_features.to("cpu"), object_ids, model_, tokenizer_)

    pprint(captions)
    return captions


if __name__ == "__main__":
    my_args = Namespace(
        model_name_or_path="/home/ubuntu/Oscar/models/checkpoint-29-66420/",
        loss_type="sfmx",
        config_name="",
        tokenizer_name="",
        max_seq_length=70,
        max_seq_a_length=40,
        do_train=False,
        do_test=True,
        do_eval=True,
        do_lower_case=True,
        mask_prob=0.15,
        max_masked_tokens=3,
        add_od_labels=True,
        drop_out=0.1,
        max_img_seq_length=50,
        img_feature_dim=2054,
        img_feature_type="frcnn",
        per_gpu_train_batch_size=1,
        per_gpu_eval_batch_size=1,
        output_mode="classification",
        num_labels=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        weight_decay=0.05,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        warmup_steps=0,
        scheduler="linear",
        num_workers=4,
        num_train_epochs=40,
        max_steps=-1,
        logging_steps=20,
        save_steps=-1,
        evaluate_during_training=False,
        no_cuda=False,
        seed=88,
        scst=False,
        eval_model_dir="/home/ubuntu/Oscar/models/checkpoint-29-66420/",
        max_gen_length=20,
        output_hidden_states=False,
        num_return_sequences=1,
        num_beams=5,
        num_keep_best=8,
        temperature=1,
        top_k=0,
        top_p=1,
        repetition_penalty=1,
        length_penalty=1,
        use_cbs=False,
        min_constraints_to_satisfy=2
    )

    my_args, model, tokenizer = load_models(my_args)
    d2_model = build_d2_model(not my_args.no_cuda)
    b = convert_image_to_b64(IMAGE_PATH)
    get_captions(my_args, b, model, tokenizer, d2_model)
