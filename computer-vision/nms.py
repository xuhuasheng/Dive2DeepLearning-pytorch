

def nms(detections, threshold=0.5):
    """
    purpose: 非极大抑制
    inputs: detections: list of dict {"bbox": (x, y, w, h), "class": cls, "conf_score": score}
            threshold: default is 0.5
    """
    # 判断非空
    if len(detections) == 0:
        return []
    # 0.预筛查：删除置信度过低的box，减小计算量
    detections = list(filter(lambda x: x >= 0.1, detections))
    # 1.按照置信度从大到小排序
    detections = sorted(detections, key=lambda x: x["conf_score"], reverse=True)
    # 定义NMS后的检测框结果
    nms_detections = []
    # 直到detections为空（其中元素要么被取为基准，要么作为冗余被删除）
    while len(detections):
        # 2.取当前置信度最高的box作为基准
        nms_detections.append(detection.pop(0))
        # 3.遍历后续box与基准box的IoU，若大于阈值，则视为当前基准box的冗余，被删除
        for i, det in enumerate(detections):
            if get_IoU(det, nms_detections[-1]) > threshold:
                detections.pop(i)

def get_IoU(bbox1, bbox2):
    b1_tl_x = bbox1[0] - bbox1[2] / 2
    b1_tl_y = bbox1[1] - bbox1[3] / 2
    b1_br_x = bbox1[0] + bbox1[2] / 2
    b1_br_y = bbox1[1] + bbox1[3] / 2
    b2_tl_x = bbox2[0] - bbox2[2] / 2
    b2_tl_y = bbox2[1] - bbox2[3] / 2
    b2_br_x = bbox2[0] + bbox2[2] / 2
    b2_br_y = bbox2[1] + bbox2[3] / 2

    overlap_w = max(0, min(b1_br_x, b2_br_x) - max(b1_tl_x, b2_tl_x))
    overlap_h = max(0, min(b1_br_y, b2_br_y) - max(b1_tl_y, b2_tl_y))
    overlap_area = overlap_w * overlap_h
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    total_area = area1 + area2 - overlap_area
    return overlap_area / total_area