import torch

from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module()
class WHOverlaps2D(object):
    """WH IoU Calculator"""

    def __call__(self, bboxes1, bboxes2):
        """Calculate WH IoU between 2D bboxes

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 2) in <w, h> format.
            bboxes2 (Tensor): bboxes have shape (n, 2) in <w, h> format, or be
                empty.
        Returns:
            ious(Tensor): shape (m, n)
        """
        assert bboxes1.size(-1) in [0, 2]
        assert bboxes2.size(-1) in [0, 2]
        return wh_overlaps(bboxes1, bboxes2)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def wh_overlaps(bboxes1, bboxes2):
    """Calculate wh overlap between two set of bboxes

    Args:
        bboxes1 (Tensor): shape (m, 2) in <w, h> format or empty.
        bboxes2 (Tensor): shape (n, 2) in <w, h> format or empty.

    Returns:
        ious (Tensor): shape (m, n)
    """

    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 2 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 2 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, cols)

    wh1 = bboxes1[:, :2]
    wh2 = bboxes2[:, :2]

    wh1 = wh1[:, None]  # (m, 1, 2)
    wh2 = wh2[None]  # (1, n , 2)

    inter = torch.min(wh1, wh2).prod(2)

    ious = inter / (wh1.prod(2) + wh2.prod(2) - inter)

    return ious
