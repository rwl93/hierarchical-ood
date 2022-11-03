import argparse
import pandas as pd
import torch
import json
from PIL import Image
from io import BytesIO
import base64


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hierarchy', metavar="FILENAME",
                        default='pruned-wn.pth',
                        help='Hierarchy pth filename')
    parser.add_argument('--outfile', metavar="FILENAME",
                        default='pruned-wn.json',
                        help='Hierarchy json output filename ' +
                             '(will be overwritten)')
    parser.add_argument('--outfile-html', metavar="FILENAME",
                        default='rdendro.html',
                        help='Radial dendrogram html output filename ' +
                             '(will be overwritten)')
    parser.add_argument('--ood-labels', metavar="FILENAME",
                        default=None,
                        help='Out-of-distribution csv label map')
    parser.add_argument('--fig-size', metavar="INT",
                        default=1100, type=int,
                        help="Size of the figure to produce")
    parser.add_argument('--has-root', action='store_true',
                        help="Hierarchy includes root. Do not add root node")
    parser.add_argument('--no-radial', action='store_true',
                        help="Use standard tree instead of radial")
    parser.add_argument('--image', default=None,
                        help="Sample image to include")
    parser.add_argument('--image-scale', metavar="FLOAT",
                        default=1., type=float,
                        help="Factor to scale image by")
    parser.add_argument('--pred-wnid', metavar="WNID",
                        default=None, type=str,
                        help="Predicted node wordnet id")
    parser.add_argument('--gt-wnid', metavar="WNID",
                        default=None, type=str,
                        help="GT node wordnet id")
    parser.add_argument('--img-caption', metavar="CAPTION",
                        default=None, type=str,
                        help="Image sample caption")
    parser.add_argument('--gt-desc', metavar="CAPTION",
                        default=None, type=str,
                        help="Groundtruth description")
    parser.add_argument('--pred-desc', metavar="CAPTION",
                        default=None, type=str,
                        help="Prediction description")
    parser.add_argument('--gt-dist', metavar="CAPTION",
                        default=None, type=str,
                        help="Groundtruth distance")
    parser.add_argument('--pred-dist', metavar="CAPTION",
                        default=None, type=str,
                        help="Prediction distance")
    return parser


def gen_subtree(wnid, idx, hierarchy, lvl=0, ood_wnids=None,
                gt_wnid=None, pred_wnid=None):
    my_children = []
    for k, v in hierarchy['class_parents'].items():
        if len(v) > 0 and v[-1] == idx:
            my_children.append(gen_subtree(k, hierarchy['class_list'].index(k),
                                           hierarchy, lvl+1, ood_wnids,
                                           gt_wnid=gt_wnid, pred_wnid=pred_wnid))
    allchildrenOOD = (all([c['ood'] for c in my_children]) and
                      len(my_children) > 0)
    isOOD = ((ood_wnids is not None) and
             ((wnid in ood_wnids) or (allchildrenOOD)))
    isGT  = (gt_wnid == wnid) 
    isPred  = (pred_wnid == wnid) 
    subtree = {
        "name": hierarchy['class_description'][wnid],
        "children": my_children,
        "colname": "level"+str(lvl),
        "ood": isOOD,
        "pred": isPred,
        "gt": isGT,
    }
    return subtree


def del_empty_children(subtree):
    for c in subtree['children']:
        del_empty_children(c)
    if len(subtree['children']) == 0:
        del subtree['children']


def main():
    parser = get_parser()
    args = parser.parse_args()

    hierarchy = torch.load(args.hierarchy)
    # Add root node to hierarchy for json generation
    if args.has_root:
        for k,v in hierarchy['child2parent'].items():
            if v is None:
                root_wnid = k
        root_idx = hierarchy['class_list'].index(root_wnid)
    else:
        hierarchy['class_description']['ROOT'] = 'Physical object'
        root_wnid = 'ROOT'
        for k, v in hierarchy['class_parents'].items():
            if len(v) == 0:
                v.append(-1)
        root_idx = -1
    if args.ood_labels:
        ood_wnids = pd.read_csv(args.ood_labels, header=None)[0].to_list()
    else:
        ood_wnids = None
    tree = gen_subtree(root_wnid, root_idx, hierarchy, lvl=0,
                       ood_wnids=ood_wnids,
                       gt_wnid=args.gt_wnid,
                       pred_wnid=args.pred_wnid,
                       )
    del_empty_children(tree)
    if args.image is not None:
        image = Image.open(args.image)
        buffered = BytesIO()
        image.save(buffered, format="jpeg")
        imageb64 = base64.b64encode(buffered.getvalue())
        imagehref = f"data:image/jpeg;base64,{imageb64.decode('utf-8')}"
        image_height, image_width = image.size
        tree["image"] = {
            "href": imagehref,
            "width": image_width * args.image_scale,
            "height": image_height * args.image_scale
        }
    with open(args.outfile, 'w') as f:
        json.dump(tree, f, sort_keys=True, indent=2)
    if args.no_radial:
        with open('d3code/tree-template.html', 'r') as f:
            html = (
                f.read()
                .replace("CONFIG_TREE_DATA", json.dumps([tree]))
                .replace("CONFIG_MARGIN_TOP", str(50))
                .replace("CONFIG_MARGIN_LEFT", str(100))
                .replace("CONFIG_SCALE", str(1))
                .replace("CONFIG_VIS_WIDTH", str(750))
                .replace("CONFIG_VIS_HEIGHT", str(400))
                .replace("CONFIG_ZOOM", str(1))
                .replace("CONFIG_BELOW_DY", str(10))
                .replace("CONFIG_TEXT_SIZE", str(50))
            )
            if args.img_caption is not None:
                html = (html.replace("CONFIG_IMG_CAPTION", args.img_caption)
                            .replace("CONFIG_GT_DESC", args.gt_desc)
                            .replace("CONFIG_GT_DIST", args.gt_dist)
                            .replace("CONFIG_PRED_DESC", args.pred_desc)
                            .replace("CONFIG_PRED_DIST", args.pred_dist)
                            .replace("CONFIG_CAPTION", "true"))
            else:
                html = html.replace("CONFIG_CAPTION", "false")
    else:
        with open('d3code/radial_dendrogram_template.html', 'r') as f:
            html = (
                f.read()
                .replace("CONFIG_TREE_DATA", json.dumps([tree]))
                .replace("CONFIG_SIZE", str(args.fig_size))
            )
    with open(args.outfile_html, 'w') as f:
        f.write(html)


if __name__ == "__main__":
    main()
