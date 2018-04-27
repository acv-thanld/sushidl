import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from config import Config
import utils
import model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "/media/than/3E7019A6701965C5/nhandt_MaskRCNN_Nouth/mask_rcnn_sushi_0030.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SushiConfig(Config):
    """Configuration for training on the number dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "sushi"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 12  # Background + number

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1024
    TRAIN_ROIS_PER_IMAGE = 20

############################################################
#  Dataset
############################################################


class SushiDataset(utils.Dataset):

    def load_multi_number(self, dataset_dir, subset):
        """Load a subset of the number dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("sushi", 1, "Ikura")
        self.add_class("sushi", 2, "Maguro")
        self.add_class("sushi", 3, "Hamachi")
        self.add_class("sushi", 4, "Seen")
        self.add_class("sushi", 5, "Shouga")
        self.add_class("sushi", 6, "Ika")
        self.add_class("sushi", 7, "Ebi")
        self.add_class("sushi", 8, "Samon")
        self.add_class("sushi", 9, "Tamago")
        self.add_class("sushi", 10, "Saba")
        self.add_class("sushi", 11, "Tsuna")
        #self.add_class("sushi", 12, "leotrg")
        #self.add_class("sushi", 13, "luon") 
        #self.add_class("sushi", 14, "comcuon")
        #self.add_class("sushi", 15, "typec")
        #self.add_class("sushi", 16, "tom")
        #self.add_class("sushi", 17, "b")
        self.add_class("sushi", 12, "browl")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "/media/than/3E7019A6701965C5/nhandt_MaskRCNN_Nouth/datasets/sushi/train/via_region_data_Nouth15Images.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            # for b in a['regions'].values():
            #    polygons = [{**b['shape_attributes'], **b['region_attributes']}]
            # print("string=", polygons)
            # for r in a['regions'].values():
            #    polygons = [r['shape_attributes']]
            #    # print("polygons=", polygons)
            #    multi_numbers = [r['region_attributes']]
                # print("multi_numbers=", multi_numbers)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            sushis = [s['region_attributes'] for s in a['regions'].values()]
            # print("multi_numbers=", multi_numbers)
            # num_ids = [n for n in multi_numbers['number'].values()]
            # for n in multi_numbers:
            num_ids = [int(n['sushi']) for n in sushis]
            # print("num_ids=", num_ids)
            # print("num_ids_new=", num_ids_new)
            # categories = [s['region_attributes'] for s in a['regions'].values()]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "sushi",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a number dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "sushi":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # print("info['num_ids']=", info['num_ids'])
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sushi":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = SushiDataset()
    dataset_train.load_multi_number(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SushiDataset()
    dataset_val.load_multi_number(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect sushi.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/sushi/dataset/",
                        help='Directory of the sushi dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SushiConfig()
    else:
        class InferenceConfig(SushiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
        # Thêm dòng này để lưu model có cả cấu trúc mạng
        # model.keras_model.save("model.h5")
    else:
        print("'{}' is not recognized. "
              "Use 'train' ".format(args.command))
