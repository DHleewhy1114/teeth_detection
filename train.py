import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import scipy


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize
# Import COCO config
#matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR=os.path.join(ROOT_DIR,"./logs")




class TeethConfig(Config):
    NAME="teeth"
    NUM_CLASSES = 1+6 #background + class 
    STEPS_PER_EPOCH = 150




class TeethDataset(utils.Dataset):
    def load_teeth(self,dataset_dir,subset):
        
        self.add_class("teeth",1,"1")
        self.add_class("teeth",2,"2")
        self.add_class("teeth",3,"3")
        self.add_class("teeth",4,"4")
        self.add_class("teeth",5,"5")
        self.add_class("teeth",6,"6")
        
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        annotations = json.load(open(os.path.join(dataset_dir,"via_region_data.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
            #print(polygons)
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            #class_ids=int(a['regions'])['shape_attributes']
            classes = [s['region_attributes'] for s in a['regions']]
            #print("classes")
            #print(classes)
            #class_ids used in prefare() methods, change to num_ids
            print(classes)
            num_ids=[int(n['name']) for n in classes]
            #num_ids=[int(n['class']) for n in classes]
            #print("class_ids")
            #print(num_ids)
            
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            #print(image.shape[:2])
            height, width = image.shape[:2]
            #print (image_path)
            self.add_image(
                "teeth",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)
            #print (self.image_info)
        
    def load_mask(self,image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "teeth":
            return super(self.__class__, self).load_mask(image_id)
        
        #print(image_info['num_ids'])
        num_ids=image_info['num_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        #exist_return = mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        #exist_return = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #            dtype=np.uint8)
        #mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids
        #mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        #mask, num_ids.astype(np.int32)
        #, num_ids
        #, num_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "teeth":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TeethDataset()
    dataset_train.load_teeth(args.dataset, "train")
    dataset_train.prepare()
    #for i, info in enumerate(dataset_train.class_info):
    #    print("{:3}. {:50}".format(i, info['name']))
    # Validation dataset
    dataset_val = TeethDataset()
    dataset_val.load_teeth(args.dataset, "val")
    dataset_val.prepare()
    #for i, info in enumerate(dataset_val.class_info):
    #    print("{:3}. {:50}".format(i, info['name']))

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
    
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash
    
def detect_and_color_splash(model, image_path=None):
    print("Running on {}".format(args.image))
    # Read image
    image = skimage.io.imread(args.image)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    #print (r)
    print (r['rois'])
    print (r['masks'])
    print (r['class_ids'])
    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    dataset=TeethDataset()
    print ("classinfo")
    print (dataset.class_info)
    #print (dataset.class_info['name'])
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)
    image = scipy.misc.imread(file_name)
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],["1","2","3","4","5","6"])
                            

        
    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect teeth.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TeethConfig()
    else:
        class InferenceConfig(TeethConfig):
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
        weights_path = model.find_last()
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
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))