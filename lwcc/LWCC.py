import torch

from .models import Bay, CSRNet, DMCount, SFANet
from .util.functions import load_image


def load_model(model_name="CSRNet", model_weights="SHA", device=None):
    """
    Builds a model for Crowd Counting and initializes it as a singleton.
    :param model_name: One of the available models: CSRNet.
    :param model_weights: Name of the dataset the model was pretrained on. Possible values vary on the model.
    :param device: Device to place the model on ('cuda', 'cpu', or None for CPU default).
    :return: Built Crowd Counting model initialized with pretrained weights.
    """

    available_models = {
        "CSRNet": CSRNet,
        "SFANet": SFANet,
        "Bay": Bay,
        "DM-Count": DMCount,
    }

    global loaded_models

    if "loaded_models" not in globals():
        loaded_models = {}

    # Determine device
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    model_full_name = "{}_{}_{}".format(model_name, model_weights, device.type)
    if model_full_name not in loaded_models.keys():
        model = available_models.get(model_name)
        if model:
            model = model.make_model(model_weights)
            model.to(device)  # Move model to specified device
            loaded_models[model_full_name] = model
            print(
                "Built model {} with weights {} on device {}".format(
                    model_name, model_weights, device
                )
            )
        else:
            raise ValueError(
                "Invalid model_name. Model {} is not available.".format(model_name)
            )

    return loaded_models[model_full_name]


def get_count(
    img_paths,
    model_name="CSRNet",
    model_weights="SHA",
    model=None,
    is_gray=False,
    return_density=False,
    resize_img=True,
    device=None,
    batch_size=None,
):
    """
    Return the count on image/s. You can use already loaded model or choose the name and pre-trained weights.
    :param img_paths: Either String (path to the image) or a list of strings (paths).
    :param model_name: If not using preloaded model, choose the model name. Default: "CSRNet".
    :param model_weights: If not using preloaded model, choose the model weights.  Default: "SHA".
    :param model: Possible preloaded model. Default: None.
    :param is_gray: Are the input images grayscale? Default: False.
    :param return_density: Return the predicted density maps for input? Default: False.
    :param resize_img: Should images with high resolution be down-scaled? This is especially good for high resolution
            images with relatively few people. For very dense crowds, False is recommended. Default: True
    :param device: Device to use for computation ('cuda', 'cpu', or None for CPU default).
    :param batch_size: Maximum number of images to process at once (useful for GPU memory management).
                      If None, all images are processed together.
    :return: Depends on whether the input is a String or list and on the return_density flag.
        If input is a String, the output is a float with the predicted count.
        If input is a list, the output is a dictionary with image names as keys, and predicted counts (float) as values.
        If return_density is True, function returns a tuple (predicted_count, density_map).
        If return_density is True and input is a list, function returns a tuple (count_dictionary, density_dictionary).
    """

    # if one path to array
    if not isinstance(img_paths, list):
        img_paths = [img_paths]

    # determine device
    if device not in [None, "cpu", "cuda"]:
        raise ValueError("Invalid device. Use 'cpu', 'cuda', or None for CPU default.")
    if device is None:
        device = torch.device("cpu")  # Default to CPU for backward compatibility
    else:
        device = torch.device(device)
        # Check if CUDA is available when requested
        if device.type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    # load model
    if model is None:
        model = load_model(model_name, model_weights, device)
    else:
        # If model is provided, ensure it's on the correct device
        model.to(device)

    # load images
    imgs, names = [], []

    for img_path in img_paths:
        img, name = load_image(img_path, model.get_name(), is_gray, resize_img)
        imgs.append(img)
        names.append(name)

    # Process images in batches if batch_size is specified
    if batch_size is not None and len(imgs) > batch_size:
        all_outputs = []
        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i : i + batch_size]
            batch_tensor = torch.cat(batch_imgs).to(device)

            with torch.no_grad():
                batch_outputs = model(batch_tensor)

            all_outputs.append(batch_outputs.cpu())

            # Clear GPU memory after each batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

        outputs = torch.cat(all_outputs).to(device)
    else:
        imgs_tensor = torch.cat(imgs).to(device)

        with torch.no_grad():
            outputs = model(imgs_tensor)

    counts = torch.sum(outputs, (1, 2, 3)).cpu().numpy()
    counts = dict(zip(names, [float(count) for count in counts]))

    densities = dict(zip(names, outputs[:, 0, :, :].cpu().numpy()))

    if len(counts) == 1:
        if return_density:
            return counts[name], densities[name]
        else:
            return counts[name]

    if return_density:
        return counts, densities

    return counts
