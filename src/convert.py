import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.imaging.color import get_predefined_colors
from supervisely.io.fs import get_file_name, get_file_size
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    batch_size = 30

    pre_ds_to_path = {
        "train": "/home/alex/DATASETS/TODO/FoodSeg103/Images/img_dir/train",
        "test": "/home/alex/DATASETS/TODO/FoodSeg103/Images/img_dir/test",
    }

    classes_path = "/home/alex/DATASETS/TODO/FoodSeg103/category_id.txt"

    images_folder = "img_dir"
    masks_folder = "ann_dir"

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        mask_path = image_path.replace(images_folder, masks_folder).replace(".jpg", ".png")

        mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
        pixels = np.unique(mask_np)
        for pixel in pixels[1:]:
            obj_class = meta.get_obj_class(pixel_to_class[pixel])
            super_meta = class_to_super.get(obj_class.name)
            super_tag = sly.Tag(super_meta)
            mask = mask_np == pixel
            curr_bitmap = sly.Bitmap(mask)
            curr_label = sly.Label(curr_bitmap, obj_class, tags=[super_tag])
            labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    pixel_to_class = {}
    with open(classes_path) as f:
        content = f.read().split("\n")
        for row in content:
            if len(row) > 1:
                pixel, class_name = row.split("\t")
                if pixel == "0":
                    continue
                # obj_class = sly.ObjClass(
                #     class_name.lstrip().lower(), sly.Bitmap)
                pixel_to_class[int(pixel)] = class_name.lstrip().lower()

    obj_classes = [
        sly.ObjClass(name, sly.Bitmap, color)
        for name, color in zip(list(pixel_to_class.values()), get_predefined_colors(103))
    ]

    vegetable_meta = sly.TagMeta("vegetable", sly.TagValueType.NONE)
    fruit_meta = sly.TagMeta("fruit", sly.TagValueType.NONE)
    main_meta = sly.TagMeta("main", sly.TagValueType.NONE)
    dessert_meta = sly.TagMeta("dessert", sly.TagValueType.NONE)
    nut_meta = sly.TagMeta("nut", sly.TagValueType.NONE)
    meat_meta = sly.TagMeta("meat", sly.TagValueType.NONE)
    beverage_meta = sly.TagMeta("beverage", sly.TagValueType.NONE)
    fungus_meta = sly.TagMeta("fungus", sly.TagValueType.NONE)
    seafood_meta = sly.TagMeta("seafood", sly.TagValueType.NONE)
    egg_meta = sly.TagMeta("egg", sly.TagValueType.NONE)
    sauce_meta = sly.TagMeta("sauce", sly.TagValueType.NONE)
    soup_meta = sly.TagMeta("soup", sly.TagValueType.NONE)
    tofu_meta = sly.TagMeta("tofu", sly.TagValueType.NONE)
    salad_meta = sly.TagMeta("salad", sly.TagValueType.NONE)
    other_meta = sly.TagMeta("other ingredients", sly.TagValueType.NONE)

    meta = sly.ProjectMeta(
        obj_classes=obj_classes,
        tag_metas=[
            vegetable_meta,
            fruit_meta,
            main_meta,
            dessert_meta,
            nut_meta,
            meat_meta,
            beverage_meta,
            fungus_meta,
            seafood_meta,
            egg_meta,
            sauce_meta,
            soup_meta,
            tofu_meta,
            salad_meta,
            other_meta,
        ],
    )

    class_to_super = {
        "candy": dessert_meta,
        "egg tart": dessert_meta,
        "french fries": dessert_meta,
        "chocolate": dessert_meta,
        "biscuit": dessert_meta,
        "popcorn": dessert_meta,
        "pudding": dessert_meta,
        "ice cream": dessert_meta,
        "cheese butter": dessert_meta,
        "cake": dessert_meta,
        "wine": beverage_meta,
        "milkshake": beverage_meta,
        "coffee": beverage_meta,
        "juice": beverage_meta,
        "milk": beverage_meta,
        "tea": beverage_meta,
        "almond": nut_meta,
        "red beans": nut_meta,
        "cashew": nut_meta,
        "dried cranberries": nut_meta,
        "soy": nut_meta,
        "walnut": nut_meta,
        "peanut": nut_meta,
        "egg": egg_meta,
        "apple": fruit_meta,
        "date": fruit_meta,
        "apricot": fruit_meta,
        "avocado": fruit_meta,
        "banana": fruit_meta,
        "strawberry": fruit_meta,
        "cherry": fruit_meta,
        "blueberry": fruit_meta,
        "raspberry": fruit_meta,
        "mango": fruit_meta,
        "olives": fruit_meta,
        "peach": fruit_meta,
        "lemon": fruit_meta,
        "pear": fruit_meta,
        "fig": fruit_meta,
        "pineapple": fruit_meta,
        "grape": fruit_meta,
        "kiwi": fruit_meta,
        "melon": fruit_meta,
        "orange": fruit_meta,
        "watermelon": fruit_meta,
        "steak": meat_meta,
        "pork": meat_meta,
        "chicken duck": meat_meta,
        "sausage": meat_meta,
        "fried meat": meat_meta,
        "lamb": meat_meta,
        "sauce": sauce_meta,
        "crab": seafood_meta,
        "fish": seafood_meta,
        "shellfish": seafood_meta,
        "shrimp": seafood_meta,
        "soup": soup_meta,
        "bread": main_meta,
        "corn": main_meta,
        "hamburg": main_meta,
        "pizza": main_meta,
        "hanamaki baozi": main_meta,
        "wonton dumplings": main_meta,
        "pasta": main_meta,
        "noodles": main_meta,
        "rice": main_meta,
        "pie": main_meta,
        "tofu": tofu_meta,
        "eggplant": vegetable_meta,
        "potato": vegetable_meta,
        "garlic": vegetable_meta,
        "cauliflower": vegetable_meta,
        "tomato": vegetable_meta,
        "kelp": vegetable_meta,
        "seaweed": vegetable_meta,
        "spring onion": vegetable_meta,
        "rape": vegetable_meta,
        "ginger": vegetable_meta,
        "okra": vegetable_meta,
        "lettuce": vegetable_meta,
        "pumpkin": vegetable_meta,
        "cucumber": vegetable_meta,
        "white radish": vegetable_meta,
        "carrot": vegetable_meta,
        "asparagus": vegetable_meta,
        "bamboo shoots": vegetable_meta,
        "broccoli": vegetable_meta,
        "celery stick": vegetable_meta,
        "cilantro mint": vegetable_meta,
        "snow peas": vegetable_meta,
        "cabbage": vegetable_meta,
        "bean sprouts": vegetable_meta,
        "onion": vegetable_meta,
        "pepper": vegetable_meta,
        "green beans": vegetable_meta,
        "french beans": vegetable_meta,
        "king oyster mushroom": fungus_meta,
        "shiitake": fungus_meta,
        "enoki mushroom": fungus_meta,
        "oyster mushroom": fungus_meta,
        "white button mushroom": fungus_meta,
        "salad": salad_meta,
        "other ingredients": other_meta,
    }

    api.project.update_meta(project.id, meta.to_json())

    for ds_name, curr_ds_path in pre_ds_to_path.items():

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        images_names = os.listdir(curr_ds_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(curr_ds_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
