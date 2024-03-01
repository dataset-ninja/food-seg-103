
The authors built a new food image dataset **FoodSeg103 Dataset** containing 9,490 images. They annotate these images with 103 ingredient classes and each image has an average of 6 ingredient labels and pixel-wise masks. This dataset encompasses 7,118 images of Western cuisine, meticulously annotated with 103 distinct ingredient classes along with their corresponding segmentation masks.

## Motivation

In recent years, food computing has garnered increasing attention from the public, serving as the backbone for advancements in food and health-related research and applications. A key objective within food computing is the automated recognition of various types of food and the profiling of their nutritional and caloric values. In the realm of computer vision, related endeavors encompass dish classification, recipe generation, and food image retrieval. However, the majority of these efforts focus on representing and analyzing food images as a whole entity, without explicitly localizing or classifying the individual ingredients comprising the cooked dish.

This distinction leads to two primary tasks: food image classification, which addresses the holistic identification of food items, and food image segmentation, which delves into the localization and classification of individual ingredients within the food image. Of the two, food image segmentation presents a greater level of complexity as it endeavors to discern each ingredient category and its respective pixel-wise locations within the food image. For instance, in an image depicting a "hamburger," a proficient segmentation model would need to accurately delineate and mask out components like "beef," "tomato," "lettuce," "onion," and the "bread roll."

Compared to semantic segmentation tasks involving general object images, food image segmentation poses heightened challenges due to the vast diversity in food appearances and the often imbalanced distribution of ingredient categories. Firstly, ingredients cooked in various manners can exhibit significant visual disparities, complicating their identification. Additionally, certain ingredients may bear striking resemblances; for instance, "pineapples" cooked with meat might closely resemble "potatoes" cooked with meat, posing a challenge for differentiation. Secondly, food datasets commonly grapple with imbalanced distributions, both in terms of overall food classes and individual ingredient categories. This disparity arises due to two primary factors: Firstly, a few popular food classes dominate a large portion of food images, while the majority of food classes remain less represented. Secondly, a selection bias may exist in the construction of food image collections, further exacerbating the imbalance in class distribution.

<img src="https://github.com/dataset-ninja/food-seg-103/assets/120389559/deeed149-ebed-49f7-8660-2e39a6b63ae1" alt="image" width="600">

<span style="font-size: smaller; font-style: italic;">The first row shows a source image and its segmentation masks on our FoodSeg103. The second row shows example images to reveal the difficulties of food image segmentation, e.g., the pineapples in (a) and (b) look different, while the pineapple in (a) and the potato in \(c\) look quite similar.</span>

## Dataset description

To enable precise fine-grained food image segmentation, the authors have developed a comprehensive dataset known as FoodSeg103. This dataset encompasses 7,118 images of Western cuisine, meticulously annotated with 103 distinct ingredient classes along with their corresponding segmentation masks. The annotation process involved a meticulous approach, entailing careful data selection and iterative refinement of labels and masks to ensure the highest quality annotations. However, it's worth noting that the annotation process undertaken by the authors was both resource-intensive and time-consuming. The source images utilized in constructing the FoodSeg103 dataset were sourced from an existing food dataset known as [Recipe1M](http://pic2recipe.csail.mit.edu/), which boasts millions of images and accompanying cooking recipes. These recipes not only detail the cooking instructions but also specify the ingredients used. Leveraging this auxiliary information, the authors incorporated recipe details into the training process of semantic segmentation models, thereby enhancing the model's understanding of food composition and facilitating more accurate segmentation. 

FoodSeg103 serves as a subset of the broader FoodSeg154 dataset, which encompasses an additional subset dedicated to Asian cuisine images and annotations. In dataset, the authors meticulously curated 7,118 images, defining 103 distinct ingredient categories and providing corresponding category labels along with segmentation masks. Additionally, within FoodSeg154, the authors collected a supplementary set comprising 2,372 images showcasing diverse Asian culinary offerings. This subset boasts a greater variety compared to the Western food images present in FoodSeg103. The authors specifically utilize this subset to assess the domain adaptation capabilities of their food image segmentation models. While FoodSeg103 is made publicly available to support research endeavors, the subset containing Asian food images cannot be released to the public at this time due to confidentiality constraints associated with the images.

<img src="https://github.com/dataset-ninja/food-seg-103/assets/120389559/087a4456-59af-459a-8808-d6d83859599a" alt="image" width="1000">

<span style="font-size: smaller; font-style: italic;">Foodseg103 examples: source images (left) and annotations (right).</span>

The authors utilized FoodSeg103 as a case study to elucidate the process of constructing the dataset. They detailed the origins of the images, the compilation of ingredient categories, and the selection of images as follows:
* **Image Source:** The authors sourced images from Recipe1M, a dataset comprising 900k images featuring cooking instructions and ingredient labels. This dataset serves various purposes such as food image retrieval and recipe generation tasks.
* **Categories:** Initially, the authors surveyed the frequency of all ingredient categories within Recipe1M. Despite the dataset containing around 1.5k ingredient categories, many proved challenging to mask out from images effectively. Consequently, the authors streamlined the categories to retain only the top 124 (later refined to 103) ingredients. Any ingredients not falling under these categories were assigned to the *other ingredients* category.
* **Image Selection:** Within each fine-grained ingredient category, the authors sampled images from Recipe1M based on two criteria: 1) Each image should feature at least two ingredients, either of the same or different categories, with a maximum of 16 ingredients per image; and 2) The ingredients must be clearly visible and easily annotatable within the images. Following this selection process, the authors obtained 7,118 images for annotation with segmentation masks.

<img src="https://github.com/dataset-ninja/food-seg-103/assets/120389559/26dc5fe9-c8de-49c8-b3a9-f9ba24260be7" alt="image" width="1000">

<span style="font-size: smaller; font-style: italic;">More annotation examples of FoodSeg103. The source images are in the left hand, while the annotation masks are in the right hand.</span>

The subsequent phase involves annotating segmentation masks, which entails delineating polygons to cover the pixel-wise locations of various ingredients. This process comprises two main stages: annotation and refinement.
* **Annotation:** The authors enlisted the assistance of a data annotation company to undertake the meticulous task of mask annotation. Each image was meticulously examined by a human annotator, who initially identified the ingredient categories present, assigned the appropriate category label to each ingredient, and delineated the pixel-wise mask accordingly. Annotators were instructed to disregard minuscule image regions, even if they contained some ingredients, if their area covered less than 5% of the entire image.
* **Refinement:** Upon receiving all the masks from the annotation company, the authors proceeded with an extensive refinement process. This involved adhering to three primary refinement criteria: 1) rectifying any mislabeled data; 2) eliminating unpopular category labels assigned to fewer than 5 images; and 3) consolidating visually similar ingredient categories, such as merging "orange" and "citrus." Following refinement, the initial set of 125 ingredient categories was streamlined to 103. The annotation and refinement endeavors spanned approximately one year.

<img src="https://github.com/dataset-ninja/food-seg-103/assets/120389559/6a343da5-68a3-4ed4-af4c-849127730878" alt="image" width="600">

<span style="font-size: smaller; font-style: italic;">Examples of dataset refinement. (a) sources images (b) before refinement (wrong or confusing labels exist), and \(c\) after refinement.</span>

<img src="https://github.com/dataset-ninja/food-seg-103/assets/120389559/a2a7220b-4af4-4937-ad8f-cd64b582c8c2" alt="image" width="1000">




