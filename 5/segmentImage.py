#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load MaskFormer fine-tuned on COCO panoptic segmentation
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

image = Image.open("origImage.jpg")
inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to feature_extractor for postprocessing
result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
predicted_panoptic_map = result["segmentation"]


# In[3]:


result


# In[4]:


result["segmentation"]


# In[5]:


np.unique(result["segmentation"])


# In[6]:


plt.imshow(image)


# In[7]:


plt.imshow(result["segmentation"]==3)


# In[8]:


# Extract the mask corresponding to the bird 
predicted_panoptic_map = predicted_panoptic_map.numpy()
bird_mask = (predicted_panoptic_map == 3).astype(np.uint8) * 255  # Convert to binary mask


# In[9]:


# Save the mask as an image
mask_image = Image.fromarray(bird_mask)
mask_image.save("maskImage.png")

