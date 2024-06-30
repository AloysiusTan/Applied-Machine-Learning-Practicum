#!/usr/bin/env python
# coding: utf-8

# In[5]:


from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "Photograph of a beautiful empty scene, highest quality"

init_image = Image.open("newBird.jpg").convert("RGB").resize((512,512))
mask_image = Image.open("newBird_maskImage.png").convert("RGB").resize((512,512))

image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
# image.save("bird1Removed.png")


# In[13]:


plt.imshow(image)

