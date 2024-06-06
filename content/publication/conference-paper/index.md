---
title: "Learning Deep Representations via Contrastive Learning for Instance Retrieval"

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Tie Luo
  - Donald Wunsch

# Author notes (optional)
# author_notes:
#   - 'Equal contribution'
#   - 'Equal contribution'

date: '2022-12-04'
doi: ''

# Schedule page publish date (NOT publication's date).
#publishDate: '2017-01-01T00:00:00Z'

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ['paper-conference']

# Publication name and optional abbreviated publication name.
publication: IEEE Symposium Series On Computational Intelligence (SSCI), 2022
publication_short: In *ICW*

abstract: Instance-level Image Retrieval (IIR), or simply Instance Retrieval, deals with the problem of finding all the images within an dataset that contain a query instance (e.g. an object). This paper makes the first attempt that tackles this problem using instance-discrimination based contrastive learning (CL). While CL has shown impressive performance for many computer vision tasks, the similar success has never been found in the field of IIR. In this work, we approach this problem by exploring the capability of deriving discriminative representations from pre-trained and fine-tuned CL models. To begin with, we investigate the efficacy of transfer learning in IIR, by comparing off-the-shelf features learned by a pre-trained deep neural network (DNN) classifier with features learned by a CL model. The findings inspired us to propose a new training strategy that optimizes CL towards learning IIR-oriented features, by using an Average Precision (AP) loss together with a fine-tuning method to learn contrastive feature representations that are tailored to IIR. Our empirical evaluation demonstrates significant performance enhancement over the off-the-shelf features learned from a pre-trained DNN classifier on the challenging Oxford and Paris datasets.

# Summary. An optional shortened abstract.
summary: IEEE Symposium Series On Computational Intelligence (SSCI), 2022

tags:
  - Contrastive Learning
  - Instance retrieval

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: ''
#url_code: 'https://github.com/HugoBlox/hugo-blox-builder'
#url_dataset: 'https://github.com/HugoBlox/hugo-blox-builder'
url_poster: ''
url_project: ''
url_slides: ''
#url_source: 'https://github.com/HugoBlox/hugo-blox-builder'
#url_video: 'https://youtube.com'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  #caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/pLCdAaMFLTE)'
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
# projects:
#   - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
# slides: example
---

<!-- {{% callout note %}}
Click the _Cite_ button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the _Slides_ button to check out the example.
{{% /callout %}}

Add the publication's **full text** or **supplementary notes** here. You can use rich formatting such as including [code, math, and images](https://docs.hugoblox.com/content/writing-markdown-latex/). -->
