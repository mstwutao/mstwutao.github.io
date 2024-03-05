---
title: "Learning Deep Representations via Contrastive Learning for Instance Retrieval"
authors:
- admin
- Tie Luo
- Donald Wunsch
#author_notes:
#- "Equal contribution"
#- "Equal contribution"
date: "2022-12-04"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "IEEE Symposium Series On Computational Intelligence (SSCI), 2022"
publication_short: ""

abstract: Instance-level Image Retrieval (IIR), or simply Instance Retrieval, deals with the problem of finding all the images within an dataset that contain a query instance (e.g. an object). This paper makes the first attempt that tackles this problem using instance-discrimination based contrastive learning (CL). While CL has shown impressive performance for many computer vision tasks, the similar success has never been found in the field of IIR. In this work, we approach this problem by exploring the capability of deriving discriminative representations from pre-trained and fine-tuned CL models. To begin with, we investigate the efficacy of transfer learning in IIR, by comparing off-the-shelf features learned by a pre-trained deep neural network (DNN) classifier with features learned by a CL model. The findings inspired us to propose a new training strategy that optimizes CL towards learning IIR-oriented features, by using an Average Precision (AP) loss together with a fine-tuning method to learn contrastive feature representations that are tailored to IIR. Our empirical evaluation demonstrates significant performance enhancement over the off-the-shelf features learned from a pre-trained DNN classifier on the challenging Oxford and Paris datasets.

# Summary. An optional shortened abstract.
summary: IEEE Symposium Series On Computational Intelligence (SSCI), 2022

#tags:
#- Source Themes
featured: true

# links:
# - name: ""
#   url: ""
url_pdf: https://arxiv.org/pdf/2209.13832.pdf
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ''
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
